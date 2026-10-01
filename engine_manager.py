import os
import time
import subprocess
import atexit
import logging
import urllib.request
import urllib.error
from config import config, LLAMA_SERVER_EXE, BIN_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("backend_manager")

class BackendManager:
    def __init__(self):
        self.process = None
        self.backend_url = f"http://{config.backend_host}:{config.backend_port}"

    def start(self):
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model file does not exist at: {config.model_path}")

        if not os.path.exists(LLAMA_SERVER_EXE):
            raise FileNotFoundError(f"llama-server executable not found at: {LLAMA_SERVER_EXE}")

        cmd = [
            LLAMA_SERVER_EXE,
            "-m", config.model_path,
            "-ngl", str(config.n_gpu_layers),
            "-c", str(config.n_ctx),
            "-b", str(config.n_batch),
            "-t", str(config.n_threads),
            "--host", config.backend_host,
            "--port", str(config.backend_port),
            "--alias", config.model_alias
        ]

        if config.mmproj_file and os.path.exists(config.mmproj_file):
            logger.info(f"Multimodal vision projector detected: {config.mmproj_file}")
            cmd.extend(["--mmproj", config.mmproj_file])
        else:
            logger.info("Running in standard text mode (no mmproj projector attached).")

        logger.info(f"Starting GTX 1080 Ti hardware engine on port {config.backend_port}...")
        logger.info(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["PATH"] = BIN_DIR + os.pathsep + env.get("PATH", "")

        self.process = subprocess.Popen(
            cmd,
            cwd=BIN_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        atexit.register(self.stop)
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: int = 45):
        logger.info("Waiting for model to load into GTX 1080 Ti VRAM...")
        start_time = time.time()
        health_url = f"{self.backend_url}/health"

        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                raise RuntimeError("Backend engine process terminated unexpectedly during startup.")
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        logger.info("GTX 1080 Ti engine is ready and active!")
                        return True
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                time.sleep(1)

        raise TimeoutError("Backend failed to start within timeout.")

    def stop(self):
        if self.process and self.process.poll() is None:
            logger.info("Shutting down backend process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

manager = BackendManager()
