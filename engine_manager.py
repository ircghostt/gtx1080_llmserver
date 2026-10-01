import os
import time
import subprocess
import atexit
import logging
import urllib.request
import urllib.error
from config import config, load_config, LLAMA_SERVER_EXE, BIN_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("backend_manager")

class BackendManager:
    def __init__(self):
        self.process = None
        self.backend_url = f"http://{config.backend_host}:{config.backend_port}"

    def start(self):
        # Refresh config in case config.json was modified
        current_cfg = load_config()

        if not os.path.exists(current_cfg.model_path):
            raise FileNotFoundError(f"Model file does not exist at: {current_cfg.model_path}")

        if not os.path.exists(LLAMA_SERVER_EXE):
            raise FileNotFoundError(f"llama-server executable not found at: {LLAMA_SERVER_EXE}")

        cmd = [
            LLAMA_SERVER_EXE,
            "-m", current_cfg.model_path,
            "-ngl", str(current_cfg.n_gpu_layers),
            "-c", str(current_cfg.n_ctx),
            "-b", str(current_cfg.n_batch),
            "-t", str(current_cfg.n_threads),
            "--host", current_cfg.backend_host,
            "--port", str(current_cfg.backend_port),
            "--alias", current_cfg.model_alias
        ]

        if current_cfg.mmproj_file and os.path.exists(current_cfg.mmproj_file):
            logger.info(f"Multimodal vision projector detected: {current_cfg.mmproj_file}")
            cmd.extend(["--mmproj", current_cfg.mmproj_file])
        else:
            logger.info("Running in standard text mode (no mmproj projector attached).")

        logger.info(f"Starting GTX 1080 Ti hardware engine on port {current_cfg.backend_port} with {current_cfg.n_gpu_layers} GPU layers...")
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

    def restart(self):
        logger.info("Restarting backend engine to apply new hardware/layer settings...")
        self.stop()
        time.sleep(0.5)
        self.start()

manager = BackendManager()
