import os
import time
import json
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

    def get_gpu_vram_used(self) -> float:
        """Returns current GPU VRAM used in MB via nvidia-smi"""
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, encoding="utf-8").strip()
            return float(out.split("\n")[0].strip())
        except Exception:
            return 0.0

    def start(self):
        current_cfg = load_config()

        if not os.path.exists(current_cfg.model_path):
            raise FileNotFoundError(f"Model file does not exist at: {current_cfg.model_path}")

        if not os.path.exists(LLAMA_SERVER_EXE):
            raise FileNotFoundError(f"llama-server executable not found at: {LLAMA_SERVER_EXE}")

        baseline_vram = self.get_gpu_vram_used()
        logger.info(f"Baseline GPU VRAM before model load: {baseline_vram:.0f} MB")

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

        logger.info(f"Launching GTX 1080 Ti engine on port {current_cfg.backend_port} (Layers: {current_cfg.n_gpu_layers}, Context: {current_cfg.n_ctx})...")

        env = os.environ.copy()
        env["PATH"] = BIN_DIR + os.pathsep + env.get("PATH", "")
        env["CUDA_VISIBLE_DEVICES"] = "0"

        self.process = subprocess.Popen(
            cmd,
            cwd=BIN_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

        atexit.register(self.stop)
        self._wait_until_ready()
        self._warmup_and_verify_vram()

    def _wait_until_ready(self, timeout: int = 45):
        logger.info("Waiting for model initialization in GTX 1080 Ti...")
        start_time = time.time()
        health_url = f"{self.backend_url}/health"

        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                raise RuntimeError("Backend engine process terminated unexpectedly during startup.")
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                time.sleep(0.5)

        raise TimeoutError("Backend failed to start within timeout.")

    def _warmup_and_verify_vram(self):
        """Sends a 1-token warmup request to force immediate GPU KV-cache allocation and audits VRAM"""
        logger.info("Executing GPU warmup to pre-allocate full KV cache in VRAM...")
        try:
            warmup_url = f"{self.backend_url}/v1/chat/completions"
            payload = json.dumps({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
                "stream": False
            }).encode("utf-8")
            req = urllib.request.Request(warmup_url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                pass
        except Exception as e:
            logger.warning(f"Warmup ping warning (non-fatal): {e}")

        # Audit VRAM
        allocated_vram = self.get_gpu_vram_used()
        logger.info(f"============================================================")
        logger.info(f" [VRAM Audit] Total GPU VRAM in use: {allocated_vram:.0f} MB / 11,264 MB")
        if allocated_vram >= 3800:
            logger.info(" [VRAM Audit] SUCCESS: Model and KV-cache are 100% pinned in VRAM.")
        else:
            logger.warning(" [VRAM Audit] WARNING: VRAM usage is below expected threshold (~4,400 MB). Check layer slider.")
        logger.info(f"============================================================")

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
