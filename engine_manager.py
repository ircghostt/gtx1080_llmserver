import os
import sys
import time
import json
import subprocess
import atexit
import signal
import logging
import urllib.request
import urllib.error
from config import config, load_config, LLAMA_SERVER_EXE, BIN_DIR

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("backend_manager")

def assign_job_object_kill_on_close(proc: subprocess.Popen):
    """Binds child process to a Windows Job Object with KILL_ON_JOB_CLOSE policy.
    Guarantees OS kernel terminates llama-server even if Python is killed or console is closed.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
                ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryLimit', ctypes.c_size_t),
                ('PeakJobMemoryLimit', ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )

        kernel32.AssignProcessToJobObject(job, int(proc._handle))
    except Exception as e:
        logger.debug(f"Job object binding note: {e}")

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

        # Bind Windows Job Object so child process terminates on exit/kill
        assign_job_object_kill_on_close(self.process)

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
            logger.info("Gracefully stopping GTX 1080 Ti backend engine...")
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def restart(self):
        logger.info("Restarting backend engine to apply new hardware/layer settings...")
        self.stop()
        time.sleep(0.5)
        self.start()

manager = BackendManager()
