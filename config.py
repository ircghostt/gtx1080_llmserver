import os
import re
import json
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_JSON_PATH = os.path.join(BASE_DIR, "config.json")
BIN_DIR = os.path.join(BASE_DIR, "bin")
LLAMA_SERVER_EXE = os.path.join(BIN_DIR, "llama-server.exe")

class ServerConfig(BaseModel):
    # Server Gateway & Ports
    host: str = "0.0.0.0"
    port: int = 8000
    backend_host: str = "127.0.0.1"
    backend_port: int = 8081

    # Model configuration
    model_path: str = os.path.join(BASE_DIR, "models", "gemma-3n-E4B-it-Q4_K_M.gguf")
    model_alias: str = "gemma-3n-e4b"
    mmproj_file: str = ""
    system_prompt_file: str = os.path.join(BASE_DIR, "system_prompt.txt")
    system_prompt: str = ""

    # Hardware Tuning for GTX 1080 Ti
    n_gpu_layers: int = 99
    n_ctx: int = 8192
    n_batch: int = 512
    n_threads: int = 4

    # Sampling Defaults
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 2048

def sanitize_json_content(raw_str: str) -> dict:
    """Parses JSON safely, repairing unescaped Windows backslashes if present."""
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        # Auto-escape single Windows backslashes that are not valid JSON escape sequences
        repaired = re.sub(r'\\(?![/\\ntbrf"u])', r'\\\\', raw_str)
        return json.loads(repaired)

def load_config() -> ServerConfig:
    cfg_data = {}
    if os.path.exists(CONFIG_JSON_PATH):
        try:
            with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
                raw = f.read()
            cfg_data = sanitize_json_content(raw)
        except Exception as e:
            print(f"Warning: Failed to parse config.json: {e}")

    srv = cfg_data.get("server_network", cfg_data.get("server", {}))
    mdl = cfg_data.get("model_settings", cfg_data.get("model", {}))
    hw = cfg_data.get("hardware_gtx1080_settings", cfg_data.get("hardware", {}))
    smp = cfg_data.get("generation_defaults", cfg_data.get("sampling", {}))

    model_file = mdl.get("model_file", "models/gemma-3n-E4B-it-Q4_K_M.gguf")
    if not os.path.isabs(model_file):
        model_file = os.path.normpath(os.path.join(BASE_DIR, model_file))

    mmproj_file = mdl.get("vision_projector_file", mdl.get("mmproj_file", ""))
    if mmproj_file and not os.path.isabs(mmproj_file):
        mmproj_file = os.path.normpath(os.path.join(BASE_DIR, mmproj_file))

    sys_prompt_file = mdl.get("system_prompt_file", "system_prompt.txt")
    if not os.path.isabs(sys_prompt_file):
        sys_prompt_file = os.path.normpath(os.path.join(BASE_DIR, sys_prompt_file))

    sys_prompt_content = ""
    if os.path.exists(sys_prompt_file):
        try:
            with open(sys_prompt_file, "r", encoding="utf-8") as f:
                sys_prompt_content = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read system prompt file: {e}")
    else:
        print(f"Warning: Specified system prompt file does not exist: {sys_prompt_file}")

    main_port = srv.get("openai_compatible_api_port", srv.get("chat_interface_port", srv.get("port", 8000)))
    backend_port = srv.get("internal_gpu_engine_port", srv.get("backend_port", 8081))

    return ServerConfig(
        host=srv.get("host", "0.0.0.0"),
        port=int(main_port),
        backend_host="127.0.0.1",
        backend_port=int(backend_port),
        model_path=model_file,
        model_alias=mdl.get("model_display_name", mdl.get("model_alias", "gemma-3n-e4b")),
        mmproj_file=mmproj_file,
        system_prompt_file=sys_prompt_file,
        system_prompt=sys_prompt_content,
        n_gpu_layers=int(hw.get("gpu_layers_offload", hw.get("n_gpu_layers", 99))),
        n_ctx=int(hw.get("context_size_tokens", hw.get("n_ctx", 8192))),
        n_batch=int(hw.get("prompt_batch_size", hw.get("n_batch", 512))),
        n_threads=int(hw.get("cpu_worker_threads", hw.get("n_threads", 4))),
        temperature=float(smp.get("temperature", 0.7)),
        top_p=float(smp.get("top_p", 0.95)),
        top_k=int(smp.get("top_k", 40)),
        max_tokens=int(smp.get("max_tokens_to_generate", smp.get("max_tokens", 2048)))
    )

config = load_config()
