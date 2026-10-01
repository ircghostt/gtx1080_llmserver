# ⚡ GTX 1080 Ti Dedicated Local LLM Server

A production-grade, OpenAI-compatible local LLM server tailored specifically for the **NVIDIA GeForce GTX 1080 Ti (11GB VRAM, Pascal SM 6.1)** and legacy **Intel/AMD CPUs without AVX2 (Sandy Bridge / Ivy Bridge)**.

[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-green.svg)](https://platform.openai.com/docs/api-reference)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4%20(SM%206.1)-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ✨ Features

- **🚀 100% GPU Offload on GTX 1080 Ti**: Full layers offloaded directly into the 11 GB VRAM via optimized CUDA 12.4 MMQ integer arithmetic (~40–100+ tokens/sec).
- **🛡️ Zero AVX2 Crashes**: Dynamically dispatches CPU instructions, preventing `0xc000001d (ILLEGAL_INSTRUCTION)` on Intel Sandy Bridge / 2nd-gen processors.
- **🔌 Full OpenAI API Compliance**: Drop-in replacement for `/v1/chat/completions` (with Server-Sent Events SSE streaming), `/v1/completions`, and `/v1/models`.
- **⏹️ Stop Generation Support**: Web UI and backend support instant cancellation via `AbortController`, freeing GPU cycles immediately.
- **📷 Photo / Image Ready**: Vision attachment support in Web UI and base64 OpenAI multimodal payload format.
- **💾 Live System Prompt Management**: Edit and save instructions directly to `system_prompt.txt` from the Web UI.
- **📊 Real-time GPU Telemetry**: Live VRAM allocation, temperature, and GPU load monitoring via `nvidia-smi`.
- **⚡ 1-Click Startup**: Launch with double-clicking `start_server.bat`.

---

## 🛠️ Quick Start

### 1. Prerequisites
- **GPU**: NVIDIA GeForce GTX 1080 Ti (or any Pascal/Turing/Ampere/Ada GPU)
- **NVIDIA Driver**: 550.x or newer
- **Python**: Python 3.10 or 3.11 (64-bit)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/your-username/gtx1080_llmserver.git
cd gtx1080_llmserver

# Setup Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Add a Model
Place your `.gguf` model file inside the `models/` folder:
```
models/
└── gemma-3n-E4B-it-Q4_K_M.gguf
```

### 4. Run the Server
Double-click **`start_server.bat`** or run:
```powershell
python server.py
```

- **Web Chat UI**: Open [http://localhost:8000](http://localhost:8000)
- **OpenAI API Base URL**: `http://localhost:8000/v1`
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚙️ Configuration (`config.json`)

Easily customize ports, hardware offloading, and sampling parameters:

```json
{
  "server_network": {
    "host": "0.0.0.0",
    "chat_interface_port": 8000,
    "openai_compatible_api_port": 8000,
    "internal_gpu_engine_port": 8081
  },
  "model_settings": {
    "model_file": "models/gemma-3n-E4B-it-Q4_K_M.gguf",
    "model_display_name": "gemma-3n-e4b",
    "vision_projector_file": "models/mmproj.gguf",
    "system_prompt_file": "system_prompt.txt"
  },
  "hardware_gtx1080_settings": {
    "gpu_layers_offload": 99,
    "context_size_tokens": 8192,
    "prompt_batch_size": 512,
    "cpu_worker_threads": 4
  },
  "generation_defaults": {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens_to_generate": 2048
  }
}
```

---

## 💻 SDK & API Integration

### Python (Official `openai` Library)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gemma-3n-e4b",
    messages=[
        {"role": "system", "content": "You are a concise expert AI assistant."},
        {"role": "user", "content": "Explain relativity in one sentence."}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)
```

### cURL
```powershell
curl http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": false}'
```

---

## 📚 Open Knowledge Format (OKF) Documentation
The full authoritative specification is available in [`docs/knowledge/index.md`](docs/knowledge/index.md):
- **Hardware Architecture**: [`docs/knowledge/architecture/hardware-pascal.md`](docs/knowledge/architecture/hardware-pascal.md)
- **Gateway Architecture**: [`docs/knowledge/architecture/dual-layer-gateway.md`](docs/knowledge/architecture/dual-layer-gateway.md)
- **OpenAPI Specs**: [`docs/knowledge/api/openai-endpoints.md`](docs/knowledge/api/openai-endpoints.md)
- **Configuration Specs**: [`docs/knowledge/configuration/system-config.md`](docs/knowledge/configuration/system-config.md)

---

## 📄 License
Released under the [MIT License](LICENSE).
