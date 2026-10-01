---
id: server.architecture.dual-layer-gateway
title: Dual-Layer Gateway & Process Lifecycle Architecture
version: 1.0.0
tags:
  - architecture/gateway
  - backend/llamacpp
  - network/fastapi
  - server/streaming
description: Architectural specification of the FastAPI proxy gateway interfacing with the modular C++ llama-server engine.
references:
  - server.architecture.index
  - server.architecture.hardware-pascal
  - server.api.openai-endpoints
---

# Dual-Layer Gateway Architecture

```mermaid
graph TD
    Client["Client / Web UI / OpenAI SDK"] -->|HTTP / SSE on Port 8000| Gateway["FastAPI Server Gateway (server.py)"]
    Gateway -->|Model Alias Normalization & System Prompt Injection| Proxy["Reverse Proxy & Client Disconnect Supervisor"]
    Proxy -->|Internal Loopback on Port 8081| Backend["llama-server C++ Engine (bin/llama-server.exe)"]
    Backend -->|CUDA 12.4 Direct Offload| GPU["GTX 1080 Ti (11GB VRAM)"]
```

## 1. Process Supervision (`engine_manager.py`)
- The Python gateway initializes `bin/llama-server.exe` as a supervised child process on loopback port `8081`.
- A blocking healthcheck polls `http://127.0.0.1:8081/health` until the model is 100% loaded into GPU VRAM before binding the public gateway on port `8000`.
- Graceful shutdown handles `SIGTERM` and `SIGKILL` on exit via Python `atexit` hooks.

## 2. Abort & Client Disconnect Handling
- When a user aborts via `AbortController` in the Web UI or closes an HTTP client connection, FastAPI's `request.is_disconnected()` triggers an immediate disconnect on the upstream socket to `llama-server.exe`.
- Token generation on the GPU halts immediately, preventing wasted compute cycles.
