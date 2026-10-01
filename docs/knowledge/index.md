---
id: server.bundle.root
title: GTX 1080 Ti Dedicated LLM Server Root Index
version: 1.0.0
tags:
  - architecture/overview
  - hardware/gtx1080ti
  - server/llm
description: Master entry point and directed acyclic graph for the GTX 1080 Ti dedicated OpenAI-compatible LLM server.
references:
  - server.architecture.dual-layer-gateway
  - server.architecture.hardware-pascal
  - server.api.openai-endpoints
  - server.configuration.system-config
---

# GTX 1080 Ti Dedicated LLM Server Knowledge Base

Welcome to the authoritative OKF documentation for the GTX 1080 Ti Dedicated LLM Server.

## Architecture & Hardware Compatibility
- [[server.architecture.hardware-pascal]]: Pascal SM 6.1 GPU acceleration and Sandy Bridge / Non-AVX2 CPU dynamic dispatching.
- [[server.architecture.dual-layer-gateway]]: High-level architecture separating FastAPI gateway proxy from the internal C++ CUDA backend engine.

## API Specifications
- [[server.api.openai-endpoints]]: Complete OpenAPI metadata and specifications for `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and telemetry APIs.

## Configuration & Runtime
- [[server.configuration.system-config]]: Comprehensive breakdown of `config.json`, `system_prompt.txt`, and hardware layer offloading.
