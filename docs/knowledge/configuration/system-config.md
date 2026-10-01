---
id: server.configuration.system-config
title: System Configuration & Parameter Specification
version: 1.0.0
tags:
  - config/hardware
  - config/json
  - config/sampling
  - config/server
description: Field-by-field reference for config.json, system_prompt.txt, and hardware layer offloading.
references:
  - server.configuration.index
  - server.architecture.hardware-pascal
---

# System Configuration Specification

## 1. `config.json` Field Reference

| Section | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `server_network` | `host` | String | `"0.0.0.0"` | Network interface bind address. |
| `server_network` | `chat_interface_port` | Integer | `8000` | Port for the Web UI and public gateway. |
| `server_network` | `openai_compatible_api_port` | Integer | `8000` | OpenAI REST API entry port. |
| `server_network` | `internal_gpu_engine_port` | Integer | `8081` | Loopback port for the C++ CUDA engine. |
| `model_settings` | `model_file` | String | `"models/gemma-3n-E4B-it-Q4_K_M.gguf"` | Relative or absolute path to `.gguf` file. |
| `model_settings` | `model_display_name` | String | `"gemma-3n-e4b"` | Model alias advertised in `/v1/models`. |
| `model_settings` | `vision_projector_file` | String | `"models/mmproj.gguf"` | Optional vision projector file path. |
| `model_settings` | `system_prompt_file` | String | `"system_prompt.txt"` | Text file containing default system instructions. |
| `hardware_gtx1080_settings` | `gpu_layers_offload` | Integer | `99` | Number of transformer layers offloaded to VRAM (`99` = 100%). |
| `hardware_gtx1080_settings` | `context_size_tokens` | Integer | `8192` | KV cache context window token limit. |
| `hardware_gtx1080_settings` | `prompt_batch_size` | Integer | `512` | Batch evaluation chunk size. |
| `hardware_gtx1080_settings` | `cpu_worker_threads` | Integer | `4` | Worker threads for auxiliary processing. |
| `generation_defaults` | `temperature` | Float | `0.7` | Randomness temperature for generation. |
| `generation_defaults` | `top_p` | Float | `0.95` | Nucleus sampling probability cutoff. |
| `generation_defaults` | `max_tokens_to_generate` | Integer | `2048` | Maximum completion tokens generated per request. |

## 2. `system_prompt.txt`
Contains raw plaintext system instructions. Injected automatically at the beginning of chat completions if the request contains no explicit system message.
