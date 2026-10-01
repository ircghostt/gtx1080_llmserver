---
id: server.api.openai-endpoints
title: OpenAI API Compatible Endpoints Specification
version: 1.0.0
tags:
  - api/chat
  - api/completions
  - api/models
  - api/openai
  - api/rest
description: Authoritative OpenAPI schema definitions and parameter mapping for the GTX 1080 Ti server.
references:
  - server.api.index
  - server.architecture.dual-layer-gateway
schema_ref: openapi/spec.yaml
schema_metadata:
  type: openapi
  version: 3.0.0
  base_path: /
  endpoints:
    - path: /v1/chat/completions
      method: POST
      description: Creates a completion for the provided chat conversation messages.
      parameters:
        - name: messages
          type: array
          required: true
          description: List of message objects with role and content.
        - name: model
          type: string
          required: false
          description: Model identifier or alias.
        - name: stream
          type: boolean
          required: false
          description: If true, streams partial message deltas via Server-Sent Events.
        - name: temperature
          type: number
          required: false
          description: Sampling temperature between 0 and 2.
      responses:
        200:
          type: object
          description: Chat completion response or SSE stream.
    - path: /v1/models
      method: GET
      description: Lists currently loaded model and compatibility aliases.
      responses:
        200:
          type: object
          description: List of available model objects.
    - path: /api/save-system-prompt
      method: POST
      description: Overwrites system_prompt.txt and updates runtime memory.
      parameters:
        - name: system_prompt
          type: string
          required: true
          description: New system prompt string.
      responses:
        200:
          type: object
          description: Status confirmation.
    - path: /api/gpu-stats
      method: GET
      description: Returns real-time GPU VRAM, temperature, and utilization percentages.
      responses:
        200:
          type: object
          description: Live GPU telemetry object.
---

# OpenAI API Compatible Endpoints

## 1. Chat Completions (`POST /v1/chat/completions`)
Standard OpenAI endpoint supporting streaming SSE deltas (`data: {...}\n\ndata: [DONE]\n\n`) and standard non-streaming payloads.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## 2. Models List (`GET /v1/models`)
Returns active loaded models and default aliases (`gemma-3n-e4b`, `gpt-4`, `gpt-3.5-turbo`).

## 3. Telemetry (`GET /api/gpu-stats`)
Returns live metrics collected from `nvidia-smi` including GPU temperature, VRAM usage in megabytes, and GPU core load percentage.
