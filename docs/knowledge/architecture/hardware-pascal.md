---
id: server.architecture.hardware-pascal
title: Pascal GPU & Legacy CPU Hardware Architecture
version: 1.0.0
tags:
  - cpu/avx1
  - gpu/cuda
  - gpu/pascal
  - hardware/gtx1080ti
description: Technical details on Pascal SM 6.1 CUDA compute optimizations and Sandy Bridge Non-AVX2 CPU dynamic feature dispatching.
references:
  - server.architecture.index
  - server.architecture.dual-layer-gateway
---

# Pascal GPU & Legacy CPU Architecture

## 1. NVIDIA GTX 1080 Ti Optimization (Pascal SM 6.1)
The NVIDIA GeForce GTX 1080 Ti provides 11,264 MB VRAM with 3584 CUDA cores based on the Pascal architecture (`sm_61`).

- **No Native FP16 Tensor Cores**: Pascal lacks modern FP16 matrix tensor units (introduced in Volta/Turing).
- **MMQ (Matrix Multiplication via Quantization)**: Standard cuBLAS relies on FP16 tensor core paths. Enabling MMQ evaluates quantized integer weights directly on standard FP32/INT8 CUDA ALUs, boosting token generation speed to ~40–100 tok/s for ~3B–4B models.
- **FlashAttention Incompatibility**: FlashAttention v2 requires SM 7.0+. FlashAttention is disabled to avoid segmentation faults on Pascal.

## 2. Legacy CPU Compatibility (Intel Sandy Bridge / Non-AVX2)
Legacy Intel 2nd Generation (Sandy Bridge / Ivy Bridge) CPUs lack the AVX2 instruction set.

- **Dynamic Dispatch Solution**: Bundled modular binaries (`bin/ggml-cpu-sandybridge.dll`, `bin/ggml-cpu-sse42.dll`) dynamically detect host CPU instructions at runtime, avoiding `0xc000001d (ILLEGAL_INSTRUCTION)` fatal exceptions.
