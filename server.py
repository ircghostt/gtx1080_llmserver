import os
import json
import time
import subprocess
import logging
import asyncio
import httpx
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import config, load_config, CONFIG_JSON_PATH
from engine_manager import manager

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gateway_server")

app = FastAPI(
    title="GTX 1080 Ti OpenAI-Compatible LLM Server",
    description="Production-grade OpenAI API server with full GTX 1080 Ti hardware acceleration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class SaveConfigRequest(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    system_prompt: Optional[str] = None
    gpu_layers: Optional[int] = None
    context_size: Optional[int] = None

@app.on_event("startup")
def on_startup():
    logger.info("Initializing backend engine with config.json settings...")
    manager.start()
    logger.info("GTX 1080 Ti LLM Gateway ready.")

@app.on_event("shutdown")
def on_shutdown():
    logger.info("Stopping backend engine...")
    manager.stop()

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>GTX 1080 Ti Server is Active.</h2>")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": config.model_alias,
        "hardware": "NVIDIA GeForce GTX 1080 Ti (11GB)",
        "context_window": config.n_ctx
    }

@app.get("/api/config")
def get_current_config():
    """Returns current active configuration for UI and clients"""
    current_cfg = load_config()
    has_vision = bool(current_cfg.mmproj_file and os.path.exists(current_cfg.mmproj_file))
    return {
        "model_alias": current_cfg.model_alias,
        "model_path": current_cfg.model_path,
        "system_prompt": current_cfg.system_prompt,
        "temperature": current_cfg.temperature,
        "top_p": current_cfg.top_p,
        "top_k": current_cfg.top_k,
        "max_tokens": current_cfg.max_tokens,
        "context_size": current_cfg.n_ctx,
        "gpu_layers": current_cfg.n_gpu_layers,
        "has_vision": has_vision
    }

@app.post("/api/save-config")
def save_config(payload: SaveConfigRequest):
    """Saves updated parameters to config.json and reloads model if hardware layers changed"""
    try:
        cfg_data = {}
        if os.path.exists(CONFIG_JSON_PATH):
            with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)

        if "generation_defaults" not in cfg_data:
            cfg_data["generation_defaults"] = {}
        if "hardware_gtx1080_settings" not in cfg_data:
            cfg_data["hardware_gtx1080_settings"] = {}

        hardware_changed = False

        if payload.temperature is not None:
            cfg_data["generation_defaults"]["temperature"] = payload.temperature
            config.temperature = payload.temperature

        if payload.max_tokens is not None:
            cfg_data["generation_defaults"]["max_tokens_to_generate"] = payload.max_tokens
            config.max_tokens = payload.max_tokens

        if payload.top_p is not None:
            cfg_data["generation_defaults"]["top_p"] = payload.top_p
            config.top_p = payload.top_p

        if payload.top_k is not None:
            cfg_data["generation_defaults"]["top_k"] = payload.top_k
            config.top_k = payload.top_k

        if payload.gpu_layers is not None:
            old_val = cfg_data["hardware_gtx1080_settings"].get("gpu_layers_offload")
            if old_val != payload.gpu_layers:
                cfg_data["hardware_gtx1080_settings"]["gpu_layers_offload"] = payload.gpu_layers
                config.n_gpu_layers = payload.gpu_layers
                hardware_changed = True

        if payload.context_size is not None:
            old_ctx = cfg_data["hardware_gtx1080_settings"].get("context_size_tokens")
            if old_ctx != payload.context_size:
                cfg_data["hardware_gtx1080_settings"]["context_size_tokens"] = payload.context_size
                config.n_ctx = payload.context_size
                hardware_changed = True

        with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)

        if payload.system_prompt is not None:
            with open(config.system_prompt_file, "w", encoding="utf-8") as f:
                f.write(payload.system_prompt.strip())
            config.system_prompt = payload.system_prompt.strip()

        if hardware_changed:
            logger.info("Hardware layer/context configuration changed. Triggering engine reload...")
            manager.restart()

        logger.info("Settings successfully written to config.json and memory.")
        return {
            "status": "success",
            "message": "Settings saved to config.json",
            "hardware_reloaded": hardware_changed,
            "gpu_layers": config.n_gpu_layers,
            "context_size": config.n_ctx
        }
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-system-prompt")
def save_system_prompt(payload: SaveConfigRequest):
    return save_config(payload)

@app.get("/api/gpu-stats")
def gpu_stats():
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, encoding="utf-8").strip()
        parts = [p.strip() for p in output.split(",")]
        return {
            "gpu_name": parts[0],
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
            "gpu_temp_c": int(parts[3]),
            "gpu_util_pct": int(parts[4]),
            "model_loaded": config.model_alias
        }
    except Exception as e:
        return {"gpu_name": "GTX 1080 Ti", "model_loaded": config.model_alias, "error": str(e)}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": config.model_alias,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gtx1080_server",
                "permission": [],
                "root": config.model_alias,
                "parent": None
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gtx1080_server",
                "permission": [],
                "root": config.model_alias,
                "parent": None
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gtx1080_server",
                "permission": [],
                "root": config.model_alias,
                "parent": None
            },
            {
                "id": "gpt-4-vision-preview",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "gtx1080_server",
                "permission": [],
                "root": config.model_alias,
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        body["model"] = config.model_alias

        messages = body.get("messages", [])
        has_system = any(m.get("role") == "system" for m in messages)
        if not has_system and config.system_prompt:
            messages.insert(0, {"role": "system", "content": config.system_prompt})
            body["messages"] = messages

        is_stream = body.get("stream", False)
        backend_url = f"{manager.backend_url}/v1/chat/completions"

        if is_stream:
            async def stream_response():
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream("POST", backend_url, json=body) as resp:
                            async for chunk in resp.aiter_bytes():
                                if await request.is_disconnected():
                                    logger.info("Client aborted request. Halting generation.")
                                    break
                                if chunk:
                                    yield chunk
                except Exception as ex:
                    logger.info(f"Stream finished/aborted: {ex}")

            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(backend_url, json=body)
                return JSONResponse(status_code=resp.status_code, content=resp.json())

    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": 500}}
        )

@app.post("/v1/completions")
async def text_completions(request: Request):
    try:
        body = await request.json()
        body["model"] = config.model_alias
        is_stream = body.get("stream", False)
        backend_url = f"{manager.backend_url}/v1/completions"

        if is_stream:
            async def stream_response():
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        async with client.stream("POST", backend_url, json=body) as resp:
                            async for chunk in resp.aiter_bytes():
                                if await request.is_disconnected():
                                    break
                                if chunk:
                                    yield chunk
                except Exception as ex:
                    logger.info(f"Stream finished/aborted: {ex}")

            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(backend_url, json=body)
                return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception as e:
        logger.error(f"Completion error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": 500}}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.host, port=config.port, reload=False)
