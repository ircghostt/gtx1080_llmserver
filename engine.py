import os
import sys
import time
import logging
from typing import Iterator, Dict, Any, List, Optional
from config import config

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("gtx1080_engine")

class LLMEngine:
    def __init__(self):
        self.llm = None
        self.model_path = config.model_path
        self.model_name = config.model_name
        self._load_model()

    def _load_model(self):
        logger.info(f"Checking model at: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info(f"Initializing Llama engine on GTX 1080 Ti...")
        logger.info(f"Parameters: GPU Layers={config.n_gpu_layers}, Context={config.n_ctx}, Batch={config.n_batch}, MMQ={config.use_mmq}")

        from llama_cpp import Llama

        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
            n_batch=config.n_batch,
            n_threads=config.n_threads,
            use_mmap=True,
            flash_attn=False,
            verbose=True
        )
        logger.info("Model loaded successfully into GTX 1080 Ti VRAM.")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stream: bool = False
    ) -> Any:
        if max_tokens is None:
            max_tokens = 2048

        return self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stream=stream
        )

    def completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Any:
        return self.llm.create_completion(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            stream=stream
        )

# Global engine singleton
engine: Optional[LLMEngine] = None

def get_engine() -> LLMEngine:
    global engine
    if engine is None:
        engine = LLMEngine()
    return engine
