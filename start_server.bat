@echo off
title GTX 1080 Ti LLM Server (OpenAI Compatible)
cd /d "%~dp0"

echo ======================================================================
echo           GTX 1080 Ti Dedicated OpenAI-Compatible LLM Server
echo ======================================================================
echo.
echo [1/3] Activating Python Virtual Environment...
call venv\Scripts\activate.bat

echo [2/3] Verifying GPU and Model Configuration...
echo       Model: Gemma 3n E4B (GGUF)
echo       Target GPU: NVIDIA GeForce GTX 1080 Ti (11GB VRAM)
echo       Port: 8000
echo.
echo [3/3] Starting Gateway and Inference Engine...
echo       - Web Chat Interface: http://localhost:8000
echo       - OpenAI API Endpoint: http://localhost:8000/v1/chat/completions
echo       - API Documentation:   http://localhost:8000/docs
echo ======================================================================
echo.

python server.py

pause
