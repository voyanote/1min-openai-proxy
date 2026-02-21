#!/usr/bin/env python3
"""
1min.ai to OpenAI API Proxy Server
Converts OpenAI-compatible requests to 1min.ai format

Usage:
  - Set API key via Authorization header: "Bearer <your-1min-api-key>"
  - Or use default key from ONEMIN_API_KEY environment variable
"""

import asyncio
import json
import os
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="1min.ai OpenAI Proxy")

# 1min.ai API configuration
ONEMIN_API_URL = "https://api.1min.ai/api/features"
DEFAULT_API_KEY = os.environ.get("ONEMIN_API_KEY", "a2e6a2350d21850d0a8bcdc2ecfb6ad5e5fa2040457cfbd156a715dc978c5512")


def get_api_key(authorization: Optional[str] = None) -> str:
    """Extract API key from Authorization header or use default"""
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]
        return authorization
    return DEFAULT_API_KEY


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    max_completion_tokens: Optional[int] = None


def convert_to_1min_format(request: ChatCompletionRequest) -> dict:
    """Convert OpenAI request to 1min.ai format"""
    # Get the last user message as prompt
    prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            prompt = msg.content
            break
    
    # Build conversation context if multiple messages
    context = ""
    if len(request.messages) > 1:
        for msg in request.messages[:-1]:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            if msg.role == "system":
                role_prefix = "System"
            context += f"{role_prefix}: {msg.content}\n"
        if context:
            prompt = f"{context}\nUser: {prompt}"
    
    return {
        "type": "CHAT_WITH_AI",
        "model": request.model,
        "promptObject": {
            "prompt": prompt,
            "isMixed": False,
            "webSearch": False
        }
    }


async def stream_1min_response(request: ChatCompletionRequest, api_key: str) -> AsyncGenerator[str, None]:
    """Stream response from 1min.ai"""
    payload = convert_to_1min_format(request)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{ONEMIN_API_URL}?isStreaming=true",
            headers={
                "Content-Type": "application/json",
                "API-KEY": api_key
            },
            json=payload
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise HTTPException(status_code=response.status_code, detail=error_text.decode())
            
            chunk_id = 0
            async for chunk in response.aiter_text():
                if chunk:
                    # Convert to OpenAI SSE format
                    openai_chunk = {
                        "id": f"chatcmpl-{chunk_id}",
                        "object": "chat.completion.chunk",
                        "created": 1234567890,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                    chunk_id += 1
            
            # Send final chunk
            final_chunk = {
                "id": f"chatcmpl-{chunk_id}",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"


async def non_stream_1min_response(request: ChatCompletionRequest, api_key: str) -> dict:
    """Non-streaming response from 1min.ai"""
    payload = convert_to_1min_format(request)
    
    # For non-streaming, still use streaming endpoint but collect all chunks
    full_response = ""
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{ONEMIN_API_URL}?isStreaming=true",
            headers={
                "Content-Type": "application/json",
                "API-KEY": api_key
            },
            json=payload
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise HTTPException(status_code=response.status_code, detail=error_text.decode())
            
            async for chunk in response.aiter_text():
                if chunk:
                    full_response += chunk
    
    return {
        "id": "chatcmpl-1min",
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": full_response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible chat completions endpoint
    
    Pass your 1min.ai API key via Authorization header:
    Authorization: Bearer <your-1min-api-key>
    """
    api_key = get_api_key(authorization)
    
    if request.stream:
        return StreamingResponse(
            stream_1min_response(request, api_key),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_1min_response(request, api_key)


@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = [
        "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-nano", "gpt-5-mini",
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
        "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101",
        "gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash",
        "mistral-large-latest", "deepseek-chat", "grok-3"
    ]
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "1min.ai"} for m in models]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8100))
    print(f"🚀 1min.ai Proxy starting on http://localhost:{port}")
    print(f"   OpenAI-compatible endpoint: http://localhost:{port}/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=port)
