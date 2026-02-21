#!/usr/bin/env python3
"""
1min.ai to OpenAI API Proxy Server
Converts OpenAI-compatible requests to 1min.ai format with conversation context support.

Usage:
  - Set API key via Authorization header: "Bearer <your-1min-api-key>"
  - Or use default key from ONEMIN_API_KEY environment variable
"""

import asyncio
import json
import os
import uuid
import hashlib
from typing import AsyncGenerator, Dict
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="1min.ai OpenAI Proxy")

# 1min.ai API configuration
ONEMIN_API_URL = "https://api.1min.ai/api/features"
ONEMIN_CONV_URL = "https://api.1min.ai/api/conversations"
DEFAULT_API_KEY = os.environ.get("ONEMIN_API_KEY", "a2e6a2350d21850d0a8bcdc2ecfb6ad5e5fa2040457cfbd156a715dc978c5512")

# In-memory conversation store: {session_key: {conversation_id, model, created_at, last_used}}
conversations: Dict[str, dict] = {}
CONVERSATION_TTL_HOURS = 24


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


def get_api_key(authorization: Optional[str] = None) -> str:
    """Extract API key from Authorization header or use default"""
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]
        return authorization
    return DEFAULT_API_KEY


def get_session_key(api_key: str, model: str, messages: List[Message]) -> str:
    """Generate a session key based on API key, model, and system message"""
    # Use system message as part of session identity (same system = same conversation)
    system_content = ""
    for msg in messages:
        if msg.role == "system":
            system_content = msg.content[:500]  # First 500 chars of system message
            break
    
    key_data = f"{api_key}:{model}:{system_content}"
    return hashlib.md5(key_data.encode()).hexdigest()[:16]


async def create_conversation(api_key: str, model: str, title: str = "OpenAI Proxy Chat") -> str:
    """Create a new conversation in 1min.ai"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            ONEMIN_CONV_URL,
            headers={
                "Content-Type": "application/json",
                "API-KEY": api_key
            },
            json={
                "type": "CHAT_WITH_AI",
                "title": title[:91],  # Max 91 chars
                "model": model
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Failed to create conversation: {response.text}")
        
        data = response.json()
        return data["conversation"]["uuid"]


async def get_or_create_conversation(api_key: str, model: str, messages: List[Message]) -> str:
    """Get existing conversation or create new one"""
    session_key = get_session_key(api_key, model, messages)
    now = datetime.now()
    
    # Clean up old conversations
    expired_keys = [
        k for k, v in conversations.items()
        if now - v["last_used"] > timedelta(hours=CONVERSATION_TTL_HOURS)
    ]
    for k in expired_keys:
        del conversations[k]
    
    # Check if we have an existing conversation
    if session_key in conversations:
        conv = conversations[session_key]
        # Check if model matches
        if conv["model"] == model:
            conv["last_used"] = now
            return conv["conversation_id"]
    
    # Create new conversation
    title = f"Chat {now.strftime('%Y-%m-%d %H:%M')}"
    conversation_id = await create_conversation(api_key, model, title)
    
    conversations[session_key] = {
        "conversation_id": conversation_id,
        "model": model,
        "created_at": now,
        "last_used": now
    }
    
    return conversation_id


def convert_to_1min_format(request: ChatCompletionRequest, conversation_id: str) -> dict:
    """Convert OpenAI request to 1min.ai format with conversation context"""
    # Get the last user message as prompt
    prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            prompt = msg.content
            break
    
    return {
        "type": "CHAT_WITH_AI",
        "conversationId": conversation_id,
        "model": request.model,
        "promptObject": {
            "prompt": prompt,
            "isMixed": False,
            "webSearch": False
        }
    }


async def stream_1min_response(request: ChatCompletionRequest, api_key: str, conversation_id: str) -> AsyncGenerator[str, None]:
    """Stream response from 1min.ai"""
    payload = convert_to_1min_format(request, conversation_id)
    
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
                        "created": int(datetime.now().timestamp()),
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
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"


async def non_stream_1min_response(request: ChatCompletionRequest, api_key: str, conversation_id: str) -> dict:
    """Non-streaming response from 1min.ai"""
    payload = convert_to_1min_format(request, conversation_id)
    
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
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
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
    """OpenAI-compatible chat completions endpoint with conversation context
    
    Pass your 1min.ai API key via Authorization header:
    Authorization: Bearer <your-1min-api-key>
    
    Conversations are automatically managed - same model + system message = same conversation.
    """
    api_key = get_api_key(authorization)
    
    # Get or create conversation for context
    conversation_id = await get_or_create_conversation(api_key, request.model, request.messages)
    
    if request.stream:
        return StreamingResponse(
            stream_1min_response(request, api_key, conversation_id),
            media_type="text/event-stream"
        )
    else:
        return await non_stream_1min_response(request, api_key, conversation_id)


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
    return {"status": "ok", "conversations": len(conversations)}


@app.get("/conversations")
async def list_conversations():
    """List active conversations (for debugging)"""
    return {
        "count": len(conversations),
        "conversations": [
            {
                "session_key": k,
                "conversation_id": v["conversation_id"],
                "model": v["model"],
                "created_at": v["created_at"].isoformat(),
                "last_used": v["last_used"].isoformat()
            }
            for k, v in conversations.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8100))
    print(f"🚀 1min.ai Proxy starting on http://localhost:{port}")
    print(f"   OpenAI-compatible endpoint: http://localhost:{port}/v1/chat/completions")
    print(f"   ✅ Conversation context enabled!")
    uvicorn.run(app, host="0.0.0.0", port=port)
