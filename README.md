# 1min.ai OpenAI Proxy

A lightweight proxy server that converts OpenAI-compatible API requests to 1min.ai format.

## Features

- 🔄 **OpenAI-compatible API** - Use any OpenAI client/SDK
- 🌊 **Streaming support** - Real-time response streaming (SSE)
- 🔑 **API key passthrough** - Use your own 1min.ai API key
- 🚀 **All 1min.ai models** - GPT-5, Claude, Gemini, Mistral, and more

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn httpx

# Set your 1min.ai API key (optional - can also pass via header)
export ONEMIN_API_KEY="your-1min-api-key"

# Run the proxy
python main.py
```

The proxy will start on `http://localhost:8100`

## Usage

### With curl

```bash
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_1MIN_API_KEY" \
  -d '{
    "model": "gpt-5-nano",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### With OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8100/v1",
    api_key="your-1min-api-key"
)

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Available Models

### OpenAI
- `gpt-5-nano`, `gpt-5-mini`, `gpt-5`, `gpt-5.1`, `gpt-5.2`
- `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`
- `o3`, `o3-mini`, `o4-mini`

### Anthropic
- `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101`
- `claude-opus-4-20250514`, `claude-haiku-4-5-20251001`

### Google
- `gemini-3-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`

### Others
- `mistral-large-latest`, `deepseek-chat`, `grok-3`, `grok-4-0709`
- `sonar-pro`, `sonar-reasoning-pro` (Perplexity)

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (OpenAI-compatible) |

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `ONEMIN_API_KEY` | Default 1min.ai API key | None |
| `PORT` | Server port | 8100 |

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi uvicorn httpx
COPY main.py .
EXPOSE 8100
CMD ["python", "main.py"]
```

## License

MIT
