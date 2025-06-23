from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import AsyncGenerator
from prometheus_fastapi_instrumentator import Instrumentator
import json, uuid, time

# Use relative path for import that will work with the new structure
from src.core.pipeline import AegisRAGPipeline

app = FastAPI(title="Aegis RAG API", version="0.1.0")

# Initialize metrics instrumentator (Prometheus)
instrumentator = (
    Instrumentator()
    .instrument(app)
    .expose(app, include_in_schema=False, should_gzip=True)
)

# CORS for local dev UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: AegisRAGPipeline | None = None


# ---------------------------- OpenAI compatibility Models ----------------------------

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "aegis"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]

# --------------------------------------------------------------------------------

def get_pipeline() -> AegisRAGPipeline:
    global pipeline
    if pipeline is None:
        # Lazy init to avoid heavy deps during import (esp. tests)
        try:
            pipeline = AegisRAGPipeline()
        except Exception as e:  # capture environment issues
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    return pipeline


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok"}


async def stream_generator(agen: AsyncGenerator[str, None]):
    """Wrap tokens as Server-Sent Events (SSE)."""
    async for token in agen:
        # Each SSE event must end with a double newline
        yield f"data: {token}\n\n"


@app.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest):
    """Endpoint to handle chat requests with a streaming response (SSE)."""
    try:
        response_generator = get_pipeline().query_stream(req.question, top_k=req.top_k)
        # Return SSE stream
        return StreamingResponse(
            stream_generator(response_generator),
            media_type="text/event-stream",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    try:
        answer = await get_pipeline().query(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    return ChatResponse(answer=answer)


# ---------------------------- OpenAI compatibility ----------------------------

@app.get("/v1/models", response_model=ModelList, tags=["openai"])
async def list_models():
    """Return a list of available models, including our custom RAG model."""
    return ModelList(
        data=[
            ModelCard(id="aegis-rag-model")
        ]
    )

@app.post("/v1/chat/completions", tags=["openai"])
async def openai_chat_completions(payload: dict = Body(...)):
    """Minimal OpenAI-compatible endpoint for Open WebUI integration.

    Only a subset of the full specification is implemented:
    - `messages`: list with at least one user message (the last user message is used).
    - `stream`: if true, SSE-style streaming chunks are returned following OpenAI format.
    Additional fields are accepted but ignored for now.
    """
    model = payload.get("model", "aegis-rag-model")
    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="'messages' field is required")

    # Extract user content from the last message
    user_content = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    if not user_content:
        raise HTTPException(status_code=400, detail="No user message found in 'messages'")

    top_k = payload.get("top_k", 5)
    stream = bool(payload.get("stream", False))

    if stream:
        async def _event_stream():
            async for token in get_pipeline().query_stream(user_content, top_k=top_k):
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {"content": token},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            # Signal completion per OpenAI spec
            yield "data: [DONE]\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

    # Non-streaming path
    answer = await get_pipeline().query(user_content, top_k=top_k)
    response_body = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        # Token usage accounting is skipped (set to zero for now)
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return response_body 