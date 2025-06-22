from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import AsyncGenerator
from prometheus_fastapi_instrumentator import Instrumentator

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