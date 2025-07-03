from fastapi import FastAPI, HTTPException, Body, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import AsyncGenerator
from prometheus_fastapi_instrumentator import Instrumentator
import json, uuid, time
import subprocess
from pathlib import Path

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

# ---------------------------- Internal Operations ----------------------------

class IngestionRequest(BaseModel):
    path: str
    collection_name: str = "aegis_docs"

class IngestionResponse(BaseModel):
    status: str
    message: str

def run_ingestion_background(data_path: str, collection_name: str = "aegis_docs"):
    """Background task to run document ingestion."""
    try:
        # Run the ingestion script
        cmd = [
            "python", "-m", "src.scripts.ingest", 
            "--data_dir", data_path,
            "--collection", collection_name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/app")
        if result.returncode == 0:
            print(f"✅ Ingestion completed for {data_path}")
        else:
            print(f"❌ Ingestion failed for {data_path}: {result.stderr}")
    except Exception as e:
        print(f"❌ Ingestion error for {data_path}: {e}")

@app.post("/internal/ingest", response_model=IngestionResponse, tags=["internal"])
async def internal_ingest(
    background_tasks: BackgroundTasks,
    path: str = Query(..., description="Path to documents directory"),
    collection: str = Query("aegis_docs", description="Qdrant collection name")
):
    """Internal endpoint for triggering document ingestion from Open WebUI file uploads.
    
    This endpoint is designed to be called by Open WebUI plugins after file upload.
    It validates the path and starts a background ingestion task.
    """
    # Validate path exists and is accessible
    doc_path = Path(path)
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Path {path} not found")
    
    if not doc_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path {path} is not a directory")
    
    # Count documents to ingest
    documents = list(doc_path.glob("**/*.md")) + list(doc_path.glob("**/*.pdf")) + list(doc_path.glob("**/*.txt"))
    if not documents:
        return IngestionResponse(
            status="no_documents", 
            message=f"No supported documents found in {path}"
        )
    
    # Start background ingestion
    background_tasks.add_task(run_ingestion_background, str(doc_path), collection)
    
    return IngestionResponse(
        status="accepted",
        message=f"Ingestion started for {len(documents)} documents in {path}"
    ) 