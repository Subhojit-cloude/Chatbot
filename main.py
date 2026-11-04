"""
============================================================
CampusMate — FastAPI Server (main.py)
============================================================
REST API for the college RAG chatbot.

Endpoints:
  GET  /api/health    — Health check
  POST /api/chat      — Standard RAG endpoint (accepts {"question": "..."})
  POST /api/rag-chat  — Rich RAG endpoint for React frontend (accepts {"query": "..."})

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
============================================================
"""

import os
import sys
import time
import asyncio
from typing import Optional, List, Any

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Base directory for relative paths (resolves properly regardless of working directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# pyrefly: ignore[missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore[missing-import]
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the RAG pipeline
from rag_pipeline import get_answer

# ============================================================
# 1. Initialize FastAPI app
# ============================================================
app = FastAPI(
    title="CampusMate API",
    description="AI-powered college assistant using RAG with Google Gemini",
    version="1.0.0",
)

# ============================================================
# 2. CORS middleware — allows frontend apps to call this API
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (Vite dev server, localhost, etc.)
    allow_credentials=False,
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

# ============================================================
# 3. Request / Response models (Pydantic validation)
# ============================================================
class ChatRequest(BaseModel):
    """Standard chat request."""
    question: Optional[str] = Field(None, description="Student question")
    query: Optional[str] = Field(None, description="Alternative query field")


class ChatResponse(BaseModel):
    """Standard structured response."""
    answer: str
    sources: list[str]


class RetrievedSourceItem(BaseModel):
    """Rich source chunk item matching frontend card requirements."""
    id: str
    title: str
    section: str
    score: float
    snippet: str
    url: Optional[str] = None


class RagChatRequest(BaseModel):
    """Rich RAG request from the React frontend."""
    query: Optional[str] = None
    question: Optional[str] = None
    collection_name: Optional[str] = None
    top_k: Optional[int] = 15
    similarity_threshold: Optional[float] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    history: Optional[List[Any]] = []


class RagChatResponse(BaseModel):
    """Rich RAG response tailored for the frontend UI."""
    response: str
    answer: str
    sources: List[RetrievedSourceItem]
    latency_ms: int
    tokens_used: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str

# ============================================================
# 4. GET /api/health — Health check endpoint
# ============================================================
@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the current health status of the CampusMate API.",
)
async def health_check():
    """Returns API health status."""
    return HealthResponse(
        status="healthy",
        message="CampusMate API is running",
    )

# ============================================================
# 5. POST /api/chat — Standard chat endpoint
# ============================================================
@app.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Ask CampusMate",
    description="Send a college-related question and receive an AI-generated answer with sources.",
)
async def chat(request: ChatRequest):
    """
    Accepts a student question, runs it through the RAG pipeline in a background thread,
    and returns the answer with source documents.
    """
    user_query = (request.question or request.query or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Missing question or query field.")

    try:
        result = await asyncio.to_thread(get_answer, user_query)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        print(f"❌ Error processing question: {e}", flush=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Please try again later.",
        )

# ============================================================
# 6. POST /api/rag-chat — Rich endpoint for the React frontend
# ============================================================
@app.post(
    "/api/rag-chat",
    response_model=RagChatResponse,
    summary="Rich RAG Chat for React Frontend",
    description="Endpoint matching the frontend RagBackendConfig with rich source chunks and latency metrics.",
)
async def handle_rag_chat(request: RagChatRequest):
    """
    Processes queries sent from the React UI settings panel or chat workspace.
    Includes instant mock response for health check pings.
    """
    start_time = time.time()
    user_query = (request.query or request.question or "").strip()

    # Fast-path for frontend "Test Connection" button ping
    if not user_query or user_query == "ping_healthcheck_test":
        latency = int((time.time() - start_time) * 1000)
        return RagChatResponse(
            response="CampusMate RAG API is connected and healthy.",
            answer="CampusMate RAG API is connected and healthy.",
            sources=[],
            latency_ms=max(1, latency),
            tokens_used=5,
        )

    try:
        result = await asyncio.to_thread(get_answer, user_query)
        latency = int((time.time() - start_time) * 1000)

        raw_chunks = result.get("chunks", [])
        formatted_sources = [
            RetrievedSourceItem(
                id=c.get("id", f"chunk-{i+1}"),
                title=c.get("title", "College Document"),
                section=c.get("section", "Verified Section"),
                score=c.get("score", 0.90),
                snippet=c.get("snippet", ""),
                url=c.get("url"),
            )
            for i, c in enumerate(raw_chunks)
        ]

        tokens_est = int((len(user_query) + len(result["answer"])) / 3.8)

        return RagChatResponse(
            response=result["answer"],
            answer=result["answer"],
            sources=formatted_sources,
            latency_ms=latency,
            tokens_used=tokens_est,
        )
    except Exception as e:
        print(f"❌ Error in /api/rag-chat: {e}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}",
        )

# ============================================================
# 7. Root endpoint — friendly redirect info
# ============================================================
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "CampusMate API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "chat": "POST /api/chat",
        "rag_chat": "POST /api/rag-chat",
    }

# ============================================================
# 8. Run with uvicorn when executed directly
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting CampusMate API server...", flush=True)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
