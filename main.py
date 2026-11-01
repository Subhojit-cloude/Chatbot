"""
============================================================
CampusMate — FastAPI Server (main.py)
============================================================
REST API for the college RAG chatbot.

Endpoints:
  GET  /api/health  — Health check
  POST /api/chat    — Ask CampusMate a question

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
============================================================
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    allow_origins=["*"],          # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"],          # Allow all headers
)

# ============================================================
# 3. Request / Response models (Pydantic validation)
# ============================================================
class ChatRequest(BaseModel):
    """Validates the incoming chat question."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Student question (1–500 characters)",
        examples=["What courses does the college offer?"],
    )


class ChatResponse(BaseModel):
    """Structured response with answer and sources."""
    answer: str = Field(
        ...,
        description="CampusMate's answer to the question",
    )
    sources: list[str] = Field(
        ...,
        description="List of unique source documents used",
    )


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
# 5. POST /api/chat — Main chat endpoint
# ============================================================
@app.post(
    "/api/chat",
    response_model=ChatResponse,
    summary="Ask CampusMate",
    description="Send a college-related question and receive an AI-generated answer with sources.",
)
async def chat(request: ChatRequest):
    """
    Accepts a student question, runs it through the RAG pipeline,
    and returns the answer with source documents.
    """
    try:
        # Call the RAG pipeline
        result = get_answer(request.question)

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
        )

    except Exception as e:
        # Log the error for debugging (visible in server console)
        print(f" Error processing question: {e}")

        # Return a friendly error to the client
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate an answer. Please try again later.",
                "message": "The AI service encountered an issue. "
                           "If this persists, please contact support.",
            },
        )

# ============================================================
# 6. Root endpoint — friendly redirect info
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
    }

# ============================================================
# 7. Run with uvicorn when executed directly
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting CampusMate API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
