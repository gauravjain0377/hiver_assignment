"""
FastAPI Backend — Phase 5
REST API for the Hiver AI Support Agent.
"""
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings
from src.agent.orchestrator import agent

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hiver AI Support Agent API",
    description="AI-powered customer support agent trained on Twitter data for AppleSupport",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ──────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Customer tweet/message")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I can't log into my Apple account. I've tried resetting my password but it's still not working."
            }
        }


class ClassificationResult(BaseModel):
    intent: str
    intent_label: str
    confidence: float
    method: str


from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class DraftResult(BaseModel):
    reply: str
    retrieved_examples_count: int
    avg_retrieval_score: float
    fallback_used: bool
    retrieved_examples: list = []


class RoutingResult(BaseModel):
    should_escalate: bool
    reason: str
    trigger: str
    priority: str


class ProcessResponse(BaseModel):
    classification: ClassificationResult
    draft: DraftResult
    routing: RoutingResult
    processing_time_ms: float
    brand: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint — used by Docker and Render."""
    return {
        "status": "healthy",
        "brand": settings.target_brand,
        "model": settings.groq_model,
        "environment": settings.environment,
    }


@app.get("/")
async def root():
    """Serve the interactive web UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "name": "Hiver AI Support Agent",
        "brand": settings.target_brand,
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/process", response_model=ProcessResponse)
async def process_message(request: ProcessRequest):
    """
    Full agent pipeline: classify → retrieve → draft → route.

    Returns intent classification, draft reply, and escalation decision.
    """
    try:
        response = agent.process(request.message)
        return ProcessResponse(
            classification=ClassificationResult(
                intent=response.intent,
                intent_label=response.intent_label,
                confidence=response.intent_confidence,
                method=response.classification_method,
            ),
            draft=DraftResult(
                reply=response.draft_reply,
                retrieved_examples_count=response.retrieved_examples_count,
                avg_retrieval_score=response.avg_retrieval_score,
                fallback_used=response.fallback_used,
                retrieved_examples=response.retrieved_examples or [],
            ),
            routing=RoutingResult(
                should_escalate=response.should_escalate,
                reason=response.escalation_reason,
                trigger=response.escalation_trigger,
                priority=response.priority,
            ),
            processing_time_ms=response.processing_time_ms,
            brand=response.brand,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")


@app.post("/classify")
async def classify_message(request: ProcessRequest):
    """Classify intent only (faster, no LLM drafting)."""
    from src.classifier.classifier import classifier
    result = classifier.predict(request.message)
    return result


@app.get("/intents")
async def list_intents():
    """List all supported intent types."""
    from src.classifier.intents import INTENTS
    return {
        name: {
            "label": defn["label"],
            "description": defn["description"],
            "always_escalates": defn.get("escalate_if", False),
        }
        for name, defn in INTENTS.items()
    }
