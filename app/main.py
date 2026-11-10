
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_models import router as models_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_results import router as results_router
from app.api.routes_reports import router as reports_router

from storage.database import initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources when the API starts."""
    initialize_database()
    yield


app = FastAPI(
    title="LLM Preprocessing Comparison System",
    description=(
        "A research platform for comparing baseline and "
        "preprocessed queries across local LLMs using Ollama. "
        "The system supports experiments, evaluation metrics, "
        "result storage, visualizations, and PDF reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def home():
    """Redirect users to the interactive API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health_check():
    """Return the API health status."""
    return {
        "status": "healthy",
        "application": "LLM Preprocessing Comparison System",
        "version": app.version,
    }


app.include_router(
    chat_router,
    prefix="/api/chat",
    tags=["Chat"],
)

app.include_router(
    models_router,
    prefix="/api/models",
    tags=["Models"],
)

app.include_router(
    experiments_router,
    prefix="/api/experiments",
    tags=["Experiments"],
)

app.include_router(
    results_router,
    prefix="/api/results",
    tags=["Results"],
)

app.include_router(
    reports_router,
    prefix="/api/reports",
    tags=["Reports"],
)
