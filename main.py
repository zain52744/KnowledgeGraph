"""
FastAPI main application with lifespan, CORS, and router registration.
"""
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config import settings


def _configure_logging() -> None:
    log_dir = settings.log_dir
    os.makedirs(log_dir, exist_ok=True)

    structured_fmt = logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "message": "%(message)s"}'
    )
    plain_fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(structured_fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(plain_fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid duplicate handlers on reload
    if not root.handlers:
        root.addHandler(file_handler)
        root.addHandler(console_handler)
    else:
        root.handlers.clear()
        root.addHandler(file_handler)
        root.addHandler(console_handler)


_configure_logging()
logger = logging.getLogger(__name__)

# Global MongoDB connection
mongo_client: AsyncIOMotorClient = None
mongo_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup/shutdown."""
    logger.info("Starting up Knowledge Graph API...")
    global mongo_client, mongo_db

    try:
        mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
        mongo_db = mongo_client["knowledge_graph"]
        await mongo_db.command("ping")
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")

    # Redis cache
    try:
        from api.cache import cache
        cache.connect()
    except Exception as e:
        logger.warning(f"Redis connection failed (cache disabled): {e}")

    # Initialize LangFuse tracing first so callbacks are ready before LLMs are created
    from monitoring.langfuse_setup import langfuse_manager
    langfuse_manager.init()

    # Initialize shared graph, vector store, and pipeline
    try:
        from api.dependencies import init_shared_resources
        init_shared_resources()
    except Exception as e:
        logger.error(f"Failed to initialize shared resources: {e}")
        raise

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down Knowledge Graph API...")

    # Persist FAISS index to disk before exit
    try:
        from api.dependencies import get_vector_store
        get_vector_store().close()
        logger.info("FAISS index saved to disk")
    except Exception as e:
        logger.warning(f"Failed to save FAISS index on shutdown: {e}")

    try:
        from api.cache import cache
        cache.close()
    except Exception as e:
        logger.warning(f"Redis shutdown error: {e}")

    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")


# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Knowledge Graph RAG API",
    description="Hybrid RAG system with knowledge graphs and vector search",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiter
from api.limiter import limiter  # noqa: E402  (after app creation to avoid circular)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restrict origins via env var in production
allowed_origins = settings.allowed_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from api.routes import ingest, query  # noqa: E402

app.include_router(ingest.router)
app.include_router(query.router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check for all critical dependencies.

    Reports MongoDB, FAISS index, and pipeline readiness.
    """
    checks: dict = {}

    # MongoDB
    checks["mongodb"] = "connected" if mongo_db is not None else "disconnected"

    # FAISS vector store
    try:
        from api.dependencies import get_vector_store
        vs = get_vector_store()
        stats = vs.get_stats()
        checks["faiss"] = {"status": "loaded", "total_chunks": stats["total_chunks"]}
    except Exception as e:
        checks["faiss"] = {"status": "unavailable", "error": str(e)}

    # Pipeline
    try:
        from api.dependencies import get_pipeline
        get_pipeline()
        checks["pipeline"] = "ready"
    except Exception as e:
        checks["pipeline"] = f"unavailable: {e}"

    overall = "healthy" if all(
        (v == "connected" or v == "ready" or (isinstance(v, dict) and v.get("status") == "loaded"))
        for v in checks.values()
    ) else "degraded"

    return {"status": overall, **checks}


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Knowledge Graph RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ingest_pdf": "POST /ingest/pdf",
            "ingest_csv": "POST /ingest/csv",
            "query": "POST /query",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.api_port,
        log_level="info",
    )
