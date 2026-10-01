import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import text

from app.config import setup_logger
from app.database import engine
from app.routers import rag

logger = setup_logger("api_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    yield

app = FastAPI(
    title="Corporate AI Knowledge API",
    description="Production-ready RAG API built with FastAPI, PostgreSQL (pgvector) and OpenAI.",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Observability Middleware:
    Tracks the execution time of every request and logs it.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(f"Method={request.method} Path={request.url.path} Status={response.status_code} Time={process_time:.4f}s")
    return response

app.include_router(rag.router)
