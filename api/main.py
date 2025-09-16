from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List, Optional
import redis.asyncio as redis
from prometheus_fastapi_instrumentator import Instrumentator

from .routers import upload, search, health
from .dependencies import get_vector_store, get_redis_client
from .models.document import DocumentResponse
from .models.query import QueryRequest, QueryResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.redis = await redis.from_url("redis://localhost:6379")
    app.state.vector_store = await get_vector_store()
    yield
    # Shutdown
    await app.state.redis.close()

app = FastAPI(
    title="GenAI Knowledge Extraction API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Include routers
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])

@app.get("/")
async def root():
    return {"message": "GenAI Knowledge Extraction API", "status": "operational"}
