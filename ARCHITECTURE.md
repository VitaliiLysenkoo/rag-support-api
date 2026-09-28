# Architecture & Project Structure

This document outlines the high-level architecture, data flows, and the Single Responsibility Principle (SRP) applied to the codebase.

## 📁 Directory Structure

The project strictly separates concerns. Each file has exactly one responsibility:

```text
rag_support_api/
├── alembic/            # Database migration scripts (auto-generated versions).
├── app/
│   ├── config.py       # Centralized config (Pydantic BaseSettings) & Logger setup.
│   ├── database.py     # Async SQLAlchemy engine (asyncpg) and session management.
│   ├── models.py       # SQLAlchemy ORM models (DocumentChunk and SemanticCache).
│   ├── repository.py   # Database abstraction layer (async CRUD and Cache operations).
│   ├── ai_service.py   # AsyncOpenAI client wrappers with Tenacity retry logic.
│   ├── rag_engine.py   # Core RAG logic: parsing PDFs/TXTs, chunking, orchestrating search.
│   ├── routers/
│   │   └── rag.py      # APIRouter containing the /seed and /ask REST endpoints.
│   └── main.py         # App lifecycle (lifespan), Observability Middleware, and router inclusion.
├── tests/
│   └── test_main.py    # Pytest asynchronous API tests using httpx.
├── test_documents/     # Directory for dropping PDF/TXT files to be seeded.
├── docker-compose.yml  # Infrastructure definition (PostgreSQL + pgvector + API).
├── Dockerfile          # Container build instructions (uses Gunicorn for production).
└── requirements.txt    # Python dependencies.
```

---

## 🔄 Data Flows (Mermaid Diagrams)

### 1. Document Ingestion Flow (`POST /seed`)
Runs asynchronously via FastAPI `BackgroundTasks` to avoid blocking the API.

```mermaid
sequenceDiagram
    participant User
    participant FastAPI (rag.py)
    participant RAGEngine (rag_engine.py)
    participant OpenAI (ai_service.py)
    participant Postgres (pgvector)

    User->>FastAPI: POST /seed
    FastAPI-->>User: "Started processing..." (Immediate Response)
    
    FastAPI-)RAGEngine: Queue Background Task
    Note over RAGEngine: Read PDF/TXT & Chunking (LangChain)
    
    loop For each text chunk
        RAGEngine->>OpenAI: get_embedding(text)
        OpenAI-->>RAGEngine: Return 1536-dim Vector
        RAGEngine->>Postgres: Save (text, vector, metadata)
    end
```

### 2. Retrieval-Augmented Generation Flow (`POST /ask`)
Uses LLM-based Citation to filter mathematical noise and return only accurate sources.

```mermaid
sequenceDiagram
    participant User
    participant FastAPI (rag.py)
    participant Postgres (pgvector)
    participant OpenAI (ai_service.py)

    User->>FastAPI: POST /ask {"question": "Who is the manager?"}
    FastAPI->>OpenAI: get_embedding("Who is the manager?")
    OpenAI-->>FastAPI: Return Query Vector
    
    FastAPI->>Postgres: Check SemanticCache (Similarity > 95%)
    
    alt Cache Hit
        Postgres-->>FastAPI: Return Cached Answer & Sources
        FastAPI-->>User: Instant Response (cached: True)
    else Cache Miss
        Postgres-->>FastAPI: Null
        FastAPI->>Postgres: L2 Distance Search in DocumentChunks (limit 30)
        Postgres-->>FastAPI: Top 30 Similar Chunks
        
        FastAPI->>OpenAI: Send 30 Chunks + Question (Request JSON)
        Note over OpenAI: LLM reads all 30 chunks, extracts answer
        OpenAI-->>FastAPI: JSON {answer: "...", used_source_ids: [28]}
        
        Note over FastAPI: Filter retrieved chunks by used_source_ids
        FastAPI->>Postgres: Save Answer to SemanticCache
        FastAPI-->>User: Return exact answer + ONLY used sources (cached: False)
    end
```
