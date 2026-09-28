# Corporate AI Knowledge Base (RAG API)

A production-ready Retrieval-Augmented Generation (RAG) microservice built with **FastAPI**, **PostgreSQL (`pgvector`)**, and **OpenAI**. 

It reads corporate documents (PDF/TXT), chunks them semantically, and answers user questions with strict context boundaries to prevent hallucinations.

## 🛠 Tech Stack
- **Backend:** FastAPI, PostgreSQL 16 + pgvector, SQLAlchemy (Async), Alembic, Pydantic Settings
- **AI:** AsyncOpenAI (`text-embedding-3-small`, `gpt-4o-mini`), LangChain
- **Architecture:** Fully Async, Repository Pattern, Background Tasks, Gunicorn, LLM-based Citation
- **Testing:** Pytest, pytest-asyncio, httpx

## 🌐 Live Demo
You don't need to deploy this locally to test it! The API is currently running on my live server:
👉 **[Swagger UI: http://166.88.227.29:8000/docs](http://166.88.227.29:8000/docs)**

*Note: The database is already seeded with test documents (including the Valve Employee Handbook). You can test the `/ask` endpoint immediately. If you'd like to test the system on your own corporate documents, feel free to send them to me and I will upload them to the server!*

## 📦 Quickstart (Docker)

1. Add your API keys:
```bash
cp .env.example .env
nano .env # Insert OPENAI_API_KEY
```

2. Run database migrations:
```bash
docker-compose build api
docker-compose run --rm api alembic upgrade head
```

3. Run the application (Production Gunicorn server):
```bash
docker-compose up -d
```

4. Open **http://localhost:8000/docs** in your browser.

## 💡 API Usage

1. **`POST /seed`**: Scans the `/test_documents` directory and ingests all files into the vector database in the background.
2. **`POST /ask`**: Ask a question (e.g., *"Who is my manager at Valve?"*). Returns the AI answer and cites the exact document sources. **Note (LLM-based Citation & Semantic Cache):** The system uses a two-stage retrieval process. First, the vector database finds the top 30 most mathematically similar text chunks. Then, the LLM reads all 30 chunks, forms an answer, and strictly filters the `sources` array to return *only* the specific snippets that actually contained the answer, hiding the irrelevant noise from the user. Finally, the generated answer is cached semantically; future identical questions will return instantly in <0.05s without invoking the LLM.

## Architecture Highlights

- **Fully Asynchronous (asyncpg):** Maximum throughput under load using `AsyncSession`, `AsyncOpenAI`, and `await db.execute`.
- **Database Migrations (Alembic):** Version-controlled schema migrations instead of fragile `create_all()`.
- **Production-Ready Server (Gunicorn):** Deployed with `uvicorn.workers.UvicornWorker` across 4 workers to fully utilize CPU cores.
- **Observability Middleware:** Intercepts and logs the exact execution time (`X-Process-Time`) of every REST request.
- **pgvector over SaaS:** Keeps vectors and relational data in the same ACID-compliant database, reducing infrastructure complexity.
- **Resiliency (Tenacity):** External API calls to OpenAI are wrapped with exponential backoff retries (`tenacity`) to gracefully handle API errors or timeouts.
- **Race-Condition Protection:** The `/seed` endpoint utilizes an in-memory lock (`processing_files`) and isolated DB sessions to prevent duplicate chunking if background tasks overlap.
- **Config Management:** Strict environment variable validation using Pydantic `BaseSettings` instead of fragile `os.getenv`.
- **Enterprise Routing:** REST endpoints are decoupled from the application lifecycle using `APIRouter`.
- **Semantic Caching:** Dramatically reduces OpenAI API costs and latency by caching previous LLM answers and retrieving them when mathematically similar (>95%) questions are asked.
- **Pure Python RAG:** LangChain is used *only* for semantic text-splitting. The core retrieval and prompt injection logic is written in pure Python for maximum transparency and control.

---

> 📖 **Curious how it works under the hood?** Check out [ARCHITECTURE.md](./ARCHITECTURE.md) for data flow diagrams and file structure explanations!
