# Palm Mind RAG

**A production-ready Retrieval-Augmented Generation (RAG) backend** that lets you upload documents or videos, ask natural-language questions about them in a chat session, and book interviews — all from a single conversational API.

Built with a **completely free-tier stack**: no OpenAI billing, no Pinecone subscription, no hidden costs.

---

## What it does

Palm Mind RAG is a FastAPI service with three core capabilities that work together in a single chat session:

| Capability | What happens |
|---|---|
| **Document Ingestion** | Upload a PDF, TXT, or video file. The system chunks the text, generates local embeddings, and stores them in a vector database ready for retrieval. Videos are transcribed automatically via Groq Whisper in the background — the upload returns immediately. |
| **Conversational Q&A** | Send a message referencing your uploaded content. The system rewrites your query, retrieves the most relevant chunks, builds a grounded prompt, and returns an answer with source citations. Conversation history is kept per session via Redis. |
| **Interview Booking** | Within the same chat session, say something like *"Book me next Friday at 3pm, I am Bikram, email me at bikram@example.com."* The system extracts the structured fields, asks follow-up questions for anything missing, and persists a confirmed booking to the database. |

---

## Architecture at a glance

```
User
 │
 ▼
FastAPI (port 8000)
 ├── POST /api/v1/documents/ingest   ← upload files
 ├── GET  /api/v1/documents/{id}     ← check ingestion status
 ├── POST /api/v1/chat               ← Q&A + booking in one endpoint
 └── GET  /health                    ← service health check
 │
 ├── PostgreSQL  — document records, booking records
 ├── Redis       — per-session conversation history
 └── Qdrant      — vector index for chunk retrieval
```

The RAG pipeline is **explicit and transparent** — retrieval, prompt construction, and generation are separate steps in the code. No LangChain abstraction wrappers.

---

## Technology stack

| Layer | Tool | Notes |
|---|---|---|
| API framework | **FastAPI** | Async, auto-generates `/docs` and `/redoc` |
| Validation | **Pydantic v2** | All request/response bodies are typed models |
| Relational DB | **PostgreSQL 15** + SQLAlchemy (async) | Documents and bookings |
| Session memory | **Redis 7** | Sliding conversation history per `session_id` |
| Vector DB | **Qdrant** (self-hosted via Docker) | Stores chunk embeddings for retrieval |
| PDF extraction | **PyMuPDF** | Fast, no external service needed |
| TXT extraction | Python stdlib | Zero dependency |
| Video ingestion | **ffmpeg** + **Groq Whisper Large v3 Turbo** | Non-blocking background task |
| LLM | **Groq** — Llama 3.3 70B / Llama 3.1 8B | Free tier; used for Q&A, query rewriting, and field extraction |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Local model, no API calls |
| Containerization | **Docker + Docker Compose** | Single command to start everything |
| Linting / formatting | **Ruff** | |
| Testing | **pytest** | |

**Running cost: $0** — every external API used has a free tier sufficient for development and moderate production load.

---

## Prerequisites

Before you start, make sure you have:

- **Docker Desktop** (or Docker Engine + Docker Compose v2) installed and running
- A free **Groq API key** — get one at [console.groq.com](https://console.groq.com) (takes ~30 seconds, no credit card)
- `ffmpeg` on your system **only if** you want to test video uploads outside Docker (inside Docker it is already included)

That is it. Python, Postgres, Redis, and Qdrant are all managed inside Docker — you do not need to install them locally.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/imgrooty/palmRAG.git
cd palmRAG
```

### 2. Create your environment file

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

All other values have working defaults for Docker Compose and do not need to be changed to get started.

### 3. Start all services

```bash
docker-compose up --build
```

This starts four containers: the FastAPI app, PostgreSQL, Redis, and Qdrant. The first run takes a few minutes to build the image and download the embedding model.

### 4. Verify everything is running

```bash
curl http://localhost:8000/health
```

Expected response (all three backing services must be `"connected"`):

```json
{
  "status": "ok",
  "postgres": "connected",
  "redis": "connected",
  "qdrant": "connected"
}
```

### 5. Open the interactive API docs

Navigate to **http://localhost:8000/docs** in your browser. Every endpoint is documented there with a live "Try it out" interface.

---

## Environment variables

Copy `.env.example` to `.env` and fill in the values marked **Required**.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | API key from console.groq.com |
| `DATABASE_URL` | No | set by Docker Compose | SQLAlchemy async connection string for PostgreSQL |
| `REDIS_URL` | No | set by Docker Compose | Redis connection string |
| `QDRANT_URL` | No | set by Docker Compose | Qdrant HTTP endpoint |
| `QDRANT_COLLECTION` | No | `documents` | Vector collection name |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `GROQ_CHAT_MODEL` | No | `llama-3.3-70b-versatile` | Main answer generation model |
| `GROQ_FAST_MODEL` | No | `llama-3.1-8b-instant` | Used for query rewriting and field extraction |
| `GROQ_WHISPER_MODEL` | No | `whisper-large-v3-turbo` | Transcription model for video/audio |
| `MAX_UPLOAD_MB` | No | `50` | Max upload file size in megabytes |
| `ENV` | No | `development` | `development` or `production` |

> When running via `docker-compose up`, `DATABASE_URL`, `REDIS_URL`, and `QDRANT_URL` are pre-wired between containers. You only need to set `GROQ_API_KEY`.

---

## API usage

### Ingest a document

Upload a PDF or TXT file. The response comes back immediately with a `document_id`.

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@/path/to/your/document.pdf" \
  -F "chunking_strategy=recursive"
```

Available chunking strategies: `recursive` (recommended), `fixed`, `sentence`.

**Response:**
```json
{
  "document_id": "a1b2c3d4-...",
  "filename": "document.pdf",
  "file_type": "pdf",
  "chunking_strategy": "recursive",
  "chunks_created": 42,
  "status": "completed"
}
```

### Ingest a video

For video files (`.mp4`, `.mov`, etc.), transcription runs in the background. The endpoint returns immediately with `"status": "processing"`.

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@/path/to/interview.mp4" \
  -F "chunking_strategy=sentence"
```

Poll the status until it reads `"completed"`:

```bash
curl http://localhost:8000/api/v1/documents/{document_id}
```

### Ask a question

Use any `session_id` string to start a conversation. Re-use the same `session_id` in follow-up messages to maintain context.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-01",
    "message": "What are the key qualifications mentioned in the document?"
  }'
```

**Response:**
```json
{
  "answer": "The document highlights three key qualifications: ...",
  "sources": [
    { "document_id": "a1b2c3d4-...", "chunk_index": 3, "score": 0.91 }
  ],
  "booking": null
}
```

### Book an interview

No special endpoint — just say it in plain English within the same chat session:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session-01",
    "message": "I would like to book an interview for next Monday at 10am. My name is Bikram and my email is bikram@example.com."
  }'
```

**Response:**
```json
{
  "answer": "Your interview has been booked for Monday at 10:00 AM. A confirmation will be sent to bikram@example.com.",
  "sources": [],
  "booking": {
    "id": "f9e8d7c6-...",
    "name": "Bikram",
    "email": "bikram@example.com",
    "scheduled_at": "2026-09-07T10:00:00",
    "status": "confirmed"
  }
}
```

If any required field is missing (name, email, date, or time), the system will ask a follow-up question in the same conversation before confirming.

---

## Running tests

```bash
# Run the full test suite
pytest -v

# Check linting
ruff check .

# Check formatting
ruff format --check .
```

Unit tests mock all external calls (Groq, Qdrant, Redis, PostgreSQL), so you can run `pytest` without any services running locally.

---

## Project structure

```
palmRAG/
├── app/
│   ├── api/v1/          # FastAPI route handlers (documents, chat)
│   ├── core/            # Config (Pydantic Settings), logging
│   ├── db/              # SQLAlchemy engine and session factory
│   ├── integrations/    # Groq, Qdrant, Redis client wrappers
│   ├── models/          # SQLAlchemy ORM models
│   ├── repositories/    # All database access (no raw queries in routes)
│   ├── schemas/         # Pydantic v2 request/response models
│   ├── services/        # Business logic: ingestion, RAG, booking, chunking
│   └── main.py          # FastAPI app factory and health check
├── tests/               # pytest test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Common issues

**`GROQ_API_KEY` not set — app refuses to start**
Set the key in your `.env` file before running `docker-compose up`.

**Health check shows `"qdrant": "disconnected"`**
Qdrant takes a few seconds longer to become ready. Wait 10–15 seconds and retry. If it persists, run `docker-compose logs qdrant` to inspect.

**Video upload stays at `processing` indefinitely**
Verify ffmpeg is available inside the container:
```bash
docker exec palmmind_api ffmpeg -version
```
If it is missing, rebuild the image with `docker-compose up --build`.

**`413 Request Entity Too Large` on upload**
Increase `MAX_UPLOAD_MB` in your `.env` file and restart the service.

**Embedding model downloads on every container restart**
This is expected on the first cold start only. Subsequent starts use the cached model from the container layer.

---

## Stopping the services

```bash
# Stop containers, keep all data volumes (documents, bookings, vectors)
docker-compose down

# Stop containers and delete all data permanently
docker-compose down -v
```

---

## License

This project is provided for evaluation and reference purposes.
