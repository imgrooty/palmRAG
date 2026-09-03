# Palm Mind RAG

**A high-performance, production-ready Retrieval-Augmented Generation (RAG) backend** that lets you upload documents or videos, ask natural-language questions about them in a chat session, and book interviews — all from a single conversational API.

Built with a **completely free-tier stack**: no OpenAI billing, no Pinecone subscription, no hidden costs. Optimized for ultra-low memory footprint (~80 MB embedding RAM vs 600 MB PyTorch) and high throughput using Rust-backed components (`granian`, `orjson`, `fastembed`, `uvloop`).

---

## What it does

Palm Mind RAG is a FastAPI service with three core capabilities that work together in a single chat session:

| Capability | What happens |
|---|---|
| **Document Ingestion** | Upload a PDF, TXT, or video file. The system chunks the text using selectable strategies (`fixed` or `recursive`), generates local ONNX embeddings via `fastembed`, and stores them in Qdrant. Videos are transcribed automatically via Groq Whisper in the background — the upload returns immediately. |
| **Conversational Q&A** | Send a message referencing your uploaded content. The system rewrites your query, retrieves the most relevant chunks from Qdrant, builds a grounded prompt, and returns an answer with source citations. Conversation history is kept per session via Redis using `orjson`. |
| **Interview Booking** | Within the same chat session, say something like *"Book me next Friday at 3pm, I am Bikram, email me at bikram@example.com."* The system extracts structured fields via Groq LLM, asks follow-up questions for anything missing, and persists a confirmed booking to MongoDB. |

---

## Architecture at a glance

```
User
 │
 ▼
Granian / FastAPI (port 8000)
 ├── POST /api/v1/documents/ingest   ← upload files
 ├── POST /api/v1/chat               ← Q&A + booking in one endpoint
 └── GET  /health                    ← service health check
 │
 ├── MongoDB (Motor) — document records, chunk audit, booking records
 ├── Redis           — per-session conversation history
 └── Qdrant          — vector index for chunk retrieval
```

The RAG pipeline is **explicit and transparent** — retrieval, prompt construction, and generation are separate steps in the code. No LangChain abstraction wrappers.

---

## Technology stack

| Layer | Tool | Notes |
|---|---|---|
| API server | **Granian** | Rust-backed high-throughput ASGI server |
| Framework | **FastAPI** | Async framework, auto-generates `/docs` and `/redoc` |
| Validation | **Pydantic v2** | All request/response bodies are typed models |
| Database | **MongoDB** + **Motor** | Async-native NoSQL storage for documents & bookings |
| Session memory | **Redis 7** + **orjson** | Sliding conversation history per `session_id` |
| Vector DB | **Qdrant** (self-hosted via Docker) | Stores chunk embeddings for retrieval |
| PDF extraction | **pypdfium2** | Google PDFium engine — fast, low memory |
| TXT extraction | Python stdlib | Zero dependency |
| Video ingestion | **ffmpeg** + **Groq Whisper Large v3 Turbo** | Non-blocking background task |
| LLM | **Groq** — Llama 3.3 70B / Llama 3.1 8B | Free tier; used for Q&A, query rewriting, and field extraction |
| Embeddings | **Gemini `gemini-embedding-001`** (primary) / **fastembed** (fallback) | Gemini: 768-dim, free tier 1500 RPM. Fallback: local ONNX, 384-dim, zero cost |
| Event loop | **uvloop** | Enabled automatically in Linux production containers |
| Containerization | **Docker + Docker Compose** | Single command to start everything |
| Linting / formatting | **Ruff** | |
| Testing | **pytest** | |

**Running cost: $0** — every external API used has a free tier sufficient for development and production load.

---

## Prerequisites

Before you start, make sure you have:

- **Docker Desktop** (or Docker Engine + Docker Compose v2) installed and running
- A free **Groq API key** — get one at [console.groq.com](https://console.groq.com) (takes ~30 seconds, no credit card)
- A free **Gemini API key** (recommended) — get one at [aistudio.google.com](https://aistudio.google.com) for higher quality embeddings. Without it, the system falls back to local fastembed.
- `ffmpeg` on your system **only if** you want to test video uploads outside Docker (inside Docker it is already included)

That is it. Python, MongoDB, Redis, and Qdrant are all managed inside Docker — you do not need to install them locally.

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
GEMINI_API_KEY=your_gemini_key_here   # optional — omit to use local fastembed
```

All other values have working defaults for Docker Compose and do not need to be changed to get started.

### 3. Start all services via Docker Compose

```bash
docker-compose up --build
```

This starts four containers: the FastAPI app (running on Granian), MongoDB, Redis, and Qdrant.

### 4. Or run locally on Windows (Development)

```bash
pip install -r requirements.txt
python -m granian --interface asgi app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Verify service health

```bash
curl http://localhost:8000/health
```

Expected response (all three backing services must be `"connected"`):

```json
{
  "status": "ok",
  "mongodb": "connected",
  "redis": "connected",
  "qdrant": "connected"
}
```

### 6. Open the interactive API docs

Navigate to **http://localhost:8000/docs** in your browser. Every endpoint is documented there with a live "Try it out" interface.

---

## Environment variables

Copy `.env.example` to `.env` and fill in the values marked **Required**.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | API key from console.groq.com |
| `MONGODB_URL` | No | `mongodb://localhost:27017` | MongoDB connection URI |
| `MONGODB_DB_NAME` | No | `palmmind` | MongoDB database name |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection string |
| `QDRANT_URL` | No | `http://localhost:6333` | Qdrant HTTP endpoint |
| `QDRANT_COLLECTION` | No | `documents` | Vector collection name |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Fastembed model name |
| `GROQ_CHAT_MODEL` | No | `groq/compound` | Main answer generation model |
| `GROQ_FAST_MODEL` | No | `groq/compound-mini` | Used for query rewriting and field extraction |
| `GROQ_WHISPER_MODEL` | No | `whisper-large-v3-turbo` | Transcription model for video/audio |
| `MAX_UPLOAD_MB` | No | `50` | Max upload file size in megabytes |
| `ENV` | No | `development` | `development` or `production` |

---

## API usage

### Ingest a document

Upload a PDF or TXT file. The response comes back immediately with a `document_id`.

```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest \
  -F "file=@/path/to/your/document.pdf" \
  -F "chunking_strategy=recursive"
```

Available chunking strategies: `recursive` (recommended), `fixed`.

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
  -F "chunking_strategy=recursive"
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

No special endpoint — just say it in plain English within the same chat session. The system understands relative dates (like "tomorrow" or "next Monday") and seamlessly switches between booking mode and general Q&A. You can ask a document question in the middle of a booking, and it will answer without losing your booking progress!

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
    "date": "2026-09-07",
    "time": "10:00:00",
    "status": "confirmed"
  }
}
```

---

## Running tests

```bash
# Run the full test suite
pytest -v

# Check linting
ruff check .

# Check formatting
ruff format .
```

Unit tests mock external connections, so you can run `pytest` without any services running locally.

---

## Project structure

```
palmRAG/
├── app/
│   ├── api/v1/          # FastAPI route handlers (documents, chat)
│   ├── core/            # Config (Pydantic Settings), logging
│   ├── db/              # Motor (MongoDB) client and index setup
│   ├── integrations/    # Groq, Qdrant, Redis client wrappers
│   ├── repositories/    # All database access (no raw queries in routes)
│   ├── schemas/         # Pydantic v2 request/response models
│   ├── services/        # Business logic: ingestion, RAG, booking, chunking, embeddings
│   └── main.py          # FastAPI app factory, lifespan, and health check
├── tests/               # pytest test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

## Stopping the services

```bash
# Stop containers, keep data volumes
docker-compose down

# Stop containers and delete all data permanently
docker-compose down -v
```

---

## License

This project is provided for evaluation and reference purposes.
