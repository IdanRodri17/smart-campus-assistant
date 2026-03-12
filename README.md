# 🎓 Smart Campus Assistant

AI-powered campus information assistant built with RAG (Retrieval-Augmented Generation). Students ask natural-language questions about office hours, exam schedules, rooms, and campus services — and receive accurate, AI-generated answers.

**CyberPro AI-Powered Development Bootcamp** | ELAD Software — Hackathon Project

---

## Architecture

```
[React Frontend] → [FastAPI Backend] → [AI Layer] → [Supabase + pgvector]
                                           │
                              ┌─────────────┼─────────────┐
                              │             │             │
                         Classifier    RAG Service    LLM Generator
                        (categorize)  (vector search)  (answer gen)
```

**Tech Stack:**

| Layer | Technology |
|---|---|
| Frontend | HTML / CSS / JavaScript |
| Backend | Python + FastAPI |
| AI / LLM | OpenAI API (GPT-4o-mini) |
| Database | Supabase (PostgreSQL + pgvector) |
| Containers | Docker + Docker Compose |
| CI/CD | GitHub Actions + Trivy security scan |
| Monitoring | Prometheus + Grafana |
| IaC | Terraform (Supabase provisioning) |
| Testing | pytest (32 unit tests) |

---

## Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API key
- Supabase project (free tier works)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/smart-campus-assistant.git
cd smart-campus-assistant/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create `backend/.env`:

```env
# Supabase (from: supabase.com/dashboard → Settings → API)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres.your-project:password@aws-0-region.pooler.supabase.com:6543/postgres

# OpenAI (from: platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-key

# App Config
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CONFIDENCE_THRESHOLD=0.7
RAG_TOP_K=5

# Environment
ENV=development
DEBUG=true
```

### 3. Seed the Database

```bash
cd backend
python -m app.services.seed
```

This creates the database schema (with pgvector), inserts 16 campus data records, and generates vector embeddings for RAG.

### 4. Run the Server

```bash
uvicorn app.main:app --reload
```

- **API docs:** http://localhost:8000/docs
- **Prometheus metrics:** http://localhost:8000/metrics

### 5. Open the Frontend

```bash
cd frontend
python -m http.server 3000
```

Open http://localhost:3000 in your browser.

---

## Docker (Full Stack)

Run the entire system with one command:

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 (admin / campus2026) |

---

## AI Pipeline

The main `POST /api/ask` endpoint processes questions through a 6-step pipeline:

1. **Input Validation** — Length check, encoding, prompt injection detection
2. **Classification** — LLM categorizes question: `schedule` | `general_info` | `technical_issue` | `out_of_scope`
3. **RAG Retrieval** — Embeds question → pgvector similarity search with metadata filtering by category → two-pass strategy (filtered first, unfiltered fallback)
4. **Structured Data** — Queries campus_data table by classified category
5. **LLM Generation** — Versioned system prompt + context → answer with confidence score
6. **Confidence Gate** — If confidence ≥ 0.7: return answer. If < 0.7: return fallback with staff contact info

### Prompt Versioning

Two system prompt versions (`v1`, `v2`) enable A/B testing. Each interaction logs which prompt version was used, enabling data-driven prompt improvement.

### Security

- Prompt injection detection (9 patterns)
- Input sanitization before AI processing
- System prompt guards ("You are ONLY a campus assistant")
- API keys managed via `.env` (never committed to git)

---

## Testing

```bash
cd backend
pytest tests/test_assistant.py -v
```

32 tests covering: input validation, prompt injection detection, classification, schema validation, RAG helpers, prompt versioning, and mocked LLM responses.

---

## Monitoring

Prometheus scrapes FastAPI metrics at `/metrics`. Pre-configured Grafana dashboard shows:

- Total API requests
- Request rate per minute
- P50 / P95 / P99 response times
- HTTP status code distribution
- Error rate percentage

---

## Project Structure

```
smart-campus-assistant/
├── backend/
│   ├── app/
│   │   ├── ai/              # AI pipeline modules
│   │   │   ├── classifier.py    # Question categorization
│   │   │   ├── rag.py           # RAG retrieval + pgvector search
│   │   │   ├── generator.py     # LLM answer generation (v1/v2 prompts)
│   │   │   └── orchestrator.py  # Pipeline orchestration + logging
│   │   ├── core/             # Config & database
│   │   ├── models/           # Pydantic schemas
│   │   ├── routes/           # API endpoints
│   │   └── services/         # Seed script
│   ├── tests/
│   │   └── test_assistant.py # 32 unit tests
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── monitoring/
│   ├── prometheus/
│   └── grafana/
├── terraform/
│   └── main.tf
├── docs/                     # SRS, diagrams, experiments, work log
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | Supabase anonymous key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service role key |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `LLM_MODEL` | No | LLM model name (default: gpt-4o-mini) |
| `EMBEDDING_MODEL` | No | Embedding model (default: text-embedding-3-small) |
| `CONFIDENCE_THRESHOLD` | No | Fallback threshold (default: 0.7) |
| `RAG_TOP_K` | No | Number of RAG results (default: 5) |

---

## Author

**Idan Rodrigez** — CyberPro AI-Powered Development Bootcamp

Mentor: Ishay Elimelech
