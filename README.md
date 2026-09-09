# 🤖 Hiver AI Twitter Support Agent

An AI-powered customer support agent trained on real Twitter customer support data for **AppleSupport**, demonstrating intent classification, RAG-grounded reply drafting, and intelligent escalation routing.

Built as part of the HiverHQ AI Support Agent assignment.

---

## 📋 Table of Contents
- [Quick Start (< 15 min)](#-quick-start)
- [Environment Setup](#-environment-setup)
- [Project Structure](#-project-structure)
- [Phase Guide](#-phase-guide)
- [Running Evaluations](#-running-evaluations)
- [Deployment](#-deployment)
- [Report](#-report)

---

## ⚡ Quick Start

Reproduce headline results in under 15 minutes:

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/hiver-support-agent.git
cd hiver-support-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your API keys (see Environment Setup below)

# 5. Download and process data
python src/data/download.py
python src/data/preprocess.py

# 6. Run the full evaluation on golden eval set
python src/evaluation/run_eval.py --split golden_eval

# 7. Start the API + Frontend
docker compose up
# OR run separately:
# uvicorn src.api.main:app --reload  (backend)
# cd frontend && npm run dev         (frontend)
```

---

## 🔑 Environment Setup

Copy `.env.example` to `.env` and fill in each key. Detailed instructions are inside `.env.example`.

| Variable | Required | Service | Free? |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) | ✅ Free |
| `KAGGLE_USERNAME` | ✅ Yes (data download) | [kaggle.com/settings](https://www.kaggle.com/settings) | ✅ Free |
| `KAGGLE_KEY` | ✅ Yes (data download) | [kaggle.com/settings](https://www.kaggle.com/settings) | ✅ Free |
| `QDRANT_URL` | ✅ Yes (production) | [cloud.qdrant.io](https://cloud.qdrant.io) | ✅ Free 1GB |
| `QDRANT_API_KEY` | ✅ Yes (production) | [cloud.qdrant.io](https://cloud.qdrant.io) | ✅ Free |
| `HF_TOKEN` | Optional | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Free |
| `MISTRAL_API_KEY` | Optional (fallback LLM) | [console.mistral.ai](https://console.mistral.ai) | ✅ Free |

---

## 📁 Project Structure

```
hiver-support-agent/
├── src/
│   ├── config.py               # Central config (Pydantic Settings)
│   ├── data/
│   │   ├── download.py         # Kaggle dataset download
│   │   ├── preprocess.py       # Filter, clean, reconstruct threads
│   │   └── eda.py              # Exploratory data analysis
│   ├── classifier/
│   │   ├── intents.py          # Intent taxonomy (10 intents)
│   │   ├── build_eval_set.py   # Golden eval set builder
│   │   ├── keyword_classifier.py   # Baseline: keyword/regex
│   │   ├── setfit_classifier.py    # SetFit few-shot classifier
│   │   └── llm_classifier.py       # LLM-based classifier (Groq)
│   ├── rag/
│   │   ├── embeddings.py       # Sentence transformer embeddings
│   │   ├── vector_store.py     # ChromaDB / Qdrant store
│   │   └── retriever.py        # Retrieval pipeline
│   ├── agent/
│   │   ├── drafter.py          # RAG-grounded reply drafter
│   │   ├── router.py           # Escalation router
│   │   └── orchestrator.py     # Main agent pipeline
│   ├── evaluation/
│   │   ├── metrics.py          # Classification + reply metrics
│   │   ├── llm_judge.py        # LLM-as-judge rubric
│   │   └── run_eval.py         # Full evaluation harness
│   └── api/
│       └── main.py             # FastAPI app
├── frontend/                   # React + TypeScript + Vite
├── data/
│   ├── raw/                    # Raw Kaggle CSV (not in git)
│   ├── processed/              # Filtered brand conversations
│   └── golden_eval/            # 200 hand-labelled examples ✅ in git
├── tests/                      # Pytest test suite
├── report/
│   ├── report.md               # 6-page report
│   └── decision_log.md         # 15 non-obvious decisions
├── .env.example                # Full env setup guide
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

---

## 🔄 Phase Guide

### Phase 1: Data
```bash
python src/data/download.py         # Download from Kaggle (~500MB)
python src/data/preprocess.py       # Filter & process
python src/data/eda.py              # Run EDA analysis
```

### Phase 2: Intent Labelling
```bash
python src/classifier/build_eval_set.py build    # Create 200-example CSV
# → Manually label data/golden_eval/golden_eval_set.csv
python src/classifier/build_eval_set.py validate # Check labelling
```

### Phase 3: Train Agent
```bash
python src/classifier/setfit_classifier.py train
python src/rag/vector_store.py build
```

### Phase 4: Evaluate
```bash
python src/evaluation/run_eval.py --split golden_eval --output results/
```

---

## 📊 Running Evaluations

```bash
# Full eval (classification + reply quality + escalation + LLM judge)
python src/evaluation/run_eval.py

# Individual components
python src/evaluation/metrics.py --metric classification
python src/evaluation/metrics.py --metric reply_quality
python src/evaluation/llm_judge.py --sample 50
```

Results are saved to `results/` directory as JSON files.

---

## 🚀 Deployment

### Backend → Render (Free)
1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
6. Add all env variables from `.env` in Render dashboard

### Frontend → Vercel (Free)
1. Go to [vercel.com](https://vercel.com) → New Project
2. Import GitHub repo → select `frontend/` as root directory
3. Framework: Vite
4. Add env var: `VITE_API_URL=https://your-render-url.onrender.com`
5. Deploy

### Local (Docker)
```bash
docker compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

---

## 📄 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Groq (Llama 3.1 70B) | Free, fast, high quality |
| Embeddings | sentence-transformers (local) | Zero cost, no API |
| Vector DB | Qdrant Cloud | Free 1GB, production-ready |
| Intent Classifier | SetFit + Groq | Few-shot, no big labeled set needed |
| Backend | FastAPI + Python | Hiver's stack, async |
| Frontend | React + TypeScript + Vite | Hiver's stack |
| Storage | SQLite | Zero setup, ships with Python |
| Deployment | Render + Vercel | Free tier, auto-deploy from Git |
| Containers | Docker Compose | One-command local run |
