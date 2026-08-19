
```markdown
# Diabetes Clinical Assistant

Evidence-based clinical decision support system for diabetes mellitus using WHO and USPSTF guidelines.

---

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that answers clinical questions about diabetes using official guidelines. Every answer includes citations with document name and page number.

**Key Features:**
- Semantic search using local embeddings
- Cited answers with page references
- Out-of-scope detection and refusal
- Confidence scoring for every answer
- 100% local - no API keys required
- Lightweight models (~400MB total)

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Embeddings | BAAI/bge-small-en-v1.5 (100MB) |
| LLM | google/flan-t5-small (300MB) |
| Vector DB | ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |

---

## Project Structure

```
chatbot--Diabetes-Mellitus/
│
├── app/
│   ├── main.py          # FastAPI backend
│   └── ui.py            # Streamlit frontend
│
├── core/
│   ├── ingest.py        # PDF processing and indexing
│   ├── retrieval.py     # Semantic search with hybrid
│   ├── generation.py    # LLM with refusal logic
│   └── evaluation.py    # Test suite
│
├── data/
│   └── raw_pdfs/
│       ├── dsa698.pdf
│       ├── type2es.pdf
│       └── uspstf_recommendation_statement_2021.pdf
│
├── eval/
│   └── test_set.csv
│
├── scripts/
│   └── run_evaluation.py
│
├── config.py
├── requirements.txt
├── run.py
└── README.md
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd chatbot--Diabetes-Mellitus

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
python run.py
```

### Access

| Service | URL |
|---------|-----|
| Web UI | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## Configuration

Edit `config.py` to customize:

```python
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 6
SIMILARITY_THRESHOLD = 0.50

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "google/flan-t5-small"

ENABLE_HYBRID_SEARCH = True
ENABLE_QUERY_EXPANSION = True
```

---

## Example Questions

**In-scope (will answer):**
- What is the recommended screening for diabetes?
- What is the target blood pressure for diabetes?
- What are the first-line medications for diabetes?
- What is the recommended HbA1c target?

**Out-of-scope (will refuse):**
- What is the treatment for breast cancer?
- How do I treat COVID-19?

---

## API Endpoints

### POST /query

Ask a clinical question.

**Request:**
```json
{
    "question": "What is the recommended screening for diabetes?",
    "top_k": 5
}
```

**Response:**
```json
{
    "question": "What is the recommended screening for diabetes?",
    "answer": "The USPSTF recommends screening for prediabetes...",
    "sources": [
        {
            "content": "...",
            "document": "uspstf_recommendation_statement_2021",
            "page": 1,
            "score": 0.85
        }
    ],
    "confidence": 0.85,
    "is_confident": true,
    "is_out_of_scope": false,
    "timestamp": "2024-01-15T10:30:00"
}
```

### GET /health

Check system status.

### GET /evaluate

Run evaluation suite.

---

## Evaluation

Run the test suite:

```bash
python scripts/run_evaluation.py
```

Or via API:

```bash
curl http://localhost:8000/evaluate
```

---

## Troubleshooting

**ChromaDB disk full error:**
```bash
Remove-Item -Recurse -Force "data/chroma_db"
python run.py
```

**LLM loading issues:**
- First load may take 1-2 minutes
- Model is cached after first download

**Memory issues:**
- Reduce CHUNK_SIZE in config.py
- Use smaller embedding model: `all-MiniLM-L6-v2`

---

## License

MIT License

---

## Disclaimer

This is a demonstration system for educational purposes only. Not for direct clinical use. Always consult a qualified healthcare professional for clinical decisions.

---

## Credits

Based on guidelines from:
- World Health Organization (WHO)
- US Preventive Services Task Force (USPSTF)
```