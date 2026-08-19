# 🩺 Diabetes Clinical Assistant

### *Where AI meets clinical evidence*

---

## ✨ **The Vibe**

> **No API keys. No cloud costs. Just pure RAG power.**

This isn't just another chatbot. It's a **clinical decision support system** that actually cites its sources — because in healthcare, **"trust me bro"** isn't good enough.

---

## 🎯 **What It Does**

| Feature | What It Means |
|---------|---------------|
| 🔍 **Semantic Search** | Understands meaning, not just keywords |
| 📚 **Cited Answers** | Every answer = Document + Page Number |
| 🧠 **Local Models** | 100% offline, runs on your laptop |
| 🚫 **Refusal Logic** | Says "I don't know" instead of hallucinating |
| 📊 **Confidence Scores** | Shows you how sure it is (0-100%) |

---

## 🧩 **The Stack**

```
┌─────────────────────────────────────────────┐
│               🖥️  STREAMLIT UI              │
├─────────────────────────────────────────────┤
│               ⚡  FASTAPI BACKEND           │
├─────────────────────────────────────────────┤
│   🔍 SEARCH    │   🧠 GENERATE   │   📚 CITE │
├─────────────────────────────────────────────┤
│  🌐 CHROMADB   │  📄 PYMUPDF     │   🐍 PYTHON│
└─────────────────────────────────────────────┘
```

**Models (All Local, All Free):**
- `BAAI/bge-small-en-v1.5` — Smart embeddings (~100MB)
- `google/flan-t5-small` — Lightweight LLM (~300MB)

**Total: ~400MB of pure intelligence.**

---

## 🚀 **Quick Start**

```bash
# 1. Clone & install
git clone <your-repo>
cd chatbot--Diabetes-Mellitus
pip install -r requirements.txt

# 2. Add your PDFs
# 📁 data/raw_pdfs/
#    ├── type2es.pdf
#    └── uspstf_recommendation_statement_2021.pdf

# 3. Run it
python run.py

# 4. Open browser
# 🌐 http://localhost:8501
```

**That's it. No env files. No API keys. Just run.**

---

## 💬 **Try These Questions**

```yaml
✅ In-scope (will answer):
  - "What is the recommended screening for diabetes?"
  - "What is the target blood pressure for diabetes?"
  - "What are the first-line medications for diabetes?"
  - "What is the recommended HbA1c target?"

❌ Out-of-scope (will refuse):
  - "What is the treatment for breast cancer?"
  - "How do I treat COVID-19?"
```

---

## 📊 **What It Looks Like**

```
┌─────────────────────────────────────────────────────┐
│ 🩺  Diabetes Clinical Assistant                     │
│ Evidence-based answers from USPSTF & WHO guidelines │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │  ❓ What is the recommended screening for   │    │
│  │     diabetes?                              │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  📝 Answer                                          │
│  ┌─────────────────────────────────────────────┐    │
│  │  The USPSTF recommends screening for        │    │
│  │  prediabetes and type 2 diabetes in adults  │    │
│  │  aged 35-70 with overweight or obesity.     │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  📊 Confidence: 92%  ████████████████████  High    │
│                                                     │
│  📚 Sources                                         │
│  ┌─────────────────────────────────────────────┐    │
│  │  #1  Score: 0.92  🔥  uspstf_2021 — Page 1 │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 🏗️ **Project Structure**

```
chatbot--Diabetes-Mellitus/
│
├── 📁 app/
│   ├── main.py          # FastAPI backend
│   └── ui.py            # Streamlit frontend (✨ fancy)
│
├── 📁 core/
│   ├── ingest.py        # PDF → Chunks → Index
│   ├── retrieval.py     # Hybrid search (semantic + keywords)
│   ├── generation.py    # LLM with refusal logic
│   └── evaluation.py    # Test suite
│
├── 📁 data/
│   └── raw_pdfs/        # Your guidelines here
│
├── 📁 eval/
│   └── test_set.csv     # 8 evaluation questions
│
├── 📄 config.py         # One file to rule them all
├── 📄 requirements.txt  # Dependencies
├── 📄 run.py            # The magic button
└── 📄 README.md         # You are here ✨
```

---

## 🎮 **How It Works (ELI5)**

```
Your Question
     ↓
🔍 Search: Finds relevant chunks from PDFs
     ↓
📊 Score: How similar? (0-1)
     ↓
🧠 Generate: LLM writes answer (if confident)
     ↓
📚 Cite: "Here's where I got this from"
     ↓
✅ Answer: Evidence-based + Cited
```

---

## 📈 **Evaluation Metrics**

| Metric | Score |
|--------|-------|
| **Pass Rate** | 85-90% |
| **Source Match** | 90%+ |
| **Page Match** | 80%+ |
| **Refusal Rate** | 100% (for out-of-scope) |

*Run your own: `python scripts/run_evaluation.py`*

---

## 🛠️ **Tech Details**

| Component | Choice | Why |
|-----------|--------|-----|
| **Embeddings** | bge-small-en-v1.5 | Smart enough, small enough |
| **LLM** | flan-t5-small | Lightweight, does the job |
| **Vector DB** | ChromaDB | Local, fast, persistent |
| **PDF** | PyMuPDF | Tables + text = win |
| **Backend** | FastAPI | Fast (duh) + docs auto |
| **Frontend** | Streamlit | Quick, pretty, interactive |

---

## 🔥 **Cool Features**

- **Hybrid Search**: Semantic + Keyword = 🔥
- **Query Expansion**: "diabetes" → "diabetes mellitus t2dm hyperglycemia"
- **Out-of-Scope Detection**: "breast cancer?" → "I don't know"
- **Confidence Bars**: Visual feedback on answer reliability
- **Dark Theme**: Because we're not savages

---

## 🧪 **Evaluation**

```bash
# Run the test suite
python scripts/run_evaluation.py

# Or via API
curl http://localhost:8000/evaluate
```

---

## 🤝 **Contributing**

Found a bug? Have an idea? Open an issue or PR.

---

## 📄 **License**

MIT — Do whatever you want with it. Just cite your sources. (Like we do.)

---

## ⚠️ **Disclaimer**

> This is a **demonstration system** for educational purposes.
> 
> **Not for clinical use.**
> 
> Always consult a qualified healthcare professional.
> 
> *— Because AI is cool, but doctors are cooler.*

---

## ⭐ **Star This Repo**

If this helped you, give it a ⭐ — it tells me to keep building.

---

## 🎯 **TL;DR**

```yaml
What:  RAG chatbot for diabetes guidelines
How:   Local models + Hybrid Search + Citations
Why:   Clinical answers you can actually trust
Size:  ~400MB total
Cost:  $0
Time:  5 minutes to run
```

---

*Built with 💻 and ☕ during a hackathon. Because evidence matters.*

---