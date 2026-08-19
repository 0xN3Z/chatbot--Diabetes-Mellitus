from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw_pdfs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
EVAL_DIR = BASE_DIR / "eval"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "google/flan-t5-small"
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.0

TOP_K = 5
SIMILARITY_THRESHOLD = 0.65

BATCH_SIZE = 32