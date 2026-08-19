from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import config
from core.ingest import load_index
from core.retrieval import DiabetesRetriever
from core.generation import LocalLLM

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

app = FastAPI(title="Diabetes Clinical Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
llm = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = config.TOP_K


class Source(BaseModel):
    content: str
    document: str
    page: int
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    confidence: float
    is_confident: bool
    timestamp: str


@app.on_event("startup")
async def startup():
    global retriever, llm
    
    logger.info("Starting Diabetes Chatbot...")
    
    try:
        vectordb = load_index()
        retriever = DiabetesRetriever(vectordb)
        llm = LocalLLM()
        logger.info("All components ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        results = retriever.retrieve(request.question, k=request.top_k)
        
        if not results:
            return QueryResponse(
                question=request.question,
                answer="No relevant information found in the guidelines.",
                sources=[],
                confidence=0.0,
                is_confident=False,
                timestamp=datetime.now().isoformat()
            )
        
        is_confident, max_score = retriever.check_confidence(results)
        
        context = retriever.prepare_context(results)
        response = llm.generate(request.question, context)
        
        sources = [
            Source(
                content=s["content"][:300] + "...",
                document=s["metadata"].get("document_name", "Unknown"),
                page=s["metadata"].get("page_number", 0),
                score=s["score"]
            )
            for s in results[:3]
        ]
        
        return QueryResponse(
            question=request.question,
            answer=response["answer"],
            sources=sources,
            confidence=max_score,
            is_confident=is_confident,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy" if retriever else "initializing",
        "chunks": retriever.get_total_chunks() if retriever else 0
    }