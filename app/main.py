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
from core.evaluation import run_evaluation, load_evaluation_results

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
    is_out_of_scope: bool = False
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
                answer="I don't have enough information to answer this question. This system only provides answers based on diabetes guidelines from USPSTF and WHO.",
                sources=[],
                confidence=0.0,
                is_confident=False,
                is_out_of_scope=True,
                timestamp=datetime.now().isoformat()
            )
        
        is_confident, max_score = retriever.check_confidence(results)
        
        context = retriever.prepare_context(results)
        response = llm.generate(request.question, context)
        
        is_out_of_scope = response.get("is_out_of_scope", False)
        
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
            sources=sources if not is_out_of_scope else [],
            confidence=max_score if not is_out_of_scope else 0.0,
            is_confident=is_confident and not is_out_of_scope,
            is_out_of_scope=is_out_of_scope,
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


@app.get("/evaluate")
async def evaluate():
    """Run evaluation and return results"""
    if not retriever:
        return {"error": "Retriever not initialized"}
    
    try:
        results = run_evaluation(retriever, llm)
        return results
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {"error": str(e)}


@app.get("/evaluation/results")
async def get_evaluation_results():
    """Get saved evaluation results"""
    results = load_evaluation_results()
    if not results:
        return {"status": "No evaluation results found. Run /evaluate first."}
    return results