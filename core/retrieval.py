from typing import List, Dict, Tuple, Optional
from langchain_chroma import Chroma

import config


class DiabetesRetriever:
    def __init__(self, vectordb: Chroma):
        self.vectordb = vectordb
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        k = k or config.TOP_K
        
        results = self.vectordb.similarity_search_with_relevance_scores(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "chunk_id": doc.metadata.get("chunk_id", "unknown")
            }
            for doc, score in results
        ]
    
    def check_confidence(self, results: List[Dict]) -> Tuple[bool, float]:
        if not results:
            return False, 0.0
        
        max_score = max(r["score"] for r in results)
        is_confident = max_score >= config.SIMILARITY_THRESHOLD
        
        return is_confident, max_score
    
    def prepare_context(self, chunks: List[Dict], max_chunks: int = 3) -> str:
        parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            doc = chunk["metadata"].get("document_name", "Unknown")
            page = chunk["metadata"].get("page_number", "?")
            parts.append(
                f"Passage {i} ({doc}, page {page}):\n{chunk['content'][:500]}"
            )
        return "\n\n".join(parts)
    
    def get_total_chunks(self) -> int:
        try:
            return self.vectordb._collection.count()
        except:
            return 0