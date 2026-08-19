from typing import List, Dict, Tuple, Optional
from langchain_chroma import Chroma
import re

import config


class DiabetesRetriever:
    def __init__(self, vectordb: Chroma):
        self.vectordb = vectordb
    
    def expand_query(self, query: str) -> List[str]:
        if not config.ENABLE_QUERY_EXPANSION:
            return [query]
        
        expansions = {
            "screening": ["test", "detection", "diagnosis", "check", "examination", "assessment"],
            "diabetes": ["diabetes mellitus", "type 2", "t2dm", "hyperglycemia", "high blood sugar"],
            "blood pressure": ["hypertension", "bp", "systolic", "diastolic", "high blood pressure"],
            "medication": ["drug", "treatment", "therapy", "pharmaceutical", "medicine", "prescription"],
            "target": ["goal", "level", "range", "value", "threshold", "cutoff"],
            "risk": ["factor", "predisposition", "susceptibility", "likelihood", "probability"],
            "management": ["control", "treatment", "care", "therapy", "intervention"],
            "glucose": ["blood sugar", "glycemia", "glycemic", "fasting glucose", "postprandial"],
            "insulin": ["blood sugar", "glycemic control", "hormone", "pancreatic"],
            "a1c": ["hba1c", "glycated hemoglobin", "glycosylated hemoglobin", "hemoglobin a1c"],
            "prevention": ["prevent", "avoid", "delay", "reduce risk", "prophylaxis"],
            "guideline": ["recommendation", "protocol", "standard", "policy", "practice"],
            "treatment": ["therapy", "intervention", "management", "care", "approach"],
            "hypertension": ["high blood pressure", "bp", "systolic", "diastolic"],
            "cholesterol": ["lipid", "hyperlipidemia", "ldl", "hdl", "triglycerides"],
            "exercise": ["physical activity", "fitness", "workout", "walking", "aerobic"],
            "diet": ["nutrition", "eating", "food", "meal", "dietary", "nutritional"],
            "diagnosis": ["diagnostic", "identify", "detect", "determine", "assess"],
            "complication": ["side effect", "condition", "disease", "disorder", "problem"]
        }
        
        expanded = [query]
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for word in query_words:
            word_clean = re.sub(r'[^a-z]', '', word)
            for key, synonyms in expansions.items():
                if key in word_clean or word_clean in key:
                    for syn in synonyms[:2]:
                        new_query = query.replace(word, syn) if syn in query else f"{query} {syn}"
                        if len(new_query) < len(query) + 30:
                            expanded.append(new_query)
                    break
        
        expanded = list(set(expanded))[:config.MAX_EXPANDED_QUERIES + 1]
        
        return expanded
    
    def keyword_search(self, query: str, chunks: List[Dict]) -> List[Dict]:
        keywords = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out',
                     'use', 'way', 'who', 'why', 'has', 'his', 'how', 'its', 'let', 'may', 'new', 'now', 'old', 'see', 'too',
                     'any', 'ask', 'big', 'day', 'end', 'few', 'get', 'god', 'got', 'hot', 'job', 'man', 'men', 'own', 'pay',
                     'put', 'run', 'set', 'sit', 'son', 'ten', 'top', 'try', 'two', 'war', 'way', 'why', 'yes'}

        keywords = [k for k in keywords if k not in stopwords]
        
        if not keywords:
            return chunks
        
        scored = []
        for chunk in chunks:
            content_lower = chunk["content"].lower()
            score = 0
            
            for kw in keywords:
                if kw in content_lower:
                    score += 1
                    score += content_lower.count(kw) * 0.2
            
            metadata = chunk.get("metadata", {})
            doc_name = metadata.get("document_name", "").lower()
            if any(kw in doc_name for kw in keywords):
                score += 2
            
            scored.append((chunk, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, s in scored[:len(chunks)] if s > 0]
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        k = k or config.TOP_K
        
        expanded_queries = self.expand_query(query)
        
        all_semantic_results = []
        seen_chunks = set()
        
        for q in expanded_queries:
            results = self.vectordb.similarity_search_with_relevance_scores(q, k=k*2)
            
            for doc, score in results:
                chunk_id = doc.metadata.get("chunk_id", "")
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_semantic_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                        "chunk_id": chunk_id,
                        "semantic_score": score
                    })
        
        if not all_semantic_results:
            return []
        
        all_semantic_results.sort(key=lambda x: x["semantic_score"], reverse=True)
        
        top_semantic = all_semantic_results[:k*2]
        
        keyword_results = self.keyword_search(query, top_semantic)
        keyword_ids = {r["chunk_id"] for r in keyword_results}
        
        combined = {}
        
        for r in all_semantic_results[:k*2]:
            chunk_id = r["chunk_id"]
            combined[chunk_id] = r
            combined[chunk_id]["score"] = r["semantic_score"] * config.SEMANTIC_WEIGHT
            combined[chunk_id]["keyword_score"] = 0
        
        for r in keyword_results:
            chunk_id = r["chunk_id"]
            if chunk_id in combined:
                combined[chunk_id]["score"] += config.KEYWORD_WEIGHT
                combined[chunk_id]["keyword_score"] = config.KEYWORD_WEIGHT
        
        results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        
        formatted = []
        for r in results[:k]:
            formatted.append({
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r["score"],
                "chunk_id": r["chunk_id"],
                "semantic_score": r.get("semantic_score", 0),
                "keyword_score": r.get("keyword_score", 0)
            })
        
        return formatted
    
    def check_confidence(self, results: List[Dict]) -> Tuple[bool, float]:
        if not results:
            return False, 0.0
        
        max_score = max(r["score"] for r in results)
        is_confident = max_score >= config.SIMILARITY_THRESHOLD
        
        return is_confident, max_score
    
    def prepare_context(self, chunks: List[Dict], max_chunks: int = 4) -> str:
        parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            doc = chunk["metadata"].get("document_name", "Unknown")
            page = chunk["metadata"].get("page_number", "?")
            doc_type = chunk["metadata"].get("document_type", "guideline")
            parts.append(
                f"Passage {i} ({doc}, {doc_type}, page {page}):\n{chunk['content'][:700]}"
            )
        return "\n\n".join(parts)
    
    def get_total_chunks(self) -> int:
        try:
            return self.vectordb._collection.count()
        except:
            return 0