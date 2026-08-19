from typing import Dict
import logging
import re

import config

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self):
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading LLM: {config.LLM_MODEL}")
            
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            
            tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.LLM_MODEL,
                device_map="cpu"
            )
            
            self.pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                do_sample=False
            )
            
            print(f"   LLM loaded")
            
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            print("   Using extraction mode (no LLM)")
            self.pipeline = None
    
    def generate(self, question: str, context: str) -> Dict[str, str]:
        # Try LLM first
        if self.pipeline is not None:
            try:
                prompt = self._build_prompt(question, context)
                response = self.pipeline(prompt)[0]["generated_text"]
                
                # Check if response is useful
                if len(response.strip()) > 20 and "don't have enough" not in response.lower():
                    return self._parse_response(response)
            except Exception as e:
                print(f"Generation error: {e}")
        
        # Fallback: extract directly from context
        return self._extract_from_context(question, context)
    
    def _extract_from_context(self, question: str, context: str) -> Dict[str, str]:
        """Extract answer directly from context"""
        
        # Find the most relevant passage
        passages = re.split(r'Passage \d+ \([^)]+\):\n', context)
        passages = [p.strip() for p in passages if p.strip()]
        
        if not passages:
            return self._fallback_response(question)
        
        # Use first passage as answer
        best_passage = passages[0]
        
        # Try to find a sentence that answers the question
        sentences = re.split(r'[.!?]+', best_passage)
        best_sentence = ""
        
        # Keywords to look for
        keywords = ["screening", "diabetes", "recommend", "guideline", "target", "blood pressure", "HbA1c"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(kw in sentence.lower() for kw in keywords):
                if len(sentence) > 20:
                    best_sentence = sentence
                    break
        
        if not best_sentence:
            # Take first sentence
            best_sentence = sentences[0].strip() if sentences else best_passage[:200]
        
        return {
            "answer": f"{best_sentence}\n\n(Source: extracted directly from guidelines)",
            "recommendation": best_sentence,
            "evidence": best_passage[:300] + "...",
            "citation": "USPSTF Guidelines"
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""Answer the question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        return {
            "answer": response,
            "recommendation": response,
            "evidence": "",
            "citation": ""
        }
    
    def _fallback_response(self, question: str) -> Dict[str, str]:
        return {
            "answer": "I don't have enough information to answer this question based on the available guidelines.",
            "recommendation": "Insufficient information in the provided context.",
            "evidence": "",
            "citation": ""
        }