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
            
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                device_map="cpu"
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                do_sample=False
            )
            
            print(f"   LLM loaded")
            
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            print("   Using extraction mode")
            self.pipeline = None
    
    def _is_diabetes_question(self, question: str) -> bool:
        """Check if question is specifically about diabetes"""
        
        # Core diabetes keywords - MUST have at least one
        diabetes_core = [
            'diabetes', 'diabetic', 'type 2', 'type2', 't2dm',
            'prediabetes', 'pre-diabetes', 'hyperglycemia',
            'hypoglycemia', 'glucose', 'insulin', 'a1c', 'hba1c',
            'blood sugar', 'fasting glucose', 'ogtt',
            'metformin', 'sulfonylurea', 'glycemic',
            'retinopathy', 'nephropathy', 'neuropathy',
            'ketoacidosis', 'mmol', 'mg/dL',
            'screening for diabetes', 'diabetes management',
            'diabetes treatment', 'diabetes prevention',
            'diabetes complications', 'diabetes care'
        ]
        
        question_lower = question.lower()
        
        # Must have at least one core diabetes keyword
        has_core = any(kw in question_lower for kw in diabetes_core)
        
        if not has_core:
            return False
        
        return True
    
    def _is_clearly_out_of_scope(self, question: str) -> bool:
        """Check if question is clearly about something else"""
        
        out_of_scope = [
            'breast cancer', 'lung cancer', 'colon cancer', 'prostate cancer',
            'cancer treatment', 'chemotherapy', 'radiation therapy',
            'covid', 'coronavirus', 'vaccine', 'covid-19',
            'alzheimer', 'dementia', 'parkinson',
            'heart attack', 'stroke', 'cardiac arrest',
            'pregnancy', 'childbirth', 'obstetric',
            'pediatric', 'children', 'infant',
            'hypertension only', 'high blood pressure only',
            'cholesterol only', 'lipid only',
            'asthma', 'copd', 'lung disease',
            'kidney disease only', 'liver disease',
            'thyroid', 'osteoporosis', 'arthritis'
        ]
        
        question_lower = question.lower()
        
        # Check if question is about something else
        for kw in out_of_scope:
            if kw in question_lower:
                # Make sure it's not about diabetes complications
                if 'diabetes' not in question_lower:
                    return True
        
        return False
    
    def _is_meaningful_question(self, question: str) -> bool:
        question = question.strip()
        
        if len(question) < 3:
            return False
        
        if len(re.findall(r'[a-zA-Z]', question)) < 2:
            return False
        
        return True
    
    def _is_out_of_scope(self, question: str, context: str) -> bool:
        # Basic check
        if not self._is_meaningful_question(question):
            return True
        
        # Check if clearly out of scope
        if self._is_clearly_out_of_scope(question):
            return True
        
        # Check if it's a diabetes question
        if not self._is_diabetes_question(question):
            return True
        
        # Check if context has diabetes content
        context_lower = context.lower()
        diabetes_indicators = ['diabetes', 'glucose', 'insulin', 'a1c', 'hba1c', 'screening', 'mmol', 'mg/dL']
        has_diabetes_context = any(kw in context_lower for kw in diabetes_indicators)
        
        if not has_diabetes_context:
            return True
        
        return False
    
    def generate(self, question: str, context: str) -> Dict[str, str]:
        # Check if out of scope
        if self._is_out_of_scope(question, context):
            return self._out_of_scope_response()
        
        # Try LLM first
        if self.pipeline is not None:
            try:
                prompt = self._build_prompt(question, context)
                response = self.pipeline(prompt)[0]["generated_text"]
                
                if len(response.strip()) > 30 and "don't have enough" not in response.lower():
                    return self._parse_response(response)
            except Exception as e:
                print(f"Generation error: {e}")
        
        return self._extract_from_context(question, context)
    
    def _extract_from_context(self, question: str, context: str) -> Dict[str, str]:
        context_words = len(context.split())
        if context_words < 20:
            return self._out_of_scope_response()
        
        passages = re.split(r'Passage \d+ \([^)]+\):\n', context)
        passages = [p.strip() for p in passages if p.strip()]
        
        if not passages:
            return self._out_of_scope_response()
        
        full_text = " ".join(passages)
        
        # ===== SCREENING QUESTIONS =====
        if "screening" in question.lower():
            uspstf_pattern = r'USPSTF recommends screening for prediabetes and type 2 diabetes in adults aged (\d+) to (\d+) years who have overweight or obesity'
            match = re.search(uspstf_pattern, full_text, re.IGNORECASE)
            
            if match:
                age_start, age_end = match.groups()
                answer = f"The USPSTF recommends screening for prediabetes and type 2 diabetes in adults aged {age_start} to {age_end} years who have overweight or obesity."
                
                interval_pattern = r'screening every (\d+) years'
                interval_match = re.search(interval_pattern, full_text, re.IGNORECASE)
                if interval_match:
                    answer += f" If results are normal, screening should be repeated every {interval_match.group(1)} years."
                
                tests_pattern = r'(fasting plasma glucose|FPG|HbA1c|oral glucose tolerance test|OGTT)'
                tests = re.findall(tests_pattern, full_text, re.IGNORECASE)
                if tests:
                    unique_tests = list(set(tests))
                    answer += f" Screening tests include: {', '.join(unique_tests[:3])}."
                
                return {
                    "answer": answer,
                    "recommendation": answer,
                    "evidence": full_text[:400] + "...",
                    "citation": "USPSTF Recommendation Statement",
                    "is_out_of_scope": False
                }
            
            ada_age_match = re.search(r'adults? (\d+) years', full_text, re.IGNORECASE)
            ada_bmi_match = re.search(r'BMI [≥=] (\d+)', full_text, re.IGNORECASE)
            ada_interval_match = re.search(r'(\d+)-year intervals?', full_text, re.IGNORECASE)
            
            ada_age = ada_age_match.group(1) if ada_age_match else "45"
            ada_bmi = ada_bmi_match.group(1) if ada_bmi_match else "25"
            ada_interval = ada_interval_match.group(1) if ada_interval_match else "3"
            
            if "ADA" in full_text or "American Diabetes Association" in full_text:
                answer = f"The American Diabetes Association (ADA) recommends screening for prediabetes and diabetes in all adults aged {ada_age} years and older."
                answer += f" For adults who have overweight or obesity (BMI ≥ {ada_bmi}), screening is recommended regardless of age."
                answer += f" If results are normal, repeat screening every {ada_interval} years."
                answer += " Screening tests include: fasting plasma glucose, HbA1c, or oral glucose tolerance test."
                
                return {
                    "answer": answer,
                    "recommendation": answer,
                    "evidence": full_text[:400] + "...",
                    "citation": "ADA Guidelines",
                    "is_out_of_scope": False
                }
        
        # ===== BLOOD PRESSURE =====
        if "blood pressure" in question.lower() or "bp" in question.lower():
            bp_pattern = r'blood pressure target.*?(\d+)\s*/\s*(\d+)'
            match = re.search(bp_pattern, full_text, re.IGNORECASE)
            if match:
                sys, dia = match.groups()
                return {
                    "answer": f"The target blood pressure for diabetes is {sys}/{dia} mmHg.",
                    "recommendation": f"Target blood pressure: {sys}/{dia} mmHg",
                    "evidence": full_text[:400] + "...",
                    "citation": "Diabetes Guidelines",
                    "is_out_of_scope": False
                }
        
        # ===== GENERAL EXTRACTION =====
        sentences = re.split(r'[.!?]+', full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if not sentences:
            return self._out_of_scope_response()
        
        clinical_keywords = ['screening', 'diagnosis', 'treatment', 'management', 
                            'glucose', 'blood', 'pressure', 'target', 'recommend',
                            'guideline', 'prevention', 'control', 'insulin',
                            'metformin', 'diet', 'exercise', 'complications',
                            'retinopathy', 'nephropathy', 'neuropathy',
                            'mmol', 'mg/dL', 'HbA1c', 'A1c', 'USPSTF', 'ADA', 'WHO']
        
        scored_sentences = []
        question_words = set(question.lower().split())
        question_words = {w for w in question_words if len(w) > 3}
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            word_overlap = sum(1 for w in question_words if w in sentence_lower)
            keyword_score = sum(1 for kw in clinical_keywords if kw in sentence_lower)
            total_score = word_overlap * 2 + keyword_score
            
            if any(phrase in sentence_lower for phrase in ['recommend', 'should', 'target', 'goal', 'optimal']):
                total_score += 3
            
            if re.search(r'\d+', sentence):
                total_score += 1
            
            scored_sentences.append((sentence, total_score, i))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_sentences or scored_sentences[0][1] < 3:
            return self._out_of_scope_response()
        
        best_sentences = [s[0] for s in scored_sentences[:4] if s[1] > 0]
        
        if not best_sentences:
            return self._out_of_scope_response()
        
        best_idx = scored_sentences[0][2]
        start = max(0, best_idx - 1)
        end = min(len(sentences), best_idx + 4)
        context_sentences = sentences[start:end]
        
        if len(" ".join(context_sentences)) > len(" ".join(best_sentences)):
            best_sentences = context_sentences
        
        answer = ". ".join(best_sentences).strip()
        
        if len(answer) < 30:
            return self._out_of_scope_response()
        
        return {
            "answer": answer,
            "recommendation": best_sentences[0] if best_sentences else answer,
            "evidence": full_text[:400] + "...",
            "citation": "Diabetes Guidelines",
            "is_out_of_scope": False
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a clinical expert answering questions about diabetes screening and management.

Use ONLY the provided context to answer the question.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Provide a clear, complete, and direct answer
2. Include specific details like age ranges, BMI thresholds, and test names
3. If the context mentions a specific organization (USPSTF, ADA, WHO), mention it
4. If the context does NOT contain relevant information, say "I don't have enough information"

ANSWER:"""
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        return {
            "answer": response,
            "recommendation": response,
            "evidence": "",
            "citation": "",
            "is_out_of_scope": False
        }
    
    def _out_of_scope_response(self) -> Dict[str, str]:
        return {
            "answer": "I don't have enough information to answer this question. This system only provides answers about diabetes management, screening, and care using guidelines from WHO, USPSTF, and other official sources. Please ask a clear question related to diabetes (e.g., screening, diagnosis, treatment, complications).",
            "recommendation": "Question is out of scope.",
            "evidence": "",
            "citation": "",
            "is_out_of_scope": True
        }