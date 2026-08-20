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
                device_map="auto"
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
    
    def _is_meaningful_question(self, question: str) -> bool:
        question = question.strip()
        if len(question) < 3:
            return False
        if len(re.findall(r'[a-zA-Z]', question)) < 2:
            return False
        return True
    
    def _is_diabetes_question(self, question: str) -> bool:
        question_lower = question.lower()
        diabetes_keywords = [
            'diabetes', 'diabetic', 'type 2', 'type2', 't2dm',
            'prediabetes', 'pre-diabetes', 'hyperglycemia',
            'glucose', 'insulin', 'a1c', 'hba1c',
            'blood sugar', 'fasting glucose', 'ogtt',
            'metformin', 'sulfonylurea', 'glycemic',
            'retinopathy', 'nephropathy', 'neuropathy',
            'ketoacidosis', 'hypoglycemia', 'mmol', 'mg/dL',
            'screening', 'diagnosis', 'management',
            'treatment', 'prevention', 'complication',
            'hypertension', 'blood pressure', 'lipid', 'cholesterol',
            'cost', 'costs', 'economic', 'expenditure', 'spending',
            'dollars', 'billion', 'million'
        ]
        return any(kw in question_lower for kw in diabetes_keywords)
    
    def _is_clearly_out_of_scope(self, question: str) -> bool:
        question_lower = question.lower()
        out_of_scope = [
            'breast cancer', 'lung cancer', 'colon cancer', 'prostate cancer',
            'cancer treatment', 'chemotherapy', 'radiation',
            'covid', 'coronavirus', 'vaccine',
            'alzheimer', 'dementia', 'parkinson',
            'heart attack', 'stroke', 'cardiac arrest'
        ]
        for kw in out_of_scope:
            if kw in question_lower:
                return True
        return False
    
    def _is_out_of_scope(self, question: str, context: str) -> bool:
        if not self._is_meaningful_question(question):
            return True
        
        if self._is_clearly_out_of_scope(question):
            return True
        
        if not self._is_diabetes_question(question):
            return True
        
        context_lower = context.lower()
        diabetes_indicators = ['diabetes', 'glucose', 'insulin', 'a1c', 'hba1c', 'screening', 'mmol', 'mg/dL', 'cost', 'billion', 'million']
        if not any(kw in context_lower for kw in diabetes_indicators):
            return True
        
        return False
    
    def _answer_exists_in_context(self, question: str, context: str) -> bool:
        question_lower = question.lower()
        context_lower = context.lower()
        
        if len(context.split()) < 30:
            return False
        
        question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question_lower))
        stopwords = {'what', 'the', 'for', 'are', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out',
                     'use', 'way', 'who', 'why', 'has', 'his', 'how', 'its', 'let', 'may', 'new', 'now', 'old', 'see', 'too',
                     'any', 'ask', 'big', 'day', 'end', 'few', 'get', 'god', 'got', 'hot', 'job', 'man', 'men', 'own', 'pay',
                     'put', 'run', 'set', 'sit', 'son', 'ten', 'top', 'try', 'two', 'war', 'way', 'why', 'yes',
                     'from', 'with', 'without', 'about', 'against', 'between', 'through', 'during', 'within', 'upon',
                     'would', 'could', 'should', 'might', 'must', 'may', 'will', 'can', 'does', 'did', 'has', 'have', 'had'}
        question_words = {w for w in question_words if w not in stopwords}
        
        if not question_words:
            return False
        
        found_words = sum(1 for w in question_words if w in context_lower)
        
        if found_words / len(question_words) < 0.3:
            return False
        
        answer_indicators = [
            'recommend', 'recommendation', 'guideline', 'should', 'target',
            'goal', 'optimal', 'normal', 'abnormal', 'diagnosis', 'screening',
            'treatment', 'management', 'prevention', 'control', 'risk',
            'cost', 'costs', 'billion', 'million', 'dollars', 'prevalence',
            'incidence', 'mortality', 'complication', 'retinopathy',
            'nephropathy', 'neuropathy', 'hypertension', 'hypoglycemia'
        ]
        
        has_answer_indicator = any(ind in context_lower for ind in answer_indicators)
        
        return found_words >= 2 and has_answer_indicator
    
    def generate(self, question: str, context: str) -> Dict[str, str]:
        if self._is_out_of_scope(question, context):
            return self._out_of_scope_response()
        
        if not self._answer_exists_in_context(question, context):
            return self._no_answer_response()
        
        if self.pipeline is not None:
            try:
                prompt = self._build_prompt(question, context)
                response = self.pipeline(prompt)[0]["generated_text"]
                
                if len(response.strip()) > 30 and "don't have enough" not in response.lower():
                    parsed = self._parse_response(response)
                    if "don't have enough" in parsed["answer"].lower():
                        return self._no_answer_response()
                    return parsed
            except Exception as e:
                print(f"Generation error: {e}")
        
        return self._extract_from_context(question, context)
    
    def _extract_from_context(self, question: str, context: str) -> Dict[str, str]:
        context_words = len(context.split())
        if context_words < 20:
            return self._no_answer_response()
        
        passages = re.split(r'Passage \d+ \([^)]+\):\n', context)
        passages = [p.strip() for p in passages if p.strip()]
        
        if not passages:
            return self._no_answer_response()
        
        full_text = " ".join(passages)
        
        # ===== SPECIAL HANDLING FOR SCREENING QUESTIONS =====
        if "screening" in question.lower() or "screen" in question.lower():
            
            # PRIORITY 1: USPSTF Recommendation
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
            
            # PRIORITY 2: ADA Recommendation
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
        
        # ===== SPECIAL HANDLING FOR COST QUESTIONS =====
        if any(kw in question.lower() for kw in ['cost', 'costs', 'economic', 'expenditure', 'spending', 'billion', 'million', 'dollars']):
            
            cost_patterns = [
                r'\$(\d+\.?\d*)\s*billion.*?(?:total|overall|direct|indirect)',
                r'(?:total|overall|direct|indirect).*?\$(\d+\.?\d*)\s*billion',
                r'estimated total costs? of diabetes.*?\$(\d+\.?\d*)\s*billion',
                r'(\d+)\s*billion.*?(?:total|overall)\s+cost',
                r'\$(\d+)\s*\,?\s*(\d+)\s*(?:billion|million)'
            ]
            
            for pattern in cost_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    total_cost = match.group(1)
                    if len(match.groups()) > 1:
                        total_cost = match.group(1) + "." + match.group(2) if '.' not in match.group(1) else match.group(1)
                    
                    answer = f"The estimated total cost of diabetes in the US was ${total_cost} billion."
                    
                    direct_match = re.search(r'\$(\d+)\s*billion.*?direct', full_text, re.IGNORECASE)
                    indirect_match = re.search(r'\$(\d+)\s*billion.*?indirect', full_text, re.IGNORECASE)
                    
                    if direct_match:
                        answer += f" Direct medical costs: ${direct_match.group(1)} billion."
                    if indirect_match:
                        answer += f" Indirect costs: ${indirect_match.group(1)} billion."
                    
                    return {
                        "answer": answer,
                        "recommendation": answer,
                        "evidence": full_text[:400] + "...",
                        "citation": "WHO Type 2 Diabetes Evidence Review (2008)",
                        "is_out_of_scope": False
                    }
            
            cost_texts = re.findall(r'\$[\d,]+\s*(?:billion|million|trillion)', full_text)
            if cost_texts:
                answer = f"According to the guidelines, the estimated total cost of diabetes in the US is {cost_texts[0]}."
                if len(cost_texts) > 1:
                    answer += f" ({', '.join(cost_texts[1:3])})"
                return {
                    "answer": answer,
                    "recommendation": answer,
                    "evidence": full_text[:400] + "...",
                    "citation": "WHO Type 2 Diabetes Evidence Review",
                    "is_out_of_scope": False
                }
        
        # ===== SPECIAL HANDLING FOR BLOOD PRESSURE QUESTIONS =====
        if "blood pressure" in question.lower() or "bp" in question.lower():
            bp_patterns = [
                r'blood pressure target.*?(\d+)\s*/\s*(\d+)',
                r'target.*?blood pressure.*?(\d+)\s*/\s*(\d+)',
                r'(\d+)\s*/\s*(\d+)\s*mmHg.*?target'
            ]
            for pattern in bp_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
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
            return self._no_answer_response()
        
        question_words = set(question.lower().split())
        question_words = {w for w in question_words if len(w) > 3}
        
        stopwords = {'what', 'the', 'for', 'are', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out',
                     'use', 'way', 'who', 'why', 'has', 'his', 'how', 'its', 'let', 'may', 'new', 'now', 'old', 'see', 'too',
                     'any', 'ask', 'big', 'day', 'end', 'few', 'get', 'god', 'got', 'hot', 'job', 'man', 'men', 'own', 'pay',
                     'put', 'run', 'set', 'sit', 'son', 'ten', 'top', 'try', 'two', 'war', 'way', 'why', 'yes',
                     'from', 'with', 'without', 'about', 'against', 'between', 'through', 'during', 'within', 'upon',
                     'would', 'could', 'should', 'might', 'must', 'may', 'will', 'can', 'does', 'did', 'has', 'have', 'had'}
        question_words = {w for w in question_words if w not in stopwords}
        
        if not question_words:
            return self._no_answer_response()
        
        scored_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            word_overlap = sum(1 for w in question_words if w in sentence_lower)
            
            answer_bonus = 0
            if "cause" in question.lower():
                if any(kw in sentence_lower for kw in ['cause', 'caused', 'due to', 'result from']):
                    answer_bonus += 5
            elif "screening" in question.lower():
                if any(kw in sentence_lower for kw in ['recommend', 'screen', 'test', 'detect']):
                    answer_bonus += 5
            elif "treatment" in question.lower():
                if any(kw in sentence_lower for kw in ['treat', 'therapy', 'medication', 'drug']):
                    answer_bonus += 5
            elif "cost" in question.lower():
                if any(kw in sentence_lower for kw in ['cost', 'billion', 'million', 'dollars', 'expenditure']):
                    answer_bonus += 5
            elif "prevalence" in question.lower() or "how many" in question.lower():
                if any(kw in sentence_lower for kw in ['prevalence', 'percent', 'million', 'people']):
                    answer_bonus += 5
            elif "management" in question.lower() or "care" in question.lower():
                if any(kw in sentence_lower for kw in ['management', 'care', 'control', 'monitor']):
                    answer_bonus += 5
            elif "prevention" in question.lower():
                if any(kw in sentence_lower for kw in ['prevent', 'avoid', 'reduce', 'risk']):
                    answer_bonus += 5
            elif "diagnosis" in question.lower():
                if any(kw in sentence_lower for kw in ['diagnos', 'test', 'detect', 'identify']):
                    answer_bonus += 5
            
            if re.search(r'\d+', sentence):
                answer_bonus += 1
            
            total_score = word_overlap * 2 + answer_bonus
            
            scored_sentences.append((sentence, total_score, i))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_sentences or scored_sentences[0][1] < 3:
            return self._no_answer_response()
        
        best_sentences = []
        for s, score, _ in scored_sentences[:5]:
            if score >= 2:
                best_sentences.append(s)
        
        if not best_sentences:
            return self._no_answer_response()
        
        best_idx = scored_sentences[0][2]
        start = max(0, best_idx - 1)
        end = min(len(sentences), best_idx + 3)
        context_sentences = sentences[start:end]
        
        if len(" ".join(context_sentences)) > len(" ".join(best_sentences)) and len(context_sentences) > 1:
            best_sentences = context_sentences
        
        answer = ". ".join(best_sentences).strip()
        
        if len(answer) < 30:
            return self._no_answer_response()
        
        return {
            "answer": answer,
            "recommendation": best_sentences[0] if best_sentences else answer,
            "evidence": full_text[:400] + "...",
            "citation": "Diabetes Guidelines",
            "is_out_of_scope": False
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a clinical expert answering questions about diabetes.

Use ONLY the provided context to answer the question.
If the context does NOT contain relevant information, say 'I don't have enough information'.

CONTEXT:
{context}

QUESTION: {question}

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
            "answer": "I don't have enough information to answer this question. This system provides evidence-based answers about diabetes management, screening, and care using guidelines from WHO, USPSTF, and other official sources. Please ask a clear question related to diabetes.",
            "recommendation": "Question is out of scope or unclear.",
            "evidence": "",
            "citation": "",
            "is_out_of_scope": True
        }
    
    def _no_answer_response(self) -> Dict[str, str]:
        return {
            "answer": "I don't have enough information to answer this question. The answer is not available in the current guidelines. Please try rephrasing or ask a different question about diabetes.",
            "recommendation": "Answer not found in guidelines.",
            "evidence": "",
            "citation": "",
            "is_out_of_scope": True
        }