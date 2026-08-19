"""
Evaluation Module - Test and Metrics
"""
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import config
from core.retrieval import DiabetesRetriever
from core.generation import LocalLLM

logger = logging.getLogger(__name__)


class DiabetesEvaluator:
    """Evaluate the RAG system performance"""
    
    def __init__(self, retriever: DiabetesRetriever = None, llm: LocalLLM = None):
        self.retriever = retriever
        self.llm = llm
        self.test_set = []
        self.results = []
        self._load_test_set()
    
    def _load_test_set(self):
        """Load test set from CSV"""
        test_file = config.EVAL_DIR / "test_set.csv"
        
        if not test_file.exists():
            print(f"⚠️ Test set not found at {test_file}")
            print("   Creating default test set...")
            self._create_default_test_set()
            return self._load_test_set()
        
        with open(test_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.test_set = [row for row in reader]
        
        print(f"📊 Loaded {len(self.test_set)} test questions")
        return self.test_set
    
    def _create_default_test_set(self):
        """Create default test set if not exists"""
        test_file = config.EVAL_DIR / "test_set.csv"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_data = [
            {
                "question": "What is the recommended screening for diabetes?",
                "expected_answer": "Screening for prediabetes and type 2 diabetes in adults aged 35 to 70 years who have overweight or obesity",
                "expected_source": "uspstf_recommendation_statement_2021",
                "expected_page": "1"
            },
            {
                "question": "What is the target blood pressure for diabetes?",
                "expected_answer": "Blood pressure target should be <=135/85 mmHg",
                "expected_source": "uspstf_recommendation_statement_2021",
                "expected_page": "4"
            },
            {
                "question": "What are the first-line medications for diabetes?",
                "expected_answer": "Metformin is the first-line medication for type 2 diabetes",
                "expected_source": "type2es",
                "expected_page": "5"
            },
            {
                "question": "What is the recommended HbA1c target?",
                "expected_answer": "HbA1c target is less than 7% for most adults with type 2 diabetes",
                "expected_source": "type2es",
                "expected_page": "4"
            },
            {
                "question": "What is the treatment for breast cancer?",
                "expected_answer": "Out of Scope - System should refuse",
                "expected_source": "NOT_COVERED",
                "expected_page": "0"
            }
        ]
        
        with open(test_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["question", "expected_answer", "expected_source", "expected_page"])
            writer.writeheader()
            writer.writerows(test_data)
    
    def run_evaluation(self, save_results: bool = True) -> Dict:
        """Run full evaluation"""
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        
        print("\n" + "=" * 60)
        print("📊 RUNNING EVALUATION")
        print("=" * 60)
        
        results = []
        
        for i, test in enumerate(self.test_set, 1):
            question = test.get("question", "")
            expected_answer = test.get("expected_answer", "")
            expected_source = test.get("expected_source", "")
            expected_page = test.get("expected_page", "0")
            
            print(f"\n[{i}/{len(self.test_set)}] Testing: {question[:50]}...")
            
            # Run query
            result = self._evaluate_single(question, expected_answer, expected_source, expected_page)
            results.append(result)
            
            # Print result
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"   Status: {status}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Source match: {result['source_match']}")
            print(f"   Page match: {result['page_match']}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        # Save results
        if save_results:
            self._save_results(results, metrics)
        
        # Print summary
        self._print_summary(results, metrics)
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    def _evaluate_single(self, question: str, expected_answer: str, expected_source: str, expected_page: str) -> Dict:
        """Evaluate a single question"""
        
        # 1. Retrieve
        retrieved = self.retriever.retrieve(question, k=config.TOP_K)
        
        # 2. Check confidence
        is_confident, max_score = self.retriever.check_confidence(retrieved)
        
        # 3. Generate (if LLM available)
        generated_answer = ""
        if self.llm:
            context = self.retriever.prepare_context(retrieved)
            response = self.llm.generate(question, context)
            generated_answer = response.get("answer", "")
        
        # 4. Check if out of scope
        is_out_of_scope = expected_source == "NOT_COVERED"
        
        # 5. Check source match
        source_match = False
        page_match = False
        found_in_sources = False
        
        for r in retrieved:
            doc_name = r["metadata"].get("document_name", "")
            page = str(r["metadata"].get("page_number", "0"))
            
            if expected_source in doc_name:
                source_match = True
                if expected_page == page:
                    page_match = True
            
            # Check if answer is in retrieved content
            if expected_answer.lower() in r["content"].lower():
                found_in_sources = True
        
        # 6. Determine pass/fail
        if is_out_of_scope:
            # For out of scope questions, should refuse
            passed = is_confident == False or "don't have enough" in generated_answer.lower()
        else:
            # For in-scope questions
            passed = source_match and page_match and found_in_sources
        
        return {
            "question": question,
            "expected_answer": expected_answer,
            "expected_source": expected_source,
            "expected_page": expected_page,
            "is_out_of_scope": is_out_of_scope,
            "retrieved_count": len(retrieved),
            "max_score": max_score,
            "is_confident": is_confident,
            "source_match": source_match,
            "page_match": page_match,
            "found_in_sources": found_in_sources,
            "generated_answer": generated_answer,
            "passed": passed,
            "score": max_score
        }
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics"""
        
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        source_matches = sum(1 for r in results if r["source_match"])
        page_matches = sum(1 for r in results if r["page_match"])
        found_in_sources = sum(1 for r in results if r["found_in_sources"])
        
        # Separate in-scope and out-of-scope
        in_scope = [r for r in results if not r["is_out_of_scope"]]
        out_of_scope = [r for r in results if r["is_out_of_scope"]]
        
        in_scope_passed = sum(1 for r in in_scope if r["passed"])
        out_of_scope_refused = sum(1 for r in out_of_scope if r["passed"])
        
        # Average scores
        avg_score = sum(r["score"] for r in results) / total if total > 0 else 0
        
        return {
            "total_questions": total,
            "passed": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "source_match_rate": source_matches / total if total > 0 else 0,
            "page_match_rate": page_matches / total if total > 0 else 0,
            "found_in_sources_rate": found_in_sources / total if total > 0 else 0,
            "avg_confidence_score": avg_score,
            "in_scope": len(in_scope),
            "in_scope_passed": in_scope_passed,
            "in_scope_pass_rate": in_scope_passed / len(in_scope) if in_scope else 0,
            "out_of_scope": len(out_of_scope),
            "out_of_scope_refused": out_of_scope_refused,
            "out_of_scope_refusal_rate": out_of_scope_refused / len(out_of_scope) if out_of_scope else 0,
        }
    
    def _print_summary(self, results: List[Dict], metrics: Dict):
        """Print evaluation summary"""
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\n📝 Total Questions: {metrics['total_questions']}")
        print(f"✅ Passed: {metrics['passed']}/{metrics['total_questions']}")
        print(f"📈 Pass Rate: {metrics['pass_rate']*100:.1f}%")
        
        print(f"\n📄 Source Match Rate: {metrics['source_match_rate']*100:.1f}%")
        print(f"📄 Page Match Rate: {metrics['page_match_rate']*100:.1f}%")
        print(f"📄 Found in Sources: {metrics['found_in_sources_rate']*100:.1f}%")
        
        print(f"\n📊 Average Confidence Score: {metrics['avg_confidence_score']:.3f}")
        
        print(f"\n🎯 In-Scope Questions: {metrics['in_scope']}")
        print(f"   Passed: {metrics['in_scope_passed']}/{metrics['in_scope']}")
        print(f"   Pass Rate: {metrics['in_scope_pass_rate']*100:.1f}%")
        
        print(f"\n🚫 Out-of-Scope Questions: {metrics['out_of_scope']}")
        print(f"   Refused: {metrics['out_of_scope_refused']}/{metrics['out_of_scope']}")
        print(f"   Refusal Rate: {metrics['out_of_scope_refusal_rate']*100:.1f}%")
        
        print("\n" + "=" * 60)
        
        # Show failed questions
        failed = [r for r in results if not r["passed"]]
        if failed:
            print("\n❌ Failed Questions:")
            for r in failed:
                print(f"   - {r['question'][:50]}...")
                print(f"     Expected source: {r['expected_source']}, Page: {r['expected_page']}")
                print(f"     Source match: {r['source_match']}, Page match: {r['page_match']}")
    
    def _save_results(self, results: List[Dict], metrics: Dict):
        """Save results to JSON"""
        results_dir = config.EVAL_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = results_dir / "evaluation_results.json"
        
        data = {
            "timestamp": str(Path(".").stat().st_ctime),
            "metrics": metrics,
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def get_summary(self) -> Dict:
        """Get quick summary without running full evaluation"""
        if not self.results:
            return {"status": "No results available. Run evaluation first."}
        
        return self._calculate_metrics(self.results)


def run_evaluation(retriever: DiabetesRetriever = None, llm: LocalLLM = None) -> Dict:
    """Helper function to run evaluation"""
    evaluator = DiabetesEvaluator(retriever, llm)
    return evaluator.run_evaluation()


def load_evaluation_results() -> Optional[Dict]:
    """Load saved evaluation results"""
    results_file = config.EVAL_DIR / "results" / "evaluation_results.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)