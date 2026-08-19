#!/usr/bin/env python
"""
Run Evaluation Script
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import config
from core.ingest import load_index
from core.retrieval import DiabetesRetriever
from core.generation import LocalLLM
from core.evaluation import run_evaluation


def main():
    print("=" * 60)
    print("📊 DIABETES CHATBOT EVALUATION")
    print("=" * 60)
    
    try:
        print("\n📂 Loading index...")
        vectordb = load_index()
        retriever = DiabetesRetriever(vectordb)
        
        print("🧠 Loading LLM...")
        llm = LocalLLM()
        
        print("\n🔄 Running evaluation...")
        results = run_evaluation(retriever, llm)
        
        print("\n✅ Evaluation complete!")
        
        if results and "metrics" in results:
            metrics = results["metrics"]
            print(f"\n📈 Pass Rate: {metrics['pass_rate']*100:.1f}%")
            print(f"📈 Refusal Rate: {metrics['out_of_scope_refusal_rate']*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()