import sys
import subprocess
import time
from pathlib import Path


def print_banner():
    print("""
    ================================================================
    DIABETES CLINICAL CHATBOT
    Local | Lightweight | 100% Free
    Embeddings: all-MiniLM-L6-v2 (80MB)
    LLM: flan-t5-small (300MB)
    ================================================================
    """)


def check_pdfs():
    data_dir = Path("data/raw_pdfs")
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("\nNo PDF files in data/raw_pdfs/")
        return False
    print(f"\nFound {len(pdf_files)} PDF(s)")
    return True


def run_ingestion():
    print("\nRunning ingestion...")
    try:
        from core.ingest import main as ingest_main
        ingest_main()
        return True
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_api():
    print("\nStarting API...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])


def run_ui():
    print("\nStarting UI...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit",
        "run", "app/ui.py",
        "--server.port", "8501"
    ])


def main():
    print_banner()
    
    if not check_pdfs():
        return
    
    if not run_ingestion():
        return
    
    api = run_api()
    time.sleep(3)
    ui = run_ui()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY!")
    print("=" * 60)
    print("\nAccess:")
    print("   UI:  http://localhost:8501")
    print("   API: http://localhost:8000")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    
    try:
        api.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api.terminate()
        ui.terminate()


if __name__ == "__main__":
    main()