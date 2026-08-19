import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict
import re
import os

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

from sentence_transformers import SentenceTransformer

import config


def load_pdfs(data_dir: Path) -> List[Dict]:
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}/")
        return []
    
    all_pages = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        doc = fitz.open(pdf_path)
        
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            text = clean_text(text)
            
            sections = extract_sections(text)
            
            doc_type = "guideline"
            if "uspstf" in pdf_path.stem.lower():
                doc_type = "recommendation"
            elif "who" in pdf_path.stem.lower() or "emro" in pdf_path.stem.lower():
                doc_type = "management"
            
            all_pages.append({
                "page_number": page_num + 1,
                "text": text,
                "document_name": pdf_path.stem,
                "document_type": doc_type,
                "word_count": len(text.split()),
                "sections": sections,
                "has_tables": bool(page.find_tables())
            })
        
        doc.close()
        print(f"   -> {total_pages} pages loaded")
    
    return all_pages


def clean_text(text: str) -> str:
    text = re.sub(r'===== Page \d+ =====', '', text)
    text = re.sub(r'[=─—–]{5,}', '', text)
    text = re.sub(r'\s*\[\d+\]\s*', ' ', text)
    text = re.sub(r'\s+\d+\s+', ' ', text)
    
    medical_terms = {
        'HbA1c': 'HbA1c',
        'hba1c': 'HbA1c',
        'BMI': 'BMI',
        'bmi': 'BMI',
        'FPG': 'FPG',
        'fpg': 'FPG',
        'OGTT': 'OGTT',
        'ogtt': 'OGTT',
        'DM2': 'type 2 diabetes',
        'T2DM': 'type 2 diabetes',
        'IGT': 'impaired glucose tolerance',
        'IFG': 'impaired fasting glucose',
        'SMBG': 'self monitoring of blood glucose',
        'mmol/L': 'mmol per liter',
        'mg/dL': 'mg per deciliter'
    }
    
    for term, replacement in medical_terms.items():
        text = re.sub(rf'\b{term}\b', replacement, text, flags=re.IGNORECASE)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_sections(text: str) -> List[str]:
    sections = []
    patterns = [
        r'^(\d+)\.\s+([A-Z][A-Z\s]+)$',
        r'^Update Key Question (\d+)\.',
        r'^TABLE\s+(\d+)\.',
        r'^(Summary of Findings|Study Details|Conclusions|Discussion|Introduction)$',
        r'^(Methods|Results|Background|Purpose)$',
        r'^[A-Z][a-z]+ [A-Z][a-z]+:',
        r'^[A-Z][A-Z\s]+$'
    ]
    
    for line in text.split('\n'):
        line = line.strip()
        if len(line) > 150 or len(line) < 3:
            continue
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                sections.append(line[:100])
                break
    
    return sections


def chunk_documents(pages: List[Dict]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 3,
        chunk_overlap=config.CHUNK_OVERLAP * 3,
        separators=[
            "\n\n\n",
            "\n\n",
            "Recommendation",
            "The USPSTF",
            "The ADA",
            "WHO recommends",
            "Screening for",
            "Diagnosis of",
            "Treatment of",
            "Management of",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ],
        length_function=len
    )
    
    all_chunks = []
    
    for page in pages:
        sections = page.get("sections", [])
        
        doc = Document(
            page_content=page["text"],
            metadata={
                "page_number": page["page_number"],
                "document_name": page["document_name"],
                "document_type": page.get("document_type", "guideline"),
                "source": "pdf",
                "has_tables": page.get("has_tables", False),
                "word_count": page["word_count"],
                "sections": ", ".join(sections[:5]) if sections else "General",
                "section_count": len(sections)
            }
        )
        
        chunks = splitter.split_documents([doc])
        
        for chunk in chunks:
            chunk.page_content = clean_chunk(chunk.page_content)
            chunk.metadata["chunk_id"] = (
                f"{page['document_name']}-p{page['page_number']}-c{len(all_chunks)}"
            )
            chunk.metadata["chunk_length"] = len(chunk.page_content)
            
            if "sections" not in chunk.metadata or not chunk.metadata["sections"]:
                chunk.metadata["sections"] = "General"
            elif isinstance(chunk.metadata["sections"], list):
                chunk.metadata["sections"] = ", ".join(chunk.metadata["sections"]) if chunk.metadata["sections"] else "General"
        
        all_chunks.extend(chunks)
    
    all_chunks = [c for c in all_chunks if len(c.page_content.strip()) > config.MIN_CHUNK_SIZE]
    
    print(f"Created {len(all_chunks)} chunks")
    return all_chunks


def clean_chunk(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class LocalEmbedder:
    def __init__(self):
        print(f"Loading embedder: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
        self.dimension = self.model.get_embedding_dimension()
        print(f"   dim={self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(
            texts,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()


def build_index(chunks: List[Document], force_rebuild: bool = True) -> Chroma:
    if force_rebuild and config.CHROMA_DIR.exists():
        print(f"Removing existing index at {config.CHROMA_DIR}")
        shutil.rmtree(config.CHROMA_DIR)
    
    embedder = LocalEmbedder()
    
    print(f"Building index with {len(chunks)} chunks...")
    
    for chunk in chunks:
        for key, value in chunk.metadata.items():
            if isinstance(value, list):
                chunk.metadata[key] = ", ".join(str(v) for v in value) if value else ""
            elif value is None:
                chunk.metadata[key] = ""
    
    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            collection_name="diabetes_guidelines",
            persist_directory=str(config.CHROMA_DIR)
        )
    except Exception as e:
        print(f"❌ Failed to build index: {e}")
        print("   Trying with smaller batch...")
        
        # Try with smaller batch
        batch_size = 100
        vectordb = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            if i == 0:
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=embedder,
                    collection_name="diabetes_guidelines",
                    persist_directory=str(config.CHROMA_DIR)
                )
            else:
                vectordb.add_documents(batch)
            print(f"   Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    
    print(f"   Index saved to {config.CHROMA_DIR}/")
    return vectordb


def load_index() -> Chroma:
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(f"Index not found at {config.CHROMA_DIR}")
    
    embedder = LocalEmbedder()
    
    return Chroma(
        collection_name="diabetes_guidelines",
        embedding_function=embedder,
        persist_directory=str(config.CHROMA_DIR)
    )


def main():
    print("=" * 60)
    print("DIABETES INGESTION PIPELINE")
    print("=" * 60)
    
    pages = load_pdfs(config.DATA_DIR)
    if not pages:
        return
    
    chunks = chunk_documents(pages)
    
    vectordb = build_index(chunks, force_rebuild=True)
    
    print("\nIngestion complete!")
    print(f"Index: {config.CHROMA_DIR}/")


if __name__ == "__main__":
    main()