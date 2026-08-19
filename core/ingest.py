import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict
import re

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
            
            all_pages.append({
                "page_number": page_num + 1,
                "text": text,
                "document_name": pdf_path.stem,
                "word_count": len(text.split())
            })
        
        doc.close()
        print(f"   -> {total_pages} pages loaded")
    
    return all_pages


def clean_text(text: str) -> str:
    text = re.sub(r'===== Page \d+ =====', '', text)
    text = re.sub(r'[=─—–]{5,}', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_documents(pages: List[Dict]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    all_chunks = []
    
    for page in pages:
        doc = Document(
            page_content=page["text"],
            metadata={
                "page_number": page["page_number"],
                "document_name": page["document_name"],
                "source": "pdf"
            }
        )
        
        chunks = splitter.split_documents([doc])
        
        for chunk in chunks:
            chunk.page_content = clean_chunk(chunk.page_content)
            chunk.metadata["chunk_id"] = (
                f"{page['document_name']}-p{page['page_number']}-c{len(all_chunks)}"
            )
        
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
    
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name="diabetes_guidelines",
        persist_directory=str(config.CHROMA_DIR)
    )
    
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