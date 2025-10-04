import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from rich.console import Console

console = Console()


def load_documents_from_sources(sources: List[str]) -> List[Document]:
    """Load documents from various sources (PDF, TXT, web URLs)."""
    docs = []
    for source in sources:
        loader = None
        if source.startswith("http"):
            loader = WebBaseLoader(source)
        elif source.endswith(".pdf"):
            loader = PyPDFLoader(source)
        elif source.endswith(".txt"):
            loader = TextLoader(source)
        else:
            console.print(
                f"[yellow]Warning: Skipping unsupported source type: {source}[/yellow]"
            )
            continue
        source_for_metadata = (
            os.path.basename(source) if not source.startswith("http") else source
        )
        console.print(f"  - Loading from source: {source_for_metadata}")
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = source_for_metadata
        docs.extend(loaded_docs)
    return docs


def split_documents(
    docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)
