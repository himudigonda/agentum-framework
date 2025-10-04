from functools import lru_cache
from typing import Any, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console

from .loaders import load_documents_from_sources, split_documents
from .retrievers import RerankedRetriever

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
console = Console()


@lru_cache(maxsize=1)
def get_embedding_function(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=1)
def get_reranker_model(model_name: str):
    return CrossEncoder(model_name)


class KnowledgeBase:

    def __init__(
        self,
        name: str,
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.name = name
        self.embedding_function = get_embedding_function(embedding_model)
        self.vector_store = Chroma(
            collection_name=name,
            embedding_function=self.embedding_function,
            persist_directory=persist_directory,
        )
        if persist_directory is None:
            console.print(
                f"[yellow]Warning: KnowledgeBase '{self.name}' is in-memory/ephemeral. Use persist_directory for production.[/yellow]",
                style="bold yellow",
            )
        self.reranker = None
        if enable_reranking and CrossEncoder is not None:
            try:
                self.reranker = get_reranker_model(reranker_model)
                console.print(
                    f"🔄 Reranking enabled for KnowledgeBase '{self.name}'",
                    style="bold green",
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not initialize reranker: {e}[/yellow]"
                )
                self.reranker = None
        elif enable_reranking and CrossEncoder is None:
            console.print(
                "[yellow]Warning: Reranking requested but sentence-transformers not installed[/yellow]"
            )
        console.print(f"📚 KnowledgeBase '{self.name}' initialized.", style="bold blue")

    def add(self, sources: List[str]):
        docs = load_documents_from_sources(sources)
        splits = split_documents(docs)
        self.vector_store.add_documents(documents=splits)
        console.print(
            f"  ✅ Added {len(splits)} document chunks to the '{self.name}' knowledge base."
        )

    def as_retriever(self, **kwargs: Any):
        retriever = self.vector_store.as_retriever(**kwargs)
        if self.reranker is not None:
            return RerankedRetriever(retriever, self.reranker)
        return retriever
