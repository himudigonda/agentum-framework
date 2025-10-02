from typing import Any, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rich.console import Console

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

console = Console()


class KnowledgeBase:

    def __init__(
        self,
        name: str,
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_reranking: bool = True,
    ):
        self.name = name
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
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
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
                f"[yellow]Warning: Reranking requested but sentence-transformers not installed[/yellow]"
            )

        console.print(f"📚 KnowledgeBase '{self.name}' initialized.", style="bold blue")

    def add(self, sources: List[str]):
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

            console.print(f"  - Loading from source: {source}")
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        self.vector_store.add_documents(documents=splits)
        console.print(
            f"  ✅ Added {len(splits)} document chunks to the '{self.name}' knowledge base."
        )

    def as_retriever(self, **kwargs: Any):
        retriever = self.vector_store.as_retriever(**kwargs)

        if self.reranker is not None:
            return self._rerank_retriever(retriever)

        return retriever

    def _rerank_retriever(self, retriever):

        def rerank_get_relevant_documents(query: str) -> List[Any]:
            docs = retriever.get_relevant_documents(query)

            if not docs:
                return docs

            try:
                pairs = [(query, doc.page_content) for doc in docs]
                scores = self.reranker.predict(pairs)

                doc_scores = list(zip(docs, scores))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                reranked_docs = [doc for doc, score in doc_scores]

                console.print(
                    f"🔄 Reranked {len(docs)} documents for query: '{query[:50]}...'",
                    style="dim",
                )
                return reranked_docs
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Reranking failed, returning original results: {e}[/yellow]"
                )
                return docs

        class RerankedRetriever:
            def __init__(self, original_retriever, rerank_func):
                self.original_retriever = original_retriever
                self.rerank_func = rerank_func
                for attr in dir(original_retriever):
                    if not attr.startswith("_") and not callable(
                        getattr(original_retriever, attr)
                    ):
                        setattr(self, attr, getattr(original_retriever, attr))

            def get_relevant_documents(self, query: str) -> List[Any]:
                return self.rerank_func(query)

            def aget_relevant_documents(self, query: str) -> List[Any]:
                return self.rerank_func(query)

        return RerankedRetriever(retriever, rerank_get_relevant_documents)
        return RerankedRetriever(retriever, rerank_get_relevant_documents)
