import asyncio
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rich.console import Console

console = Console()


class RerankedRetriever(BaseRetriever):
    """A retriever that reranks documents using a cross-encoder model."""

    def __init__(self, original_retriever: BaseRetriever, reranker: Any):
        self.original_retriever = original_retriever
        self.reranker = reranker

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs = self.original_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child()
        )
        if not docs:
            return []

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

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return await asyncio.to_thread(
            self._get_relevant_documents, query, run_manager=run_manager
        )
