# agentum/tools/retrievers.py
from typing import Any

from agentum import tool


def create_vector_search_tool(kb: Any):
    """
    Creates a vector search tool bound to a specific knowledge base.
    This is a factory function that returns a tool configured for a specific KB.
    """

    @tool
    def vector_search(query: str, top_k: int = 4) -> str:
        """
        Performs a simple semantic vector search on a Knowledge Base.
        Best for general queries and finding broadly relevant information.
        """
        try:
            retriever = kb.as_retriever(search_kwargs={"k": top_k})
            results = retriever.invoke(query)
            # Format the results with source metadata for better context
            formatted_results = [
                f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
                for doc in results
            ]
            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            return f"Error performing vector search: {str(e)}"

    return vector_search
