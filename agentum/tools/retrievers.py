from typing import Any
from agentum import tool

def create_vector_search_tool(kb: Any, name: str=None, **tool_kwargs):
    tool_name = name or f"vector_search_{kb.name.lower().replace(' ', '_')}"

    @tool(name=tool_name, **tool_kwargs)
    def vector_search(query: str, top_k: int=4) -> str:
        retriever = kb.as_retriever(search_kwargs={'k': top_k})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return 'No relevant documents found.'
        result = '\n\n'.join([doc.page_content for doc in docs])
        return result
    return vector_search

def create_vector_search_with_score_tool(kb: Any, **tool_kwargs):

    @tool(**tool_kwargs)
    def vector_search_with_score(query: str, top_k: int=4) -> str:
        retriever = kb.as_retriever(search_kwargs={'k': top_k})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return 'No relevant documents found.'
        result_parts = []
        for i, doc in enumerate(docs, 1):
            score = getattr(doc, 'metadata', {}).get('score', 'N/A')
            result_parts.append(f'Document {i} (Score: {score}):\n{doc.page_content}')
        return '\n\n'.join(result_parts)
    return vector_search_with_score