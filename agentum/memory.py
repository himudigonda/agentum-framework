from datetime import datetime
from functools import lru_cache
from typing import Any, ClassVar, List, Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    get_buffer_string,
)
from pydantic import BaseModel, ConfigDict, Field


@lru_cache(maxsize=1)
def get_embedding_function():
    """Creates and caches the single, shared instance of HuggingFaceEmbeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class BaseMemory(BaseModel):
    def load_messages(self) -> List[BaseMessage]:
        raise NotImplementedError

    def save_messages(self, messages: List[BaseMessage]):
        raise NotImplementedError


class ConversationMemory(BaseMemory):

    history: List[BaseMessage] = []

    def load_messages(self) -> List[BaseMessage]:
        return self.history

    def save_messages(self, messages: List[BaseMessage]):
        self.history.extend(messages[-2:])


class SummaryMemory(BaseMemory):

    history: List[BaseMessage] = Field(default_factory=list)
    summary: str = ""
    llm: Any

    SUMMARIZER_PROMPT: ClassVar[
        str
    ] = """
    Progressively summarize the following chat history, focusing on key themes,
    entities, and the latest turns.
    
    Current Summary:
    {summary}
    
    New Chat History:
    {new_lines}
    
    New Summary:"""

    def load_messages(self) -> List[BaseMessage]:
        if self.summary:
            return [
                AIMessage(
                    content=f"Summary of prior conversation: {self.summary}",
                    role="system",
                )
            ]
        return []

    def save_messages(self, messages: List[BaseMessage]):
        self.history.extend(messages)

        new_lines = get_buffer_string(messages)

        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.SUMMARIZER_PROMPT),
        )

        self.summary = chain.invoke(
            {"summary": self.summary, "new_lines": new_lines}
        ).get("text", "")


class VectorStoreMemory(BaseMemory):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Chroma = Field(
        default_factory=lambda: Chroma(
            collection_name="vector_memory",
            embedding_function=get_embedding_function(),
        )
    )
    current_buffer: List[BaseMessage] = Field(default_factory=list)

    def load_messages(self) -> List[BaseMessage]:
        if not self.current_buffer:
            return []

        latest_message = self.current_buffer[-1].content

        relevant_docs = self.vector_store.similarity_search(str(latest_message), k=3)

        context_message = "\n".join(
            [f"Past Context: {doc.page_content}" for doc in relevant_docs]
        )

        return [
            AIMessage(
                content=f"Relevant historical context:\n{context_message}",
                role="system",
            )
        ]

    def save_messages(self, messages: List[BaseMessage]):
        text_to_save = get_buffer_string(messages)

        self.vector_store.add_texts(
            [text_to_save],
            metadatas=[{"timestamp": str(datetime.now())}],
        )

        self.current_buffer = []

    def append_message_for_search(self, message: BaseMessage):
        self.current_buffer.append(message)
