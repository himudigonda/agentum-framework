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
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


class BaseMemory(BaseModel):

    def load_messages(self, latest_input: BaseMessage) -> List[BaseMessage]:
        raise NotImplementedError

    def save_messages(self, messages: List[BaseMessage]):
        raise NotImplementedError


class ConversationMemory(BaseMemory):
    history: List[BaseMessage] = []

    def load_messages(self, latest_input: BaseMessage) -> List[BaseMessage]:
        return self.history

    def save_messages(self, messages: List[BaseMessage]):
        self.history.extend(messages[-2:])


class SummaryMemory(BaseMemory):
    history: List[BaseMessage] = Field(default_factory=list)
    summary: str = ""
    llm: Any
    SUMMARIZER_PROMPT: ClassVar[str] = (
        "\n    Progressively summarize the following chat history, focusing on key themes,\n    entities, and the latest turns.\n    \n    Current Summary:\n    {summary}\n    \n    New Chat History:\n    {new_lines}\n    \n    New Summary:"
    )

    def load_messages(self, latest_input: BaseMessage) -> List[BaseMessage]:
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
            llm=self.llm, prompt=PromptTemplate.from_template(self.SUMMARIZER_PROMPT)
        )
        self.summary = chain.invoke(
            {"summary": self.summary, "new_lines": new_lines}
        ).get("text", "")


class VectorStoreMemory(BaseMemory):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector_store: Chroma = Field(
        default_factory=lambda: Chroma(
            collection_name="vector_memory", embedding_function=get_embedding_function()
        )
    )

    def load_messages(self, latest_input: BaseMessage) -> List[BaseMessage]:
        if not latest_input or not isinstance(latest_input.content, str):
            return []

        # Search using the content of the latest message
        relevant_docs = self.vector_store.similarity_search(latest_input.content, k=3)
        if not relevant_docs:
            return []

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
        # This method should save the history, not clear a buffer.
        text_to_save = get_buffer_string(messages)
        if text_to_save:  # Avoid adding empty strings
            self.vector_store.add_texts(
                [text_to_save], metadatas=[{"timestamp": str(datetime.now())}]
            )
