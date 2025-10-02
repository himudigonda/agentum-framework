# agentum/memory.py
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


class BaseMemory(BaseModel):
    def load_messages(self) -> List[BaseMessage]:
        raise NotImplementedError

    def save_messages(self, messages: List[BaseMessage]):
        raise NotImplementedError


class ConversationMemory(BaseMemory):
    """Stores conversation history in-memory for the duration of a workflow run."""

    history: List[BaseMessage] = []

    def load_messages(self) -> List[BaseMessage]:
        return self.history

    def save_messages(self, messages: List[BaseMessage]):
        self.history.extend(messages[-2:])


class SummaryMemory(BaseMemory):
    """Stores a summarized version of the conversation history."""

    history: List[BaseMessage] = Field(default_factory=list)
    summary: str = ""
    llm: Any  # MODIFICATION: LLM is REQUIRED to be passed in, removing race condition

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

        # MODIFICATION: Remove conditional LLM instantiation, assume it is ready.
        new_lines = get_buffer_string(messages)

        chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.SUMMARIZER_PROMPT),
        )

        self.summary = chain.invoke(
            {"summary": self.summary, "new_lines": new_lines}
        ).get("text", "")


class VectorStoreMemory(BaseMemory):
    """
    Stores and retrieves conversation history using a Vector Store (Chroma).
    This enables semantic search over past conversations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Chroma = Field(
        default_factory=lambda: Chroma(
            collection_name="vector_memory",
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        )
    )
    current_buffer: List[BaseMessage] = Field(default_factory=list)

    def load_messages(self) -> List[BaseMessage]:
        """
        Retrieves relevant past messages based on the latest user input.
        We only load messages if the current buffer has a new HumanMessage.
        """
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
        """
        Converts all messages from the current turn into a single document and saves it.
        """
        text_to_save = get_buffer_string(messages)

        self.vector_store.add_texts(
            [text_to_save],
            metadatas=[{"turn": len(self.vector_store.get()["ids"]) + 1}],
        )

        self.current_buffer = []

    def append_message_for_search(self, message: BaseMessage):
        """Used by the agent to temporarily store the latest message for the semantic search trigger."""
        self.current_buffer.append(message)
