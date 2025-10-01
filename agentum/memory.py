# agentum/memory.py
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel


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
        # Just save the last human message and the AI response
        self.history.extend(messages[-2:])
