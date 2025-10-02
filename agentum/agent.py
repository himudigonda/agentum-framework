from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict
from .providers.base import BaseLLM

class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    system_prompt: str
    llm: BaseLLM
    tools: Optional[List[Any]] = None
    memory: Optional[Any] = None
    max_retries: int = 3

    def append_message_for_search(self, message: Any):
        if hasattr(self.memory, 'append_message_for_search'):
            self.memory.append_message_for_search(message)