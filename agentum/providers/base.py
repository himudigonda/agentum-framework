from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import BaseMessage


class BaseLLM(ABC):

    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def bind_tools(self, tools: List[Any]) -> "BaseLLM":
        raise NotImplementedError
