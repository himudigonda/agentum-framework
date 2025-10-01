from abc import ABC, abstractmethod
from typing import Any, List

from langchain_core.messages import BaseMessage


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers in Agentum.

    This class defines the essential interface that the Agentum engine
    expects from any language model, ensuring interchangeability.
    """

    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> Any:
        """
        Asynchronously invoke the language model with a list of messages.

        Args:
            messages: A list of message objects (e.g., HumanMessage, AIMessage).

        Returns:
            The model's response, typically an AIMessage object.
        """
        raise NotImplementedError

    @abstractmethod
    def bind_tools(self, tools: List[Any]) -> "BaseLLM":
        """
        Bind a list of tools to the language model.

        This makes the model aware of the available tools and their schemas,
        enabling it to generate tool-calling requests.

        Args:
            tools: A list of agentum tools.

        Returns:
            A new instance of the LLM with the tools bound.
        """
        raise NotImplementedError
