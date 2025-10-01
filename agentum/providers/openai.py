from langchain_openai import ChatOpenAI

from .base import BaseLLM


class OpenAILLM(ChatOpenAI, BaseLLM):
    """
    An Agentum-compatible LLM provider for OpenAI's GPT models.

    This class inherits from LangChain's ChatOpenAI, ensuring
    it has the necessary 'ainvoke' and 'bind_tools' methods, and also
    formally implements the Agentum BaseLLM interface.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        **kwargs,
    ):
        # We call the parent constructor from LangChain's class
        super().__init__(model=model, temperature=temperature, **kwargs)
