from langchain_anthropic import ChatAnthropic

from .base import BaseLLM


class AnthropicLLM(ChatAnthropic, BaseLLM):
    """
    An Agentum-compatible LLM provider for Anthropic's Claude models.

    This class inherits from LangChain's ChatAnthropic, ensuring
    it has the necessary 'ainvoke' and 'bind_tools' methods, and also
    formally implements the Agentum BaseLLM interface.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
