from langchain_anthropic import ChatAnthropic

# MODIFICATION: Import AgentumError
from agentum.exceptions import AgentumError

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
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.7,
        **kwargs,
    ):
        # 1. DEFENSIVE CHECK: If the key is missing, raise a clear Agentum error immediately.
        if not api_key:
            raise AgentumError(
                "ANTHROPIC_API_KEY not found. Please set the ANTHROPIC_API_KEY "
                "environment variable or pass it explicitly to AnthropicLLM."
            )

        # 2. PROCEED: If the key is present, pass it to the superclass.
        super().__init__(
            api_key=api_key, model=model, temperature=temperature, **kwargs
        )
