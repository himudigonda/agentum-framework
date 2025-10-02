from langchain_google_genai import ChatGoogleGenerativeAI

# MODIFICATION: Import AgentumError
from agentum.exceptions import AgentumError

from .base import BaseLLM


class GoogleLLM(ChatGoogleGenerativeAI, BaseLLM):
    """
    An Agentum-compatible LLM provider for Google's Gemini models.

    This class inherits from LangChain's ChatGoogleGenerativeAI, ensuring
    it has the necessary 'ainvoke' and 'bind_tools' methods, and also
    formally implements the Agentum BaseLLM interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.7,
        **kwargs,
    ):
        # 1. DEFENSIVE CHECK: If the key is missing, raise a clear Agentum error immediately.
        if not api_key:
            raise AgentumError(
                "GOOGLE_API_KEY not found. Please set the GOOGLE_API_KEY "
                "environment variable or pass it explicitly to GoogleLLM."
            )

        # 2. PROCEED: If the key is present, pass it to the superclass.
        # This bypasses the credentials search path that was causing the deep crash.
        super().__init__(
            api_key=api_key, model=model, temperature=temperature, **kwargs
        )
