from langchain_google_genai import ChatGoogleGenerativeAI

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
        # MODIFICATION: REMOVE the explicit `if not api_key: raise AgentumError(...)` check.
        # This allows ADC and environment variables to work correctly without a crash.
        super().__init__(
            api_key=api_key, model=model, temperature=temperature, **kwargs
        )
