# agentum/llm_providers.py
from langchain_google_genai import ChatGoogleGenerativeAI


# This is a simple wrapper for convenience. We can add more providers (OpenAI, Anthropic) here later.
class GoogleLLM(ChatGoogleGenerativeAI):
    """
    A convenience wrapper for the LangChain ChatGoogleGenerativeAI model.
    Initializes the model with Gemini 2.5 Flash and allows easy configuration.
    """

    def __init__(
        self, model: str = "gemini-2.5-flash-lite", temperature: float = 0.7, **kwargs
    ):
        # Pass the 'model' argument which is expected by the parent class.
        super().__init__(model=model, temperature=temperature, **kwargs)
