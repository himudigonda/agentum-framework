# agentum/llm_providers.py
from langchain_openai import ChatOpenAI


# This is a simple wrapper for convenience. We can add more providers (Gemini, Anthropic) here later.
class OpenAILLM(ChatOpenAI):
    """
    A convenience wrapper for the LangChain ChatOpenAI model.
    Initializes the model with a default and allows easy configuration.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, **kwargs):
        # Pass the 'model_name' argument which is expected by the parent class.
        super().__init__(model_name=model, temperature=temperature, **kwargs)
