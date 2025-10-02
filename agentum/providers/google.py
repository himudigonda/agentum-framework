from langchain_google_genai import ChatGoogleGenerativeAI
from .base import BaseLLM

class GoogleLLM(ChatGoogleGenerativeAI, BaseLLM):

    def __init__(self, api_key: str | None=None, model: str='gemini-2.5-flash-lite', temperature: float=0.7, **kwargs):
        super().__init__(api_key=api_key, model=model, temperature=temperature, **kwargs)