from langchain_openai import ChatOpenAI

from .base import BaseLLM


class OpenAILLM(ChatOpenAI, BaseLLM):

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key, model=model, temperature=temperature, **kwargs
        )
