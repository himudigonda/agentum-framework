from langchain_anthropic import ChatAnthropic

from .base import BaseLLM


class AnthropicLLM(ChatAnthropic, BaseLLM):

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key, model=model, temperature=temperature, **kwargs
        )
