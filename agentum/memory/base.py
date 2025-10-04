from typing import List

from langchain_core.messages import BaseMessage


class BaseMemory:

    def load_messages(self, latest_input: BaseMessage) -> List[BaseMessage]:
        raise NotImplementedError

    def save_messages(self, messages: List[BaseMessage]):
        raise NotImplementedError
