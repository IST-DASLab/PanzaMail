from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Literal

MessageType = Dict[Literal["role", "content"], str]
ChatHistoryType = List[MessageType]


class LLM(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def chat(self, messages: ChatHistoryType) -> str:
        pass

    @abstractmethod
    def chat_stream(self, messages: ChatHistoryType) -> Iterator[str]:
        pass
