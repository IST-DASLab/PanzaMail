from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Literal

MessageType = Dict[Literal["role", "content"], str]
ChatHistoryType = List[MessageType]


class LLM(ABC):
    def __init__(self, name: str, sampling_parameters: Dict):
        self.name = name
        self.sampling_parameters = sampling_parameters

    @abstractmethod
    def chat(self, messages: ChatHistoryType | List[ChatHistoryType]) -> List[str]:
        pass

    @abstractmethod
    def chat_stream(self, messages: ChatHistoryType) -> Iterator[str]:
        pass
