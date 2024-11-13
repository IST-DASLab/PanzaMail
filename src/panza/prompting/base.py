from abc import ABC, abstractmethod

from ..entities import Instruction
from ..retriever import DocumentRetriever


class PromptBuilder(ABC):
    def __init__(self, retriever: DocumentRetriever):
        self.retriever = retriever

    @abstractmethod
    def build_prompt(self, instruction: Instruction) -> str:
        pass
