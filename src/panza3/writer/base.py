from abc import ABC, abstractmethod

from ..entities import Instruction
from ..llm import LLM
from ..prompting import PromptBuilder


class PanzaWriter(ABC):
    def __init__(self, prompt_builder: PromptBuilder, llm: LLM):
        self.prompt_builder = prompt_builder
        self.llm = llm

    @abstractmethod
    def run(self, instruction: Instruction) -> str:
        pass
