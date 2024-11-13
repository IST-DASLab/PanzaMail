from ..entities import SummarizationInstruction
from .base import PromptBuilder


class SummarizationPromptBuilder(PromptBuilder):
    def __init__(
        self,
        summarization_prompt: str,
    ):
        self.summarization_prompt = summarization_prompt

    def build_prompt(
        self,
        instruction: SummarizationInstruction,
    ) -> str:

        prompt = self.summarization_prompt.format(email=instruction.instruction).strip()

        return prompt
