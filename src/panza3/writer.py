from typing import Iterator, List

from .entities import Instruction
from .llm import LLM, MessageType
from .prompting import PromptBuilder


# TODO: Check that instruction type is compatible with prompt_builder type?
class PanzaWriter:
    def __init__(self, prompt_builder: PromptBuilder, llm: LLM):
        self.prompt_builder = prompt_builder
        self.llm = llm

    def run(self, instruction: Instruction, stream: bool = False) -> str | Iterator[str]:
        prompt = self.prompt_builder.build_prompt(instruction)
        messages = self._create_user_message(content=prompt)
        if stream:
            return self.llm.chat_stream(messages)
        else:
            return self.llm.chat(messages)[0]

    def run_batch(self, instructions: List[Instruction]) -> List[str]:
        prompts = [self.prompt_builder.build_prompt(instruction) for instruction in instructions]
        messages = [self._create_user_message(content=prompt) for prompt in prompts]
        return self.llm.chat(messages)

    def _create_user_message(self, content: str) -> MessageType:
        return [{"role": "user", "content": content}]
