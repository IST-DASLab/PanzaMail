from typing import Iterator, List, Tuple

from .entities import Instruction
from .llm import LLM, MessageType
from .prompting import PromptBuilder


class PanzaWriter:
    def __init__(self, prompt_builder: PromptBuilder, llm: LLM):
        self.prompt_builder = prompt_builder
        self.llm = llm

    def run(
        self,
        instruction: Instruction,
        stream: bool = False,
        iterator: bool = False,
        return_prompt: bool = False,
    ) -> str | Iterator[str] | Tuple[str, str] | Tuple[Iterator[str], str]:
        prompt = self.prompt_builder.build_prompt(instruction)
        messages = self._create_user_message(content=prompt)

        if stream:
            response = self.llm.chat_stream(messages)
        else:
            response = self.llm.chat(messages)[0]

        if return_prompt:
            return response, prompt
        else:
            return response

    def run_batch(
        self, instructions: List[Instruction], return_prompt: bool = False
    ) -> List[str] | Tuple[List[str], List[str]]:
        prompts = [self.prompt_builder.build_prompt(instruction) for instruction in instructions]
        messages = [self._create_user_message(content=prompt) for prompt in prompts]

        response = self.llm.chat(messages)

        if return_prompt:
            return response, prompts
        else:
            return response

    def _create_user_message(self, content: str) -> MessageType:
        return [{"role": "user", "content": content}]
