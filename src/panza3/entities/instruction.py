from abc import ABC
from dataclasses import dataclass
from typing import List

from ..llm import ChatHistoryType


@dataclass
class Instruction(ABC):
    instruction: str
    past_messages: ChatHistoryType


@dataclass(kw_only=True)
class EmailInstruction(Instruction):
    thread: List[str]
