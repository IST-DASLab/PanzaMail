from unittest.mock import MagicMock

import pytest

from panza3.entities import EmailInstruction
from panza3.llm import LLM
from panza3.prompting import EmailPromptBuilder
from panza3.writer import PanzaWriter


def test_email_writer_init():
    # Create mock prompt builder
    mock_builder = MagicMock(spec=EmailPromptBuilder)
    mock_builder.build_prompt.side_effect = (
        lambda instruction: f"Instruction: {instruction.instruction}"
    )

    # Create mock LLM
    mock_llm = MagicMock(spec=LLM)
    mock_llm.chat.side_effect = lambda messages: [f"Received: {messages[0]['content']}"]

    panza_writer = PanzaWriter(mock_builder, mock_llm)

    instruction = EmailInstruction(instruction="Write an email.")

    output = panza_writer.run(instruction)
    assert output == "Received: Instruction: Write an email."
