from unittest.mock import MagicMock

import pytest

from panza.entities import EmailInstruction
from panza.llm import LLM
from panza.prompting import EmailPromptBuilder
from panza.writer import PanzaWriter


def test_email_writer():
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

    output, prompt = panza_writer.run(instruction, return_prompt=True)
    assert output == "Received: Instruction: Write an email."
    assert prompt == "Instruction: Write an email."
