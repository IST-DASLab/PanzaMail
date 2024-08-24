from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from panza3.entities import Email, EmailInstruction
from panza3.prompting import EmailPromptBuilder
from panza3.retriever import FaissRetriever


def test_email_prompt_builder(
    system_preamble_path: Path,
    user_preamble_path: Path,
    rag_preamble_path: Path,
    thread_preamble_path: Path,
):
    # TODO: Split into multiple tests

    # Patch the retrieve method to return a list of emails
    mock_retriever = MagicMock(spec=FaissRetriever)
    emails = [
        Email(email=f"email{i}", subject=f"subject{i}", thread=[f"thread{i}"], date=datetime.now())
        for i in range(3)
    ]
    mock_retriever.retrieve.return_value = emails

    instruction = EmailInstruction(
        instruction="Write an email.", thread=["email1", "email2", "email3"]
    )

    system_preamble, user_preamble, rag_preamble, thread_preamble = (
        EmailPromptBuilder.load_all_preambles(
            system_preamble_path=system_preamble_path,
            user_preamble_path=user_preamble_path,
            rag_preamble_path=rag_preamble_path,
            thread_preamble_path=thread_preamble_path,
        )
    )

    prompt_builder = EmailPromptBuilder(
        retriever=mock_retriever,
        system_preamble=system_preamble,
        user_preamble=user_preamble,
        rag_preamble=rag_preamble,
        thread_preamble=thread_preamble,
        number_rag_emails=3,
        rag_relevance_threshold=0.0,
        number_thread_emails=1,
    )

    rag_prompt = prompt_builder._create_rag_preamble_from_emails(emails=emails)

    assert rag_prompt == (
        "RAG PREAMBLE:\n\n"
        + "E-MAIL CONTENT:\nemail0\n\n---\n\n"
        + "E-MAIL CONTENT:\nemail1\n\n---\n\n"
        + "E-MAIL CONTENT:\nemail2\n\n---\n\n"
    )

    thread_prompt = prompt_builder._create_threading_preamble(thread=instruction.thread)

    assert thread_prompt == (
        "THREAD PREAMBLE:\n\n" + "email1\n\n---\n\n" + "email2\n\n---\n\n" + "email3\n\n---\n\n"
    )

    # Test full prompt
    prompt = prompt_builder.build_prompt(instruction=instruction, use_rag=True, use_thread=True)
    assert prompt == (
        "<SYSTEM PREAMBLE>\n\n"
        + "<USER PREAMBLE>\n\n"
        + "RAG PREAMBLE:\n\n"
        + "E-MAIL CONTENT:\nemail0\n\n---\n\n"
        + "E-MAIL CONTENT:\nemail1\n\n---\n\n"
        + "E-MAIL CONTENT:\nemail2\n\n---\n\n"
        + "THREAD PREAMBLE:\n\n"
        + "email1\n\n---\n\n"
        + "email2\n\n---\n\n"
        + "email3\n\n---\n\n"
        + "Instruction: Write an email."
    )

    # Test prompt without RAG
    prompt = prompt_builder.build_prompt(instruction=instruction, use_rag=False, use_thread=True)
    assert prompt == (
        "<SYSTEM PREAMBLE>\n\n"
        + "<USER PREAMBLE>\n\n"
        + "THREAD PREAMBLE:\n\n"
        + "email1\n\n---\n\n"
        + "email2\n\n---\n\n"
        + "email3\n\n---\n\n"
        + "Instruction: Write an email."
    )

    # Test prompt without thread
    prompt = prompt_builder.build_prompt(instruction=instruction, use_rag=True, use_thread=False)
    assert prompt == (
        "<SYSTEM PREAMBLE>\n\n"
        + "<USER PREAMBLE>\n\n"
        + "RAG PREAMBLE:\n\n"
        + "E-MAIL CONTENT:\nemail0\n\n---\n\n"
        + "E-MAIL CONTENT:\nemail1\n\n---\n\n"
        + "E-MAIL CONTENT:\nemail2\n\n---\n\n"
        + "Instruction: Write an email."
    )

    # Test prompt without RAG and thread
    prompt = prompt_builder.build_prompt(instruction=instruction, use_rag=False, use_thread=False)
    assert prompt == (
        "<SYSTEM PREAMBLE>\n\n" + "<USER PREAMBLE>\n\n" + "Instruction: Write an email."
    )
