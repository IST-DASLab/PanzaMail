from datetime import datetime
from pathlib import Path

import pytest

from panza.entities import Email
from panza.retriever import FaissRetriever


@pytest.fixture
def embedding_model() -> str:
    return "sentence-transformers/all-mpnet-base-v2"


@pytest.fixture
def generative_model() -> str:
    return "microsoft/Phi-3-mini-4k-instruct"


@pytest.fixture
def peft_model() -> str:
    return "microsoft/Phi-3-mini-4k-instruct"


@pytest.fixture
def index_name() -> str:
    return "test-index"


@pytest.fixture(scope="function")
def faiss_db_path(tmp_path: Path, index_name: str, embedding_model: str) -> Path:
    # Create a new temporary directory for each test
    base_temp_dir = tmp_path / "data"
    base_temp_dir.mkdir()  # Ensure the data directory is created

    # Define the mock emails
    emails = [
        Email(email=f"email{i}", subject=f"subject{i}", thread=[f"thread{i}"], date=datetime.now())
        for i in range(3)
    ]

    # Initialize the FaissRetriever
    retriever = FaissRetriever(
        db_path=base_temp_dir,
        index_name=index_name,
        embedding_model=embedding_model,
        device="cpu",
        document_class=Email,
    )

    # Store the mock emails in the vector database
    retriever.store(emails, chunk_size=1000, chunk_overlap=1000)
    retriever.save_db_to_disk()

    # Return the path to the directory containing all mock data
    return base_temp_dir


@pytest.fixture
def preambles_path(tmp_path: Path) -> Path:
    preambles_path = tmp_path / "prompt_preambles"
    preambles_path.mkdir(parents=True)
    return preambles_path


@pytest.fixture
def system_preamble_path(preambles_path) -> Path:
    system_preamble_path = preambles_path / "system_preamble.txt"
    system_preamble_path.write_text("<SYSTEM PREAMBLE>")
    return system_preamble_path


@pytest.fixture
def user_preamble_path(preambles_path) -> Path:
    user_preamble_path = preambles_path / "user_preamble.txt"
    user_preamble_path.write_text("<USER PREAMBLE>")
    return user_preamble_path


@pytest.fixture
def rag_preamble_path(preambles_path) -> Path:
    rag_preamble_path = preambles_path / "rag_preamble.txt"
    rag_preamble_path.write_text("RAG PREAMBLE:\n\n{rag_context}")
    return rag_preamble_path


@pytest.fixture
def thread_preamble_path(preambles_path) -> Path:
    thread_preamble_path = preambles_path / "thread_preamble.txt"
    thread_preamble_path.write_text("THREAD PREAMBLE:\n\n{threading_context}")
    return thread_preamble_path
