from datetime import datetime
from pathlib import Path

import pytest

from panza3.entities import Email
from panza3.retriever import FaissRetriever


@pytest.fixture
def embedding_model() -> str:
    return "sentence-transformers/all-mpnet-base-v2"


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
