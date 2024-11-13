from datetime import datetime
from pathlib import Path

import pytest

from panza.entities import Email
from panza.retriever import FaissRetriever


def get_faiss_retriever(
    db_path: Path, index_name: str, embedding_model: str, device: str
) -> FaissRetriever:
    retriever = FaissRetriever(
        db_path=db_path,
        index_name=index_name,
        embedding_model=embedding_model,
        device=device,
    )
    retriever.set_document_class(Email)
    return retriever


def test_faiss_retriever_init_empty(tmp_path: Path, index_name: str, embedding_model: str):
    retriever = get_faiss_retriever(tmp_path, index_name, embedding_model, "cpu")
    assert retriever is not None
    assert retriever.embedding_model is not None
    assert retriever.db is None


def test_faiss_retriever_init_existing(faiss_db_path: Path, index_name: str, embedding_model: str):
    retriever = get_faiss_retriever(faiss_db_path, index_name, embedding_model, "cpu")
    assert retriever is not None
    assert retriever.embedding_model is not None
    assert retriever.db is not None


def test_faiss_retriever_store_over_empty(tmp_path: Path, index_name: str, embedding_model: str):
    retriever = get_faiss_retriever(tmp_path, index_name, embedding_model, "cpu")

    emails = [
        Email(email=f"email{i}", subject=f"subject{i}", thread=[f"thread{i}"], date=datetime.now())
        for i in range(3)
    ]

    retriever.store(emails, chunk_size=1000, chunk_overlap=1000)
    assert retriever.db is not None


def test_faiss_retriever_store_over_existing(
    faiss_db_path: Path, index_name: str, embedding_model: str
):
    retriever = get_faiss_retriever(faiss_db_path, index_name, embedding_model, "cpu")
    assert retriever.db is not None

    number_existing_documents = len(retriever.db.index_to_docstore_id)
    assert number_existing_documents != 0

    number_new_documents = 3
    emails = [
        Email(email=f"email{i}", subject=f"subject{i}", thread=[f"thread{i}"], date=datetime.now())
        for i in range(number_new_documents)
    ]

    retriever.store(emails, chunk_size=1000, chunk_overlap=1000)

    number_total_documents = len(retriever.db.index_to_docstore_id)
    assert number_total_documents == number_existing_documents + number_new_documents
