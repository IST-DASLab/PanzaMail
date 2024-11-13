import copy
import json
import time
from abc import ABC
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass(kw_only=True)
class Email(ABC):
    email: str
    subject: str
    thread: List[str]
    summary: Optional[str] = None
    date: datetime

    def serialize(self) -> dict:
        dictionary = asdict(self)
        dictionary["date"] = self.date.isoformat()
        return dictionary

    @classmethod
    def deserialize(cls, data: Union[str, Dict]) -> "Email":
        if isinstance(data, str):
            dictionary = json.loads(data)
        elif isinstance(data, dict):
            dictionary = copy.deepcopy(data)
        else:
            raise ValueError(f"Cannot deserialize data of type {type(data)}. Must be str or dict.")
        dictionary["date"] = datetime.fromisoformat(dictionary["date"])
        return cls(**dictionary)


def get_embeddings_model(model_name) -> Embeddings:
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    return embeddings_model


def create_vector_db(docs: List[Document], embeddings_model: Embeddings) -> VectorStore:
    db = FAISS.from_documents(docs, embeddings_model)
    return db


def load_vector_db_from_disk(
    folder_path: str, index_name: str, embeddings_model: Embeddings
) -> VectorStore:
    try:
        db = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings_model,
            index_name=index_name,
            allow_dangerous_deserialization=True,  # Allows pickle deserialization
        )
        print("Faiss index loaded ")
        return db
    except Exception as e:
        print("FAISS index loading failed \n", e)


def load_emails(path: str) -> List[Email]:
    with open(path, "r") as f:
        lines = f.readlines()

    emails = [Email.deserialize(line) for line in lines]

    return emails


def process_emails(emails: List[Email], chunk_size: int, chunk_overlap: int) -> List[Document]:
    # Convert e-mails to langchain documents
    documents = [
        Document(page_content=email.email, metadata={"serialized_email": email.serialize()})
        for email in emails
    ]

    # Split long e-mails into text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(documents)

    return documents


def create_vector_store(
    path_to_emails, chunk_size, chunk_overlap, db_path, index_name, embedding_model
):
    """Create FAISS vector database for search and retrieval."""
    # Load emails
    emails = load_emails(path_to_emails)
    print(f"Loaded {len(emails)} emails.")

    # Process emails
    documents = process_emails(emails, chunk_size, chunk_overlap)
    print(f"Obtained {len(documents)} text chunks.")

    # Initialize embeddings model
    embeddings_model = get_embeddings_model(embedding_model)

    # Create vector DB
    print("Creating vector DB...")
    start = time.time()
    db = create_vector_db(documents, embeddings_model)
    print(f"Vector DB created in {time.time() - start} seconds.")

    # Save vector DB to disk
    db.save_local(folder_path=db_path, index_name=index_name)
    print(f"Vector DB index {index_name} saved to {db_path}.")
