from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


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
        print("Faiss index loading failed \n", e)
