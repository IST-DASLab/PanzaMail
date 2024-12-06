import logging
from typing import List, Optional, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from ..entities.document import Document
from .base import DocumentRetriever

LOGGER = logging.getLogger(__name__)


class FaissRetriever(DocumentRetriever):
    def __init__(
        self,
        db_path: str,
        index_name: str,
        embedding_model: str,
        device: str,
        document_class: Optional[type[Document]] = None,
    ) -> None:

        self.db_path = db_path
        self.index_name = index_name
        self.model_name = embedding_model
        self.device = device
        self.document_class = document_class

        self.embedding_model = self._get_embeddings_model(self.model_name, self.device)
        self.db = self._load_vector_db_from_disk(
            self.db_path, self.index_name, self.embedding_model
        )

    def _get_embeddings_model(self, model_name: str, device: str) -> Embeddings:
        embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": False},
        )
        return embeddings_model

    def _load_vector_db_from_disk(
        self, db_path: str, index_name: str, embeddings_model: Embeddings
    ) -> VectorStore:
        try:
            db = FAISS.load_local(
                folder_path=db_path,
                embeddings=embeddings_model,
                index_name=index_name,
                allow_dangerous_deserialization=True,  # Allows pickle deserialization
            )
            LOGGER.info(f"Loaded Faiss index {index_name} from {db_path}.")
            return db
        except Exception as e:
            LOGGER.error(
                f"Failed to load Faiss index {index_name} from {db_path}. Error: {e}\nPLEASE NOTE: if you have RAG enabled in training or inference, and do not have the Faiss index, you will fail."
            )

    def retrieve(self, query: str, k: int, score: Optional[float] = None) -> List[Document]:
        results = self.retrieve_with_score(query, k, score)
        results = [r[0] for r in results]
        return results

    def retrieve_with_score(
        self, query: str, k: int, score: Optional[float] = None
    ) -> List[Tuple[Document, float]]:

        results = self.db._similarity_search_with_relevance_scores(query, k=k)

        # Filter by score
        if score is not None:
            results = [r for r in results if r[1] >= score]

        # Deserialize metadata
        results = [
            (self.document_class.deserialize(r[0].metadata["serialized_email"]), r[1])
            for r in results
        ]

        return results

    def store(self, documents: List[Document], chunk_size: int, chunk_overlap: int):
        documents = self.document_class.process(
            documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        db = FAISS.from_documents(documents, self.embedding_model)

        if self.db:
            self.db.merge_from(db)
        else:
            LOGGER.info(f"Creating new Faiss index {self.index_name} in {self.db_path}.")
            self.db = db

    def save_db_to_disk(self):
        # Save vector DB to disk
        self.db.save_local(folder_path=self.db_path, index_name=self.index_name)
        logging.info(f"Vector DB index {self.index_name} saved to {self.db_path}.")
