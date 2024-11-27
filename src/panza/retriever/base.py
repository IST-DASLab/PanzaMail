from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..entities.document import Document


class DocumentRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int, score: Optional[float] = None) -> List[Document]:
        pass

    @abstractmethod
    def retrieve_with_score(
        self, query: str, k: int, score: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    def store(self, documents: List[Document]):
        pass

    @abstractmethod
    def save_db_to_disk(self):
        pass

    def set_document_class(self, document_class: type[Document]):
        self.document_class = document_class
