import logging
from typing import List, Optional, Tuple

from ..entities.document import Document
from .base import DocumentRetriever

LOGGER = logging.getLogger(__name__)


class NoneRetriever(DocumentRetriever):
    def __init__(
        self,
        document_class: Optional[type[Document]] = None,
    ) -> None:
        self.document_class = document_class

    def retrieve(self, query: str, k: int, score: Optional[float] = None) -> List[Document]:
        return []

    def retrieve_with_score(
        self, query: str, k: int, score: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        return []

    def store(self, documents: List[Document], chunk_size: int, chunk_overlap: int):
        pass

    def save_db_to_disk(self):
        pass
