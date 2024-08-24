import copy
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument


@dataclass
class Document(ABC):
    summary: Optional[str] = None

    @abstractmethod
    def serialize(self) -> dict:
        """Convert the document to a dictionary that can be serialized to JSON."""
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: Union[str, Dict]) -> "Document":
        """Convert a serialized document into a Document object."""
        pass

    @staticmethod
    @abstractmethod
    def process(
        documents: List["Document"], chunk_size: int, chunk_overlap: int
    ) -> List[LangchainDocument]:
        """Prepare documents for storage."""
        pass


@dataclass(kw_only=True)
class Email(Document):
    email: str
    subject: str
    thread: List[str] = field(default_factory=list)
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

    @staticmethod
    def process(documents: List["Email"], chunk_size, chunk_overlap) -> List[Document]:
        # Convert e-mails to langchain documents
        documents = [
            LangchainDocument(page_content=email.email, metadata={"serialized_document": email.serialize()})
            for email in documents
        ]

        # Split long e-mails into text chuncks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(documents)

        return documents
