import copy
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
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

        # For backward compatibility, backfill fields not present in V1 of the data.
        if "subject" not in dictionary:
            dictionary["subject"] = ""
        if "thread" not in dictionary:
            dictionary["thread"] = []

        if "date" not in dictionary:  # Date was also missing in V1.
            dictionary["date"] = datetime.min
        else:
            dictionary["date"] = datetime.fromisoformat(dictionary["date"])
        # Clean out all unexpected keys from input dictionary to avoid errors with dataclass.
        field_names = set(f.name for f in fields(Email))
        return cls(**{k: v for k, v in dictionary.items() if k in field_names})

    @staticmethod
    def process(documents: List["Email"], chunk_size, chunk_overlap) -> List[Document]:
        # Convert e-mails to langchain documents
        documents = [
            LangchainDocument(
                page_content=email.email, metadata={"serialized_document": email.serialize()}
            )
            for email in documents
        ]

        # Split long e-mails into text chuncks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(documents)

        return documents
