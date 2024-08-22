import copy
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union


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


@dataclass(kw_only=True)
class Email(Document):
    email: str
    subject: str
    thread: List[str]
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
