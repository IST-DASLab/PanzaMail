import json
from datetime import datetime

import pytest

from panza3.entities import Email


def test_email_serialization_deserialization():
    email = Email(email="email", subject="subject", thread=["thread"], date=datetime.now())
    serialized = json.dumps(email.serialize())
    deserialized = Email.deserialize(serialized)
    assert email == deserialized


def test_email_processing():
    email = Email(email="email", subject="subject", thread=["thread"], date=datetime.now())
    processed = Email.process([email], chunk_size=1000, chunk_overlap=1000)
    assert processed[0].page_content == email.email

    deserialized = Email.deserialize(processed[0].metadata["serialized_document"])
    assert email == deserialized