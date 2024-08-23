import json
from datetime import datetime

import pytest

from panza3.entities import Email


def test_email_serialization_deserialization():
    email = Email(email="email", subject="subject", thread=["thread"], date=datetime.now())
    serialized = json.dumps(email.serialize())
    deserialized = Email.deserialize(serialized)
    assert email == deserialized
