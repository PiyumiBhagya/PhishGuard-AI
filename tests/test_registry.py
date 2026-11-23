import io

import joblib
import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.db import engine
from app.registry import ModelRecord, ensure_table, newer_than, publish, version_number


@pytest.fixture(autouse=True)
def clean_registry():
    ensure_table()
    with Session(engine) as s:
        s.execute(delete(ModelRecord))
        s.commit()
    yield
    with Session(engine) as s:
        s.execute(delete(ModelRecord))
        s.commit()


def _artifact(payload) -> bytes:
    buf = io.BytesIO()
    joblib.dump(payload, buf)
    return buf.getvalue()


def test_version_number():
    assert version_number("v1") == 1
    assert version_number("v12") == 12
    assert version_number("unknown") == 0


def test_empty_registry_has_nothing_newer():
    assert newer_than("v1") is None


def test_publish_and_fetch_a_newer_model():
    publish("v2", '{"model_version": "v2"}', _artifact({"hello": "world"}))
    version, info, artifact = newer_than("v1")
    assert version == "v2"
    assert info["model_version"] == "v2"
    assert joblib.load(io.BytesIO(artifact)) == {"hello": "world"}


def test_same_or_older_registry_model_is_ignored():
    publish("v2", '{"model_version": "v2"}', _artifact(1))
    assert newer_than("v2") is None
    assert newer_than("v3") is None


def test_publishing_the_same_version_again_replaces_it():
    publish("v2", '{"model_version": "v2", "n": 1}', _artifact("old"))
    publish("v2", '{"model_version": "v2", "n": 2}', _artifact("new"))
    _, info, artifact = newer_than("v1")
    assert info["n"] == 2
    assert joblib.load(io.BytesIO(artifact)) == "new"