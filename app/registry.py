"""Model registry, stored in the same database as the prediction log.

The retrain job publishes every new champion here. The API checks the registry every
few minutes and loads a newer model without a redeploy.
"""
import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, LargeBinary, String, Text, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.db import Base, engine


class ModelRecord(Base):
    __tablename__ = "model_registry"

    version: Mapped[str] = mapped_column(String(20), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    info_json: Mapped[str] = mapped_column(Text)          # same content as champion_metrics.json
    artifact: Mapped[bytes] = mapped_column(LargeBinary)  # the champion.joblib file


_table_ready = False


def ensure_table() -> None:
    global _table_ready
    if not _table_ready:
        ModelRecord.__table__.create(engine, checkfirst=True)
        _table_ready = True


def version_number(version: str) -> int:
    try:
        return int(str(version).lstrip("v"))
    except ValueError:
        return 0


def publish(version: str, info_json: str, artifact: bytes) -> None:
    ensure_table()
    with Session(engine) as s:
        s.merge(ModelRecord(version=version, created_at=datetime.now(timezone.utc),
                            info_json=info_json, artifact=artifact))
        s.commit()


def latest_version() -> str | None:
    ensure_table()
    with Session(engine) as s:
        return s.scalar(select(ModelRecord.version).order_by(ModelRecord.created_at.desc()).limit(1))


def newer_than(current_version: str) -> tuple[str, dict, bytes] | None:
    """The registry's newest model if it is newer than `current_version`, otherwise None."""
    latest = latest_version()
    if latest is None or version_number(latest) <= version_number(current_version):
        return None
    with Session(engine) as s:
        row = s.get(ModelRecord, latest)
        return row.version, json.loads(row.info_json), bytes(row.artifact)