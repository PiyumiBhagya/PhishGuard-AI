"""Prediction log. SQLite locally; set DATABASE_URL to use Postgres in the cloud."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def _normalise(url: str) -> str:
    # some hosts hand out "postgres://", SQLAlchemy wants "postgresql://"
    return url.replace("postgres://", "postgresql://", 1) if url.startswith("postgres://") else url


DATABASE_URL = _normalise(os.getenv("DATABASE_URL", "sqlite:///data/predictions.db"))
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

if _IS_SQLITE:
    Path(DATABASE_URL.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    pool_pre_ping=True,
)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    url: Mapped[str] = mapped_column(Text)
    features_json: Mapped[str] = mapped_column(Text)
    probability: Mapped[float] = mapped_column(Float)
    verdict: Mapped[str] = mapped_column(String(20))
    model_version: Mapped[str] = mapped_column(String(20))
    latency_ms: Mapped[float] = mapped_column(Float)
    # filled later by /feedback: 1 = phishing, 0 = legitimate
    true_label: Mapped[int | None] = mapped_column(Integer, nullable=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def log_prediction(*, request_id, url, features, probability, verdict,
                   model_version, latency_ms) -> None:
    with Session(engine) as s:
        s.add(Prediction(
            request_id=request_id,
            created_at=datetime.now(timezone.utc),
            url=url,
            features_json=json.dumps(features),
            probability=probability,
            verdict=verdict,
            model_version=model_version,
            latency_ms=latency_ms,
        ))
        s.commit()

def log_predictions(rows: list[dict]) -> None:
    """Log many predictions in ONE transaction (one network round trip to the database)."""
    now = datetime.now(timezone.utc)
    with Session(engine) as s:
        s.add_all([
            Prediction(
                request_id=r["request_id"],
                created_at=now,
                url=r["url"],
                features_json=json.dumps(r["features"]),
                probability=r["probability"],
                verdict=r["verdict"],
                model_version=r["model_version"],
                latency_ms=r["latency_ms"],
            )
            for r in rows
        ])
        s.commit()

def set_feedback(request_id: str, true_label: int) -> bool:
    with Session(engine) as s:
        row = s.get(Prediction, request_id)
        if row is None:
            return False
        row.true_label = true_label
        s.commit()
        return True


def stats() -> dict:
    with Session(engine) as s:
        total = s.scalar(select(func.count()).select_from(Prediction))
        labeled = s.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.true_label.is_not(None))
        )
        by_verdict = s.execute(
            select(Prediction.verdict, func.count()).group_by(Prediction.verdict)
        ).all()
    return {
        "total_predictions": total,
        "labeled": labeled,
        "by_verdict": {v: c for v, c in by_verdict},
    }