"""Small SQL persistence layer shared by API and workers."""

from contextlib import contextmanager
import time
import uuid

from sqlalchemy import JSON, BigInteger, Float, Integer, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def uid() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    owner: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(250))
    columns: Mapped[list] = mapped_column(JSON)
    rows: Mapped[int] = mapped_column(Integer)
    checksum: Mapped[str] = mapped_column(String(64))
    original_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Record(Base):
    __tablename__ = "records"
    __table_args__ = (UniqueConstraint("dataset_id", "position"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    dataset_id: Mapped[str] = mapped_column(String(40), index=True)
    position: Mapped[int] = mapped_column(Integer)
    values: Mapped[dict] = mapped_column(JSON)


class Template(Base):
    __tablename__ = "templates"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    owner: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    recipe: Mapped[dict] = mapped_column(JSON)
    export_spec: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    owner: Mapped[str] = mapped_column(String(64), index=True)
    dataset_id: Mapped[str] = mapped_column(String(40), index=True)
    recipe: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    total: Mapped[int] = mapped_column(Integer)
    max_requests: Mapped[int] = mapped_column(Integer, default=10000)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    actual_input_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    updated_at: Mapped[float] = mapped_column(Float, default=time.time)


class WorkItem(Base):
    __tablename__ = "work_items"
    __table_args__ = (UniqueConstraint("job_id", "record_id"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    record_id: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(40), nullable=True)
    lease_until: Mapped[float] = mapped_column(Float, default=0.0)
    available_at: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[dict] = mapped_column(JSON, default=dict)


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("job_id", "record_id", "target_id"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    owner: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    record_id: Mapped[str] = mapped_column(String(40), index=True)
    source_id: Mapped[str] = mapped_column(Text)
    target_id: Mapped[str] = mapped_column(String(100))
    target: Mapped[dict] = mapped_column(JSON)
    features: Mapped[dict] = mapped_column(JSON)
    original_features: Mapped[dict] = mapped_column(JSON)
    predictions: Mapped[list] = mapped_column(JSON)
    reasons: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), index=True)
    origin: Mapped[str] = mapped_column(String(24))
    reviewed: Mapped[bool] = mapped_column(default=False)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    annotation_id: Mapped[str] = mapped_column(String(40), index=True)
    owner: Mapped[str] = mapped_column(String(64))
    previous_revision: Mapped[int] = mapped_column(Integer)
    decision: Mapped[str] = mapped_column(String(20))
    features: Mapped[dict] = mapped_column(JSON)
    note: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class PredictionCache(Base):
    __tablename__ = "prediction_cache"
    __table_args__ = (UniqueConstraint("owner", "cache_key"),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    owner: Mapped[str] = mapped_column(String(64))
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Export(Base):
    __tablename__ = "exports"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    owner: Mapped[str] = mapped_column(String(64), index=True)
    job_id: Mapped[str] = mapped_column(String(40), index=True)
    path: Mapped[str] = mapped_column(Text)
    manifest: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Database:
    def __init__(self, settings):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        opts = {"pool_pre_ping": True}
        if settings.database_url.startswith("sqlite"):
            opts["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if ":memory:" in settings.database_url:
                opts["poolclass"] = StaticPool
        self.engine = create_engine(settings.database_url, **opts)
        if self.engine.dialect.name == "sqlite":
            @event.listens_for(self.engine, "connect")
            def sqlite_setup(connection, _record):
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=30000")
                connection.execute("PRAGMA journal_mode=WAL")
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self):
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self):
        with self.sessions() as session:
            try:
                # SQLite has no row-level FOR UPDATE. Serialize its short local
                # transactions so status/revision checks and writes stay atomic.
                if self.engine.dialect.name == "sqlite":
                    session.connection().exec_driver_sql("BEGIN IMMEDIATE")
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    @contextmanager
    def snapshot(self):
        """A consistent read of annotations, reviews, and processing state."""
        with self.sessions() as session:
            try:
                if self.engine.dialect.name == "postgresql":
                    session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
                elif self.engine.dialect.name == "sqlite":
                    session.connection().exec_driver_sql("BEGIN")
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
