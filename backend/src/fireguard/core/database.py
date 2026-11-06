
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from fireguard.core.config import get_settings


class Base(DeclarativeBase):
    pass


def create_database_engine():
    settings = get_settings()

    connect_args = {}

    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(
        settings.database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


engine = create_database_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
