"""Database engine, session factory and table initialization.

Supports PostgreSQL (psycopg v3) and SQLite (useful for quick local tests).
"""

from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Load env files before config.py builds its global Settings object.
# The package-level .env wins over the repository-root .env.
_ROOT_DIR = Path(__file__).resolve().parent.parent
_PACKAGE_DIR = Path(__file__).resolve().parent

load_dotenv(_ROOT_DIR / ".env")
load_dotenv(_PACKAGE_DIR / ".env", override=True)

from config import settings  # noqa: E402

DATABASE_URL = settings.database_url
_is_sqlite = DATABASE_URL.startswith("sqlite")

_connect_args = {"check_same_thread": False} if _is_sqlite else {}

if _is_sqlite:
    _sqlite_path = Path(make_url(DATABASE_URL).database or "")
    if str(_sqlite_path) not in {"", ":memory:"}:
        _sqlite_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

# chains/chatbot.py imports this name.
SessionFactory = SessionLocal


def init_db() -> None:
    """Create missing tables.

    Importing the models registers them on ``Base`` before create_all runs.
    """
    from db.models import ChatMessage, ChatSession  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a DB session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
