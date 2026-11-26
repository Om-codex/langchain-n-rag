from datetime import datetime, timezone

from sqlalchemy import (
  String, Integer, DateTime, Text, ForeignKey
)
from sqlalchemy.orm import Mapped, mapped_column
from db.connection import Base

class ChatSession(Base):

  __tablename__ = "chat_sessions"

  session_id: Mapped[str] = mapped_column(
    String(200),
    primary_key=True
  )
  user_id: Mapped[str] = mapped_column(
    String(100),
    index=True,
    nullable=False
  )
  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default= lambda: datetime.now(timezone.utc),
    nullable=False,
  )

class ChatMessage(Base):

  __tablename__ = "chat_messages"

  id: Mapped[int] = mapped_column(
    Integer,
    primary_key=True,
    autoincrement=True
  )

  session_id: Mapped[str] = mapped_column(
    ForeignKey("chat_sessions.session_id"),
    index=True,
    nullable=False
  )

  role: Mapped[str] = mapped_column(
    String(20),
    nullable=False
  )

  content: Mapped[str] = mapped_column(
    Text,
    nullable=False
  )

  created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True), 
    default = lambda: datetime.now(timezone.utc),
    nullable=False
  )


