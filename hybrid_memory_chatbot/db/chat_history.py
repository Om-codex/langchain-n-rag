from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from sqlalchemy import delete, select
from sqlalchemy.orm import sessionmaker

from db.models import ChatSession, ChatMessage

class PostgresChatMessageHistory(BaseChatMessageHistory):
  def __init__(self, session_id: str, user_id: str, SessionFactory: sessionmaker):
    self.session_id = session_id
    self.user_id = user_id
    self.SessionFactory = SessionFactory
    self._ensure_session()

  def _ensure_session(self)->None:
    with self.SessionFactory() as db:
      if db.get(ChatSession, self.session_id) is None:
        db.add(ChatSession(session_id=self.session_id, user_id=self.user_id))
        db.commit()

  @property
  def messages(self)->list[BaseMessage]:
    with self.SessionFactory() as db:
      rows = db.scalar(
        select(ChatMessage)
        .where(ChatMessage.session_id == self.session_id)
        .order_by(ChatMessage.id.asc())
      ).all()

    history: list[BaseMessage] = []

    for row in rows:
      if row.role == "human":
        history.append(HumanMessage(content=row.content))
      elif row.role == "ai":
        history.append(AIMessage(content=row.content))
      elif row.role == "system":
        history.append(SystemMessage(content=row.content))

    return history

  def add_message(self, message: BaseMessage) -> None:
    with self.SessionFactory() as db:
      db.add(
        ChatMessage(
          session_id=self.session_id,
          role=message.type,
          content=message.content,
        )
      )
      db.commit()

  def clear(self) -> None:
    with self.SessionFactory() as db:
      db.execute(
        delete(ChatMessage).where(ChatMessage.session_id == self.session_id)
      )
      db.commit()