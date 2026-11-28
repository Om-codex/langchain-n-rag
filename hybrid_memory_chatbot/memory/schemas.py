"""This is a Data contracts for long-term memory extraction and storage."""

from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator

class MemoryType(str, Enum):
  """Allowed categories for a memory."""
  FACT = "fact"
  PREFERENCE = "preference"
  GOAL = "goal"
  CONSTRAINT = "constraint"

class MemoryCandidate(BaseModel):
  """One durable fact/preference extracted from a conversation turn."""

  content: str = Field(
    ...,
    min_length = 1,
    max_length = 500,
    description = "A short, standalone statement of the memory.",
  )
  memory_type: MemoryType = Field(
    default = MemoryType.FACT,
    description = "Category of the memory.",
  )
  importance: int = Field(
    default = 3,
    ge = 1,
    le = 5,
    description = "1 = trivial, 5 = essential. Used for prompt budgeting later.",
  )
  memory_key: str = Field(
    ...,
    min_length = 1,
    max_length = 100,
    description = "Stable snake_case key for dedupe, e.g. preferred language.",
  )
  evidence: list[str] = Field(
    default_factory = list,
    max_length = 5,
    description = "Short quotes from the conversation that support this memory.",
  )

  @field_validator("content", "memory_key")
  @classmethod
  def _strip_text(cls, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
      raise ValueError("value cannot be empty")
    return cleaned

  @field_validator("memory_key")
  @classmethod
  def _normalize_key(cls, value: str) -> str:
    return "_".join(value.lower().replace("-","_").split())

class MemoryExtraction(BaseModel):
  """Output of one Groq extraction pass."""
  memories: list[MemoryCandidate] = Field(
    default_factory=list,
    max_length=10,
    description="Durable memories found in this turn. Empty list if none.",
  )

class StoredMemory(MemoryCandidate):
  """A memory enriched with metadata before it is writtent to Qdrant."""

  memory_id: str
  user_id: str
  session_id: str
  source_message_ids: list[int] = Field(default_factory=list)
  created_at: datetime = Field(
    default_factory=lambda: datetime.now(timezone.utc)
  )
  updated_at: datetime = Field(
    default_factory=lambda: datetime.now(timezone.utc)
  )
  superseded: bool = False

  def to_payload(self) -> dict:
    """Convert to the payload shape used by the project's Qdrant stores."""
    return {
        "page_content": self.content,
        "metadata": {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "memory_type": self.memory_type.value,
            "memory_key": self.memory_key,
            "importance": self.importance,
            "evidence": self.evidence,
            "source_message_ids": self.source_message_ids,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "superseded": self.superseded,
          },
        }