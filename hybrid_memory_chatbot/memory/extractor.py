"""Groq-powered extraction of durable user memories."""
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_PACKAGE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_PACKAGE_DIR/".env",override=True)

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import settings
from memory.schemas import MemoryCandidate, MemoryExtraction

EXTRACTION_SYSTEM_PROMPT="""
Rules:
1. Extract ONLY durable information about the user: identity, stable preferences,
   goals, constraints, or recurring context.
2. Ignore one-off questions, document contents, and generic small talk.
3. Do not extract anything the assistant said unless the user explicitly agreed
   with it or repeated it.
4. Never invent or guess. If nothing durable is present, return an empty list.
5. Write each memory as a standalone sentence starting with "User ...".
6. Use a stable snake_case memory_key, e.g. "preferred_language" or "answer_style".
7. Set importance from 1 (trivial) to 5 (essential).
8. Include 1-3 short quotes from the user's own words as evidence.
"""

def _format_messages(messages: list[BaseMessage]) -> str:
  """Turn langchain messages into a readable transcript for the prompt."""
  lines: list[str] = []

  for message in messages:
    if message.type == "human":
      role = "User"
    elif message.type == "ai":
      role = "Assistant"
    else:
      role = "System"

    content = message.content
    if not isinstance(content, str):
      content = str(content)

    lines.append(f"{role}: {content}")

  return "\n".join(lines)

@lru_cache(maxsize = 1)
def get_extraction_chain():
  """Build (and cache) prompt -> structured Groq chain."""
  llm = ChatGroq(
    model_name = settings.llm_model,
    groq_api_key=(
      settings.groq_api_key.get_secret_value()
      if settings.groq_api_key
      else None
    ),
    temperature=0,
    max_tokens = settings.llm_max_tokens,
    max_retries = settings.llm_max_retries,
    request_timeout = settings.llm_timeout_seconds,
  )
  structured_llm = llm.with_structured_output(MemoryExtraction)
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system",EXTRACTION_SYSTEM_PROMPT),
      ("human", "Conversation:\n{conversation}"),
    ]
  )
  return prompt | structured_llm

def extract_memories(
  messages: list[BaseMessage],
  *,
  max_recent_messages: int = 6,
) -> list[MemoryCandidate]:
  """Return durable memories found in the recent conversation."""
  recent = messages[-max_recent_messages:]

  if not recent:
    return []

  result = get_extraction_chain().invoke(
    {"conversation": _format_messages(recent)}
  )

  # Defensive fallback in case a provider returns a plain dict
  if isinstance(result, dict):
    result = MemoryExtraction.model_validate(result)

  return result.memories