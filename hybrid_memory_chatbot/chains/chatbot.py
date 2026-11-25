from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from db.chat_history import PostgresChatMessageHistory
from db.connection import SessionFactory, init_db

init_db()

llm = ChatGroq(
  model="openai/gpt-oss-20b",
  temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
  ]
)

chain = prompt | llm

_history_cache: dict[str, PostgresChatMessageHistory] = {}

def get_session_history(session_id: str) -> PostgresChatMessageHistory:
  if session_id not in _history_cache:
    user_id = session_id.removeprefix("user:")

    _history_cache[session_id] = PostgresChatMessageHistory(
      session_id=session_id,
      user_id=user_id,
      SessionFactory=SessionFactory,
    )

  return _history_cache[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

def ask(question: str, user_id: str) -> str:
    response = chatbot.invoke(
        {"question": question},
        config={"configurable": {"session_id": f"user:{user_id}"}},
    )

    content = response.content
    return content.strip() if isinstance(content, str) else str(content)

if __name__ == "__main__":
    print(ask("My name is Om.", "123"))
    print(ask("What is my name?", "123"))