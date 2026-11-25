from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.chat_history import (
    InMemoryChatMessageHistory
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from dotenv import load_dotenv

load_dotenv()


# 1. Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# 2. Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


# 3. Create chain
chain = prompt | llm


# 4. Store conversation histories
store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# 5. Attach memory to chain
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)


# 6. First conversation
response1 = chatbot.invoke(
    {"question": "My name is Om."},
    config={"configurable": {"session_id": "user_123"}}
)

print(response1.content)


# 7. Second conversation
response2 = chatbot.invoke(
    {"question": "What is my name?"},
    config={"configurable": {"session_id": "user_123"}}
)

print(response2.content)