from app.configuration import setting
from langchain_groq import ChatGroq

def get_llm():
    return ChatGroq(model=setting.LLM_GROQ_MODEL, temperature=0)