from app.configuration import setting
from langchain.agents import create_agent


def create_teleco_agent(llm,tools):
    return create_agent(model=llm, tools=tools, system_prompt=setting.SYSTEM_PROMPT)