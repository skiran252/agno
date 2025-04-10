"""Run `pip install openai exa_py duckduckgo-search yfinance pypdf sqlalchemy 'fastapi[standard]' youtube-transcript-api python-docx agno` to install dependencies."""

from datetime import datetime
from textwrap import dedent

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app
from agno.storage.sqlite import SqliteStorage
from agno.tools.reasoning import ReasoningTools

agent_storage_file: str = "tmp/agents.db"
image_agent_storage_file: str = "tmp/image_agent.db"

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

cot_agent = Agent(
    name="COT Agent",
    role="Answer basic questions",
    agent_id="cot-agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    storage=SqliteStorage(
        table_name="simple_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
    reasoning=True,
)

reasoning_model_agent = Agent(
    name="Reasoning Model Agent",
    role="Reasoning about Math",
    model=OpenAIChat(id="gpt-4o"),
    reasoning_model=OpenAIChat(id="o3-mini"),
    instructions=["You are a reasoning agent that can reason about math."],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)

reasoning_tool_agent = Agent(
    name="Reasoning Tool Agent",
    role="Answer basic questions",
    agent_id="reasoning-tool-agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    storage=SqliteStorage(
        table_name="simple_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
    tools=[ReasoningTools()],
)

native_model_agent = Agent(
    name="Native Model Agent",
    role="Answer basic questions",
    agent_id="native-model-agent",
    model=OpenAIChat(id="o3-mini"),
    storage=SqliteStorage(
        table_name="simple_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
)


app = Playground(
    agents=[
        cot_agent,
        reasoning_tool_agent,
        reasoning_model_agent,
        native_model_agent,
    ]
).get_app()

if __name__ == "__main__":
    serve_playground_app("reasoning_demo:app", reload=True)
