from agno.agent import Agent
from agno.memory.agent import AgentMemory
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.models.openai import OpenAIChat
from agno.playground import Playground, serve_playground_app
from agno.storage.agent.sqlite import SqliteAgentStorage

agent_storage_file: str = "tmp/agents.db"

basic_agent = Agent(
    name="Basic Agent",
    model=OpenAIChat(id="gpt-4o"),
    memory=AgentMemory(
        db=SqliteMemoryDb(
            table_name="agent_memory",
            db_file="tmp/agent_memory.db",
        ),
        # Create and store personalized memories for this user
        create_user_memories=True,
        # Update memories for the user after each run
        update_user_memories_after_run=True,
        # Create and store session summaries
        create_session_summary=True,
        # Update session summaries after each run
        update_session_summary_after_run=True,
    ),
    storage=SqliteAgentStorage(
        table_name="basic_agent", db_file=agent_storage_file, auto_upgrade_schema=True
    ),
    add_history_to_messages=True,
    num_history_responses=3,
    add_datetime_to_instructions=True,
    markdown=True,
)

app = Playground(
    agents=[
        basic_agent,
    ]
).get_app()

if __name__ == "__main__":
    serve_playground_app("basic:app", reload=True)
