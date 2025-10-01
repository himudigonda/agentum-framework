# examples/01_hello_agentum.py
import os

from dotenv import load_dotenv
from pydantic import Field

from agentum import Agent, GoogleLLM, State, Workflow, tool

# Load API keys from .env file
load_dotenv()


# 1. Define the state model
class ResearchState(State):
    topic: str
    # LangGraph will automatically populate the output of the 'conduct_research'
    # task into a key with the same name. We define it here for type safety.
    conduct_research: str = Field(default="")
    save_to_file: str = Field(default="")


# 2. Define a tool
@tool
def save_summary(summary: str):
    """A mock tool to save the research summary."""
    print("\n--- TOOL EXECUTING ---")
    print(f"Saving summary: '{summary[:60]}...'")
    print("--- TOOL FINISHED ---\n")
    return f"Summary for topic '{summary.split()[0]}' saved successfully."


# 3. Configure a real LLM and define an agent
google_llm = GoogleLLM(
    api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"
)

researcher = Agent(
    name="Researcher",
    system_prompt="You are a helpful research assistant. Keep your answers concise and to the point (2-3 sentences).",
    llm=google_llm,
)

# 4. Define the workflow
research_workflow = Workflow(name="Basic_Research_Pipeline", state=ResearchState)

research_workflow.add_task(
    name="conduct_research",
    agent=researcher,
    instructions="Please research the topic: {topic}",
)
research_workflow.add_task(
    name="save_to_file",
    tool=save_summary,
    # The engine will map the output of the 'conduct_research' node
    # from the state to the 'summary' input of this tool.
    inputs={"summary": "{conduct_research}"},
)

research_workflow.set_entry_point("conduct_research")
research_workflow.add_edge("conduct_research", "save_to_file")
research_workflow.add_edge("save_to_file", research_workflow.END)

# 5. Run the workflow
if __name__ == "__main__":
    initial_state = {"topic": "The Future of AI"}

    # This will now execute a real LLM call!
    final_state = research_workflow.run(initial_state)

    print("\nFinal State:")
    print(final_state)
