# Getting Started with Agentum

This guide will walk you through building your first real-world agentic workflow: a professional research assistant that can search the web and then write a structured report.

## Prerequisites

- Python 3.11+
- API keys for Google and Tavily Search, stored in a `.env` file (`GOOGLE_API_KEY="..."`, `TAVILY_API_KEY="..."`).

## Installation

```bash
pip install agentum langchain-google-genai tavily-python
```

## Your First Multi-Agent Workflow

Let's build a research pipeline that uses two specialist agents: a Researcher and a Writer.

### Step 1: Define State

The `State` defines the data that flows through your workflow. It's the "memory" that tasks share.

```python
from agentum import State

class ReportState(State):
    topic: str              # The initial input
    research_data: str = "" # The output of the Researcher
    report: str = ""        # The final output of the Writer
```

### Step 2: Create Your Agents

We'll create two agents, each with a specific role and the right tools for their job.

```python
import os
from dotenv import load_dotenv
from agentum import Agent, GoogleLLM, search_web_tavily

load_dotenv()

# The Researcher uses a real web search tool.
researcher = Agent(
    name="Researcher",
    system_prompt="You are an expert researcher. Find comprehensive, up-to-date information on a topic and synthesize it into a coherent summary.",
    llm=GoogleLLM(model="gemini-2.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[search_web_tavily]
)

# The Writer has no tools; its only job is to format text.
writer = Agent(
    name="Writer",
    system_prompt="You are a professional writer. Your job is to format research data into a well-structured markdown report with a title and key sections.",
    llm=GoogleLLM(model="gemini-2.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))
)
```

### Step 3: Build the Workflow

The `Workflow` orchestrates the agents and defines how data moves between them.

```python
from agentum import Workflow

workflow = Workflow(name="Research_Pipeline", state=ReportState)

# Task 1: The Researcher agent gathers information.
# The output is mapped to the 'research_data' field in our state.
workflow.add_task(
    name="conduct_research",
    agent=researcher,
    instructions="Perform a thorough web search for the topic: {topic}",
    output_mapping={"research_data": "output"}
)

# Task 2: The Writer agent takes the research and formats it.
# The '{research_data}' input comes directly from the state, filled by the previous task.
workflow.add_task(
    name="write_report",
    agent=writer,
    instructions="Format this research data into a report: {research_data}",
    output_mapping={"report": "output"}
)

# Define the control flow: research -> write -> end.
workflow.set_entry_point("conduct_research")
workflow.add_edge("conduct_research", "write_report")
workflow.add_edge("write_report", workflow.END)
```

### Step 4: Run the Workflow

Now, execute the pipeline and see the final result.

```python
if __name__ == "__main__":
    initial_state = {"topic": "The future of autonomous vehicles"}
    final_state = workflow.run(initial_state)

    print("\n--- FINAL REPORT ---")
    print(final_state["report"])
```

You've just built a multi-agent workflow that performs a real-world task!

## Next Steps

- [Explore LLM Providers](features/providers.md) - Swap models from Google, Anthropic, and OpenAI.
- [Add Vision & Speech](features/multi_modality.md) - Build agents that can see, hear, and speak.
- [Master Advanced RAG](features/advanced_rag.md) - Use rerankers for high-precision search.
