# Getting Started with Agentum

This guide will walk you through building your first agentic workflow with Agentum.

## Prerequisites

- Python 3.11+
- A Google API key (for the examples)

## Installation

```bash
pip install agentum
```

## Your First Workflow

Let's build a simple research assistant that can search the web and summarize findings.

### Step 1: Define a Tool

```python
from agentum import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    # In a real implementation, this would call a search API
    return f"Search results for '{query}': Found 5 relevant articles about the topic."

@tool
def summarize_text(text: str) -> str:
    """Summarize a long text into key points."""
    return f"Summary: {text[:100]}..."
```

### Step 2: Define State

```python
from agentum import State

class ResearchState(State):
    topic: str
    search_results: str = ""
    summary: str = ""
```

### Step 3: Create an Agent

```python
from agentum import Agent, GoogleLLM
import os

researcher = Agent(
    name="Researcher",
    system_prompt="You are a research assistant. Use available tools to gather and summarize information.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[search_web, summarize_text]
)
```

### Step 4: Build the Workflow

```python
from agentum import Workflow

workflow = Workflow(name="Research_Pipeline", state=ResearchState)

# Add tasks
workflow.add_task(
    name="search",
    agent=researcher,
    instructions="Search for information about: {topic}",
    output_mapping={"search_results": "output"}
)

workflow.add_task(
    name="summarize", 
    agent=researcher,
    instructions="Summarize these search results: {search_results}",
    output_mapping={"summary": "output"}
)

# Set up the flow
workflow.set_entry_point("search")
workflow.add_edge("search", "summarize")
workflow.add_edge("summarize", workflow.END)
```

### Step 5: Run the Workflow

```python
# Execute the workflow
result = workflow.run({"topic": "artificial intelligence"})

print("Search Results:", result["search_results"])
print("Summary:", result["summary"])
```

## Understanding the Output

When you run this workflow, you'll see:

1. **Tool Binding**: The agent learns about available tools
2. **Autonomous Tool Usage**: The agent decides to call `search_web`
3. **Tool Execution**: The search tool runs with the topic
4. **State Updates**: Results flow through the workflow
5. **Final Summary**: The agent summarizes the findings

## Next Steps

- [Learn about Agents](concepts/agent.md)
- [Explore Workflows](concepts/workflow.md)
- [Add Observability](features/observability.md)
- [Enable Memory](features/memory.md)
- [Build Tests](features/testing.md)

## Common Patterns

### Conditional Logic

```python
def should_continue(state: ResearchState) -> str:
    if len(state.search_results) > 100:
        return "continue"
    return "stop"

workflow.add_conditional_edges(
    source="search",
    path=should_continue,
    paths={"continue": "summarize", "stop": workflow.END}
)
```

### Event Handling

```python
@workflow.on("agent_tool_call")
async def on_tool_call(tool_name: str, tool_args: dict):
    print(f"Agent called {tool_name} with {tool_args}")

@workflow.on("agent_end")
async def on_agent_end(agent_name: str, final_response: str):
    print(f"{agent_name} finished: {len(final_response)} chars")
```

### Memory-Enabled Agents

```python
from agentum import ConversationMemory

agent = Agent(
    name="ChatBot",
    system_prompt="You are a helpful assistant.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    memory=ConversationMemory()  # Enable memory
)
```

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure your Google API key is set in environment variables
2. **Tool Not Found**: Check that tools are properly decorated with `@tool`
3. **State Key Error**: Verify output mappings match your state fields

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Ready for More?

You've built your first agentic workflow! Now explore the advanced features:

- [Observability](features/observability.md) - Monitor and debug your workflows
- [Memory](features/memory.md) - Build stateful conversations
- [Testing](features/testing.md) - Ensure quality with automated testing
- [Streaming](features/streaming.md) - Real-time workflow execution
