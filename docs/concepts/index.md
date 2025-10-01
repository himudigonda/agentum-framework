# Agentum Concepts

This section covers the core concepts that make up the Agentum framework.

## Table of Contents

- [Agent](agent.md) - AI agents with tools and memory
- [Workflow](workflow.md) - Declarative workflow orchestration
- [State](state.md) - Type-safe state management
- [Tool](tool.md) - Function-based tool system
- [Events](events.md) - Observability and debugging

## Core Philosophy

Agentum is built on three key principles:

1. **Explicit is Better than Implicit**: Clear, declarative APIs that make intent obvious
2. **Type Safety**: Pydantic-based state management prevents runtime errors
3. **Observability First**: Built-in event system for debugging and monitoring

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Agent       │    │    Workflow     │    │      State      │
│                 │    │                 │    │                 │
│ • LLM Provider  │───▶│ • Task Graph    │───▶│ • Type Safety   │
│ • Tools         │    │ • Event System  │    │ • Validation    │
│ • Memory        │    │ • Persistence   │    │ • Serialization │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Tool       │    │     Engine      │    │     Events      │
│                 │    │                 │    │                 │
│ • Auto Schema   │    │ • Compilation   │    │ • Lifecycle     │
│ • Validation    │    │ • Execution     │    │ • Debugging     │
│ • Error Handling│    │ • Retry Logic   │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Concepts

### Agents
AI entities that can reason, use tools, and maintain memory. They are the core intelligence units of your workflows.

### Workflows
Declarative graphs that define how agents and tools interact. They provide structure and orchestration.

### State
Type-safe data containers that flow through workflows. They ensure data integrity and provide clear interfaces.

### Tools
Functions that agents can call to interact with external systems. They're automatically introspected and validated.

### Events
Observability hooks that let you monitor and debug workflow execution in real-time.

## Design Patterns

### Agentic Loop
The core pattern where agents think, act (use tools), observe results, and think again:

```
Think → Act → Observe → Think → Act → Observe → ...
```

### Event-Driven Architecture
Everything in Agentum emits events, enabling comprehensive observability:

```python
@workflow.on("agent_tool_call")
async def on_tool_call(tool_name: str, tool_args: dict):
    print(f"Agent called {tool_name}")

@workflow.on("workflow_finish")
async def on_finish(workflow_name: str, state: dict):
    print(f"Workflow {workflow_name} completed")
```

### Declarative Configuration
Workflows are defined declaratively, making them easy to understand and modify:

```python
workflow.add_task(
    name="research",
    agent=researcher,
    instructions="Research: {topic}",
    output_mapping={"results": "output"}
)
```

## Best Practices

1. **Start Simple**: Begin with basic workflows and add complexity gradually
2. **Use Type Safety**: Define clear state models with Pydantic
3. **Enable Observability**: Add event listeners for debugging
4. **Test Thoroughly**: Use the testing framework to ensure quality
5. **Handle Errors**: Implement proper error handling and retries

## Next Steps

- [Agent Guide](agent.md) - Learn about building intelligent agents
- [Workflow Guide](workflow.md) - Master workflow orchestration
- [State Management](state.md) - Understand type-safe state handling
- [Tool System](tool.md) - Create powerful tools for agents
- [Event System](events.md) - Add observability to your workflows
