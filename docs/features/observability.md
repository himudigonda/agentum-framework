# Observability

Agentum's event system provides comprehensive observability into workflow execution, enabling powerful debugging and monitoring capabilities.

## Event System Overview

The event system is built around a simple decorator pattern that lets you hook into any part of the workflow lifecycle:

```python
@workflow.on("agent_tool_call")
async def on_tool_call(tool_name: str, tool_args: dict):
    print(f"Agent called {tool_name} with {tool_args}")

@workflow.on("agent_end")
async def on_agent_end(agent_name: str, final_response: str):
    print(f"{agent_name} finished: {len(final_response)} chars")
```

## Available Events

### Workflow Events
- `workflow_start` - Workflow begins execution
- `workflow_finish` - Workflow completes

### Task Events
- `task_start` - Task begins execution
- `task_finish` - Task completes

### Agent Events
- `agent_start` - Agent begins processing
- `agent_llm_start` - LLM call begins
- `agent_llm_end` - LLM call completes
- `agent_tool_call` - Agent calls a tool
- `agent_tool_result` - Tool returns result
- `agent_end` - Agent completes processing

## Event Listeners

### Basic Event Handling

```python
from agentum import Workflow

workflow = Workflow(name="MyWorkflow", state=MyState)

@workflow.on("workflow_start")
async def on_start(workflow_name: str, state: dict):
    print(f"Starting {workflow_name}")

@workflow.on("agent_tool_call")
async def on_tool_call(tool_name: str, tool_args: dict):
    print(f"Tool called: {tool_name}")
    print(f"Arguments: {tool_args}")

@workflow.on("agent_tool_result")
async def on_tool_result(tool_name: str, result: str):
    print(f"Tool {tool_name} returned: {result}")
```

### Advanced Event Handling

```python
import json
from datetime import datetime

@workflow.on("agent_llm_start")
async def log_llm_request(messages: list):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] LLM Request:")
    for msg in messages:
        print(f"  {msg.type}: {msg.content[:100]}...")

@workflow.on("agent_llm_end")
async def log_llm_response(response):
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] LLM Response:")
    print(f"  Content: {response.content[:100]}...")
    if response.tool_calls:
        print(f"  Tool Calls: {len(response.tool_calls)}")
```

## Debugging Workflows

### Real-time Monitoring

```python
@workflow.on("agent_start")
async def monitor_agent(agent_name: str, state: dict):
    print(f"🤖 {agent_name} starting with state keys: {list(state.keys())}")

@workflow.on("agent_tool_call")
async def monitor_tools(tool_name: str, tool_args: dict):
    print(f"🔧 Tool: {tool_name}")
    print(f"   Args: {json.dumps(tool_args, indent=2)}")

@workflow.on("agent_tool_result")
async def monitor_results(tool_name: str, result: str):
    print(f"📊 {tool_name} result: {result[:200]}...")
```

### Performance Monitoring

```python
import time
from collections import defaultdict

execution_times = defaultdict(list)

@workflow.on("task_start")
async def start_timer(task_name: str, state: dict):
    state['_start_time'] = time.time()

@workflow.on("task_finish")
async def end_timer(task_name: str, state_update: dict):
    if '_start_time' in state_update:
        duration = time.time() - state_update['_start_time']
        execution_times[task_name].append(duration)
        print(f"⏱️  {task_name} took {duration:.2f}s")
```

## Logging and Monitoring

### Structured Logging

```python
import logging
import json

logger = logging.getLogger("agentum")

@workflow.on("agent_tool_call")
async def log_tool_call(tool_name: str, tool_args: dict):
    logger.info(json.dumps({
        "event": "tool_call",
        "tool": tool_name,
        "args": tool_args,
        "timestamp": datetime.now().isoformat()
    }))

@workflow.on("agent_tool_result")
async def log_tool_result(tool_name: str, result: str):
    logger.info(json.dumps({
        "event": "tool_result",
        "tool": tool_name,
        "result_length": len(result),
        "timestamp": datetime.now().isoformat()
    }))
```

### Metrics Collection

```python
from collections import Counter

tool_usage = Counter()
error_count = 0

@workflow.on("agent_tool_call")
async def count_tool_usage(tool_name: str, tool_args: dict):
    tool_usage[tool_name] += 1

@workflow.on("agent_end")
async def check_for_errors(agent_name: str, final_response: str):
    if "error" in final_response.lower():
        global error_count
        error_count += 1
        print(f"⚠️  Error detected in {agent_name}")

# Print metrics after workflow completion
@workflow.on("workflow_finish")
async def print_metrics(workflow_name: str, state: dict):
    print(f"\n📊 Metrics for {workflow_name}:")
    print(f"Tool Usage: {dict(tool_usage)}")
    print(f"Errors: {error_count}")
```

## Testing with Events

### Event-Based Testing

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_workflow_events():
    workflow = Workflow(name="TestWorkflow", state=TestState)
    
    # Mock event listeners
    tool_call_events = []
    
    @workflow.on("agent_tool_call")
    async def capture_tool_call(tool_name: str, tool_args: dict):
        tool_call_events.append((tool_name, tool_args))
    
    # Run workflow
    await workflow.arun({"input": "test"})
    
    # Assert events were emitted
    assert len(tool_call_events) > 0
    assert tool_call_events[0][0] == "expected_tool"
```

### Integration Testing

```python
async def test_workflow_integration():
    workflow = Workflow(name="IntegrationTest", state=TestState)
    
    events = []
    
    @workflow.on("workflow_start")
    async def on_start(**kwargs):
        events.append("start")
    
    @workflow.on("agent_tool_call")
    async def on_tool_call(**kwargs):
        events.append("tool_call")
    
    @workflow.on("workflow_finish")
    async def on_finish(**kwargs):
        events.append("finish")
    
    await workflow.arun({"input": "test"})
    
    # Verify event sequence
    assert events == ["start", "tool_call", "finish"]
```

## Best Practices

1. **Use Events for Debugging**: Add event listeners during development
2. **Monitor Performance**: Track execution times and resource usage
3. **Log Structured Data**: Use JSON for machine-readable logs
4. **Test Event Sequences**: Verify expected event flows
5. **Clean Up Listeners**: Remove debug listeners in production

## Advanced Patterns

### Event Aggregation

```python
class WorkflowMonitor:
    def __init__(self):
        self.events = []
        self.start_time = None
    
    @workflow.on("workflow_start")
    async def on_start(self, **kwargs):
        self.start_time = time.time()
        self.events.append(("start", kwargs))
    
    @workflow.on("workflow_finish")
    async def on_finish(self, **kwargs):
        duration = time.time() - self.start_time
        self.events.append(("finish", kwargs))
        print(f"Workflow completed in {duration:.2f}s")
```

### Conditional Event Handling

```python
@workflow.on("agent_tool_call")
async def conditional_logging(tool_name: str, tool_args: dict):
    if tool_name == "expensive_api_call":
        print(f"💰 Expensive API call: {tool_args}")
    elif tool_name == "database_query":
        print(f"🗄️  Database query: {tool_args}")
    else:
        print(f"🔧 Tool call: {tool_name}")
```

## Next Steps

- [Memory](memory.md) - Enable stateful conversations
- [Testing](testing.md) - Build comprehensive test suites
- [Streaming](streaming.md) - Real-time workflow execution
- [Persistence](persistence.md) - State checkpointing
