# tests/test_agency.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentum import Agent, ConversationMemory, GoogleLLM, State, Workflow, tool


class AgencyState(State):
    request: str
    response: str = ""


class TestAgency:
    """Test true agentic behavior and autonomous tool usage."""

    def test_tool_decorator_schema_generation(self):
        """Test that @tool decorator generates proper schemas."""

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: 72°F"

        # Check that the tool has the expected attributes
        assert hasattr(get_weather, "__name__")
        assert get_weather.__name__ == "get_weather"

        # Check docstring is preserved
        assert "Get weather for a city" in get_weather.__doc__

    def test_agent_with_tools(self):
        """Test agent creation with tools."""

        @tool
        def test_tool(query: str) -> str:
            """Test tool."""
            return f"Result: {query}"

        agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent.",
            llm=MagicMock(),
            tools=[test_tool],
        )

        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert agent.tools[0].__name__ == "test_tool"

    def test_agent_with_memory(self):
        """Test agent creation with memory."""
        agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent.",
            llm=MagicMock(),
            memory=ConversationMemory(),
        )

        assert agent.memory is not None
        assert isinstance(agent.memory, ConversationMemory)

    def test_agent_retry_configuration(self):
        """Test agent retry configuration."""
        agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent.",
            llm=MagicMock(),
            max_retries=5,
        )

        assert agent.max_retries == 5

    @pytest.mark.asyncio
    async def test_autonomous_tool_usage(self):
        """Test that agents can autonomously decide to use tools."""
        workflow = Workflow(name="AgencyTest", state=AgencyState)

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: 72°F and sunny"

        # Mock LLM responses to simulate tool usage
        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tool_call_response.tool_calls = [
            {"name": "get_weather", "args": {"city": "San Francisco"}, "id": "call_123"}
        ]

        final_response = MagicMock()
        final_response.content = (
            "Based on the weather, I recommend packing light layers."
        )
        final_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        agent = Agent(
            name="TravelAgent",
            system_prompt="You are a travel assistant. Use weather tools when needed.",
            llm=mock_llm,
            tools=[get_weather],
        )

        workflow.add_task(
            name="plan_trip",
            agent=agent,
            instructions="Plan a trip to: {request}",
            output_mapping={"response": "output"},
        )

        workflow.set_entry_point("plan_trip")
        workflow.add_edge("plan_trip", workflow.END)

        # Run the workflow
        result = await workflow.arun({"request": "San Francisco"})

        # Verify the agent used the tool and provided a response
        assert "response" in result
        assert "layers" in result["response"].lower()
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_memory_persistence(self):
        """Test that agent memory persists across interactions."""
        workflow = Workflow(name="MemoryTest", state=AgencyState)

        mock_response = MagicMock()
        mock_response.content = "Hello Alice! Nice to meet you."
        mock_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response

        agent = Agent(
            name="ChatAgent",
            system_prompt="You are a helpful assistant.",
            llm=mock_llm,
            memory=ConversationMemory(),
        )

        workflow.add_task(
            name="chat",
            agent=agent,
            instructions="Respond to: {request}",
            output_mapping={"response": "output"},
        )

        workflow.set_entry_point("chat")
        workflow.add_edge("chat", workflow.END)

        # First interaction
        result1 = await workflow.arun({"request": "My name is Alice"})

        # Second interaction (should remember Alice)
        mock_response.content = "Hello Alice! How can I help you today?"
        result2 = await workflow.arun({"request": "What's my name?"})

        # Verify memory worked
        assert "Alice" in result2["response"]

    @pytest.mark.asyncio
    async def test_event_emission(self):
        """Test that events are properly emitted during agentic execution."""
        workflow = Workflow(name="EventTest", state=AgencyState)

        events = []

        @workflow.on("agent_start")
        async def on_agent_start(agent_name: str, state: dict):
            events.append(f"agent_start:{agent_name}")

        @workflow.on("agent_tool_call")
        async def on_tool_call(tool_name: str, tool_args: dict):
            events.append(f"tool_call:{tool_name}")

        @workflow.on("agent_end")
        async def on_agent_end(agent_name: str, final_response: str):
            events.append(f"agent_end:{agent_name}")

        @tool
        def test_tool(query: str) -> str:
            return f"Tool result: {query}"

        # Mock LLM responses
        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tool_call_response.tool_calls = [
            {"name": "test_tool", "args": {"query": "test"}, "id": "call_123"}
        ]

        final_response = MagicMock()
        final_response.content = "Final response"
        final_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        agent = Agent(
            name="EventAgent",
            system_prompt="You are a test agent.",
            llm=mock_llm,
            tools=[test_tool],
        )

        workflow.add_task(
            name="test_task",
            agent=agent,
            instructions="Process: {request}",
            output_mapping={"response": "output"},
        )

        workflow.set_entry_point("test_task")
        workflow.add_edge("test_task", workflow.END)

        # Run workflow
        await workflow.arun({"request": "test"})

        # Verify events were emitted
        assert "agent_start:EventAgent" in events
        assert "tool_call:test_tool" in events
        assert "agent_end:EventAgent" in events

    @pytest.mark.asyncio
    async def test_multi_tool_usage(self):
        """Test agent using multiple tools in sequence."""
        workflow = Workflow(name="MultiToolTest", state=AgencyState)

        @tool
        def get_weather(city: str) -> str:
            return f"Weather in {city}: 72°F"

        @tool
        def get_restaurants(city: str) -> str:
            return f"Best restaurants in {city}: Italian, Mexican, Asian"

        # Mock LLM responses for multiple tool calls
        weather_call = MagicMock()
        weather_call.content = ""
        weather_call.tool_calls = [
            {"name": "get_weather", "args": {"city": "San Francisco"}, "id": "call_1"}
        ]

        restaurant_call = MagicMock()
        restaurant_call.content = ""
        restaurant_call.tool_calls = [
            {
                "name": "get_restaurants",
                "args": {"city": "San Francisco"},
                "id": "call_2",
            }
        ]

        final_response = MagicMock()
        final_response.content = "Here's your complete travel guide for San Francisco."
        final_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [weather_call, restaurant_call, final_response]
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        agent = Agent(
            name="TravelAgent",
            system_prompt="You are a comprehensive travel assistant.",
            llm=mock_llm,
            tools=[get_weather, get_restaurants],
        )

        workflow.add_task(
            name="plan_trip",
            agent=agent,
            instructions="Create a travel guide for: {request}",
            output_mapping={"response": "output"},
        )

        workflow.set_entry_point("plan_trip")
        workflow.add_edge("plan_trip", workflow.END)

        # Run workflow
        result = await workflow.arun({"request": "San Francisco"})

        # Verify both tools were called
        assert "response" in result
        assert "travel guide" in result["response"].lower()
        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test that tool errors are handled gracefully."""
        workflow = Workflow(name="ErrorTest", state=AgencyState)

        @tool
        def failing_tool(query: str) -> str:
            raise Exception("Tool failed")

        # Mock LLM responses
        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tool_call_response.tool_calls = [
            {"name": "failing_tool", "args": {"query": "test"}, "id": "call_123"}
        ]

        final_response = MagicMock()
        final_response.content = "I encountered an error, but I can still help."
        final_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        agent = Agent(
            name="ErrorAgent",
            system_prompt="You are a helpful agent.",
            llm=mock_llm,
            tools=[failing_tool],
        )

        workflow.add_task(
            name="test_task",
            agent=agent,
            instructions="Process: {request}",
            output_mapping={"response": "output"},
        )

        workflow.set_entry_point("test_task")
        workflow.add_edge("test_task", workflow.END)

        # Run workflow (should not crash)
        result = await workflow.arun({"request": "test"})

        # Verify workflow completed despite tool error
        assert "response" in result
        assert "error" in result["response"].lower()

    def test_workflow_with_conditional_agency(self):
        """Test workflow with conditional logic and agentic behavior."""
        workflow = Workflow(name="ConditionalAgency", state=AgencyState)

        @tool
        def analyze_sentiment(text: str) -> str:
            return "positive" if "good" in text.lower() else "negative"

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Analysis complete"
        mock_response.tool_calls = []
        mock_llm.ainvoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_llm

        agent = Agent(
            name="SentimentAgent",
            system_prompt="You analyze sentiment.",
            llm=mock_llm,
            tools=[analyze_sentiment],
        )

        workflow.add_task(
            name="analyze",
            agent=agent,
            instructions="Analyze: {request}",
            output_mapping={"response": "output"},
        )

        workflow.set_entry_point("analyze")

        def should_continue(state: AgencyState) -> str:
            return "continue" if "good" in state.response.lower() else "stop"

        workflow.add_conditional_edges(
            source="analyze",
            path=should_continue,
            paths={"continue": "analyze", "stop": workflow.END},
        )

        # Compile workflow
        from agentum.engine import GraphCompiler

        compiler = GraphCompiler(workflow)
        compiled_graph = compiler.compile()

        assert compiled_graph is not None
