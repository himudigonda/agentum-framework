# tests/test_engine.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentum import Agent, GoogleLLM, State, Workflow, tool
from agentum.engine import GraphCompiler


class TestState(State):
    input: str
    output: str = ""


class TestEngine:
    """Test the GraphCompiler and engine functionality."""

    def test_graph_compiler_initialization(self):
        """Test that GraphCompiler initializes correctly."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)
        assert compiler.workflow == workflow

    def test_create_agent_node_without_tools(self):
        """Test agent node creation without tools."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        agent = Agent(
            name="TestAgent", system_prompt="You are a test agent.", llm=MagicMock()
        )

        task_details = {
            "agent": agent,
            "instructions": "Process: {input}",
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_agent_node("test_task", task_details)
        assert callable(node_func)

    def test_create_agent_node_with_tools(self):
        """Test agent node creation with tools."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        @tool
        def test_tool(query: str) -> str:
            return f"Result for {query}"

        agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent.",
            llm=MagicMock(),
            tools=[test_tool],
        )

        task_details = {
            "agent": agent,
            "instructions": "Process: {input}",
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_agent_node("test_task", task_details)
        assert callable(node_func)

    def test_create_tool_node(self):
        """Test tool node creation."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        def test_tool(input_text: str) -> str:
            return f"Processed: {input_text}"

        task_details = {
            "tool": test_tool,
            "inputs": {"input_text": "{input}"},
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_tool_node("test_tool", task_details)
        assert callable(node_func)

    @pytest.mark.asyncio
    async def test_agent_node_execution(self):
        """Test agent node execution."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response

        agent = Agent(
            name="TestAgent", system_prompt="You are a test agent.", llm=mock_llm
        )

        task_details = {
            "agent": agent,
            "instructions": "Process: {input}",
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_agent_node("test_task", task_details)

        # Test state
        state = TestState(input="test input")

        # Execute node
        result = await node_func(state)

        assert "output" in result
        assert result["output"] == "Test response"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_node_with_tool_calls(self):
        """Test agent node execution with tool calls."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        @tool
        def test_tool(query: str) -> str:
            return f"Tool result for {query}"

        # Mock LLM responses
        tool_call_response = MagicMock()
        tool_call_response.content = ""
        tool_call_response.tool_calls = [
            {"name": "test_tool", "args": {"query": "test query"}, "id": "call_123"}
        ]

        final_response = MagicMock()
        final_response.content = "Final response"
        final_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent.",
            llm=mock_llm,
            tools=[test_tool],
        )

        task_details = {
            "agent": agent,
            "instructions": "Process: {input}",
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_agent_node("test_task", task_details)

        # Test state
        state = TestState(input="test input")

        # Execute node
        result = await node_func(state)

        assert "output" in result
        assert result["output"] == "Final response"
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_node_execution(self):
        """Test tool node execution."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        def test_tool(input_text: str) -> str:
            return f"Processed: {input_text}"

        task_details = {
            "tool": test_tool,
            "inputs": {"input_text": "{input}"},
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_tool_node("test_tool", task_details)

        # Test state
        state = TestState(input="test input")

        # Execute node
        result = await node_func(state)

        assert "output" in result
        assert result["output"] == "Processed: test input"

    @pytest.mark.asyncio
    async def test_agent_retry_logic(self):
        """Test agent retry logic with failures."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            Exception("Network error"),
            Exception("Rate limit"),
            MagicMock(content="Success response", tool_calls=[]),
        ]

        agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent.",
            llm=mock_llm,
            max_retries=3,
        )

        task_details = {
            "agent": agent,
            "instructions": "Process: {input}",
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_agent_node("test_task", task_details)

        # Test state
        state = TestState(input="test input")

        # Execute node (should succeed after retries)
        result = await node_func(state)

        assert "output" in result
        assert result["output"] == "Success response"
        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_agent_max_retries_exceeded(self):
        """Test agent failure when max retries exceeded."""
        workflow = Workflow(name="TestWorkflow", state=TestState)
        compiler = GraphCompiler(workflow)

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("Persistent error")

        agent = Agent(
            name="TestAgent",
            system_prompt="You are a test agent.",
            llm=mock_llm,
            max_retries=2,
        )

        task_details = {
            "agent": agent,
            "instructions": "Process: {input}",
            "output_mapping": {"output": "output"},
        }

        node_func = compiler._create_agent_node("test_task", task_details)

        # Test state
        state = TestState(input="test input")

        # Execute node (should fail after retries)
        with pytest.raises(Exception, match="Persistent error"):
            await node_func(state)

        assert mock_llm.ainvoke.call_count == 2

    def test_compile_workflow(self):
        """Test workflow compilation."""
        workflow = Workflow(name="TestWorkflow", state=TestState)

        # Add a task
        workflow.add_task(
            name="test_task",
            agent=Agent(name="TestAgent", system_prompt="Test", llm=MagicMock()),
            instructions="Process: {input}",
            output_mapping={"output": "output"},
        )

        workflow.set_entry_point("test_task")
        workflow.add_edge("test_task", workflow.END)

        compiler = GraphCompiler(workflow)
        compiled_graph = compiler.compile()

        assert compiled_graph is not None

    def test_compile_workflow_with_conditional_edges(self):
        """Test workflow compilation with conditional edges."""
        workflow = Workflow(name="TestWorkflow", state=TestState)

        # Add tasks
        workflow.add_task(
            name="task1",
            agent=Agent(name="TestAgent", system_prompt="Test", llm=MagicMock()),
            instructions="Process: {input}",
            output_mapping={"output": "output"},
        )

        workflow.add_task(
            name="task2",
            agent=Agent(name="TestAgent2", system_prompt="Test2", llm=MagicMock()),
            instructions="Process: {input}",
            output_mapping={"output": "output"},
        )

        workflow.set_entry_point("task1")

        def should_continue(state: TestState) -> str:
            return "continue" if len(state.input) > 5 else "stop"

        workflow.add_conditional_edges(
            source="task1",
            path=should_continue,
            paths={"continue": "task2", "stop": workflow.END},
        )

        compiler = GraphCompiler(workflow)
        compiled_graph = compiler.compile()

        assert compiled_graph is not None
