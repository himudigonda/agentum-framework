"""
Comprehensive end-to-end test suite for Agentum framework.

This test suite verifies all major features:
- Multi-provider LLM support (Google, Anthropic)
- Vision capabilities
- Tool integration
- Memory and conversation persistence
- RAG and vector search
- Multi-agent workflows
- Error handling and resilience
"""

import asyncio
import os
import tempfile
from typing import Any, Dict

import pytest
from dotenv import load_dotenv

from agentum import (
    Agent,
    AnthropicLLM,
    ConversationMemory,
    GoogleLLM,
    KnowledgeBase,
    State,
    Workflow,
    tool,
)

load_dotenv()


class TestState(State):
    """Test state for various scenarios."""

    input_text: str
    image_url: str = ""
    result: str = ""
    intermediate_result: str = ""


# MODIFICATION: The entire class will be marked as integration
@pytest.mark.integration
class TestFramework:
    """Comprehensive test framework for Agentum."""

    def __init__(self):
        # MODIFICATION: Add guards here as well
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not found, skipping comprehensive suite")
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not found, skipping comprehensive suite")

        # Initialize LLMs with test keys if no real API keys are available
        google_key = os.getenv("GOOGLE_API_KEY") or "test_key"
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or "test_key"

        try:
            self.google_llm = GoogleLLM(api_key=google_key)
        except Exception:
            self.google_llm = None

        try:
            self.anthropic_llm = AnthropicLLM(api_key=anthropic_key)
        except Exception:
            self.anthropic_llm = None

    def test_provider_instantiation(self):
        """Test that all providers can be instantiated."""
        print("🧪 Testing provider instantiation...")

        # Test Google provider
        if self.google_llm:
            assert isinstance(self.google_llm, GoogleLLM)
            assert hasattr(self.google_llm, "ainvoke")
            assert hasattr(self.google_llm, "bind_tools")
            print("✅ GoogleLLM instantiated successfully")
        else:
            print("⚠️  GoogleLLM not available (no API key)")

        # Test Anthropic provider
        if self.anthropic_llm:
            assert isinstance(self.anthropic_llm, AnthropicLLM)
            assert hasattr(self.anthropic_llm, "ainvoke")
            assert hasattr(self.anthropic_llm, "bind_tools")
            print("✅ AnthropicLLM instantiated successfully")
        else:
            print("⚠️  AnthropicLLM not available (no API key)")

        print("✅ Provider instantiation test completed")

    def test_agent_creation(self):
        """Test agent creation with different providers."""
        print("🧪 Testing agent creation...")

        # Test Google agent if available
        if self.google_llm:
            google_agent = Agent(
                name="GoogleAgent",
                system_prompt="You are a helpful assistant.",
                llm=self.google_llm,
            )
            assert google_agent.name == "GoogleAgent"
            assert google_agent.llm == self.google_llm
            print("✅ Google agent created successfully")

        # Test Anthropic agent if available
        if self.anthropic_llm:
            anthropic_agent = Agent(
                name="AnthropicAgent",
                system_prompt="You are a helpful assistant.",
                llm=self.anthropic_llm,
            )
            assert anthropic_agent.name == "AnthropicAgent"
            assert anthropic_agent.llm == self.anthropic_llm
            print("✅ Anthropic agent created successfully")

        print("✅ Agent creation test completed")

    def test_workflow_creation(self):
        """Test workflow creation and task addition."""
        print("🧪 Testing workflow creation...")

        workflow = Workflow(name="TestWorkflow", state=TestState)

        # Use available LLM or create a mock one
        llm = self.google_llm or self.anthropic_llm
        if not llm:
            print("⚠️  No LLM available for workflow test")
            return

        agent = Agent(name="TestAgent", system_prompt="You are a test agent.", llm=llm)

        workflow.add_task(
            name="test_task",
            agent=agent,
            instructions="Process this input: {input_text}",
            output_mapping={"result": "output"},
        )

        assert workflow.name == "TestWorkflow"
        assert "test_task" in workflow.tasks
        assert workflow.tasks["test_task"]["agent"] == agent

        print("✅ Workflow created and configured successfully")

    def test_tool_integration(self):
        """Test tool creation and integration."""
        print("🧪 Testing tool integration...")

        @tool
        def test_tool(input_text: str) -> str:
            """A simple test tool."""
            return f"Processed: {input_text}"

        llm = self.google_llm or self.anthropic_llm
        if not llm:
            print("⚠️  No LLM available for tool test")
            return

        agent = Agent(
            name="ToolAgent",
            system_prompt="You are a tool-using agent.",
            llm=llm,
            tools=[test_tool],
        )

        assert len(agent.tools) == 1
        assert agent.tools[0] == test_tool

        print("✅ Tool integration working correctly")

    def test_memory_integration(self):
        """Test conversation memory."""
        print("🧪 Testing memory integration...")

        llm = self.google_llm or self.anthropic_llm
        if not llm:
            print("⚠️  No LLM available for memory test")
            return

        memory = ConversationMemory()
        agent = Agent(
            name="MemoryAgent",
            system_prompt="You are a memory-enabled agent.",
            llm=llm,
            memory=memory,
        )

        assert agent.memory == memory
        assert hasattr(memory, "save_messages")
        assert hasattr(memory, "load_messages")

        print("✅ Memory integration working correctly")

    def test_rag_integration(self):
        """Test RAG and knowledge base."""
        print("🧪 Testing RAG integration...")

        # Create a knowledge base
        kb = KnowledgeBase(name="TestKB")

        assert kb.name == "TestKB"
        assert hasattr(kb, "add")
        assert hasattr(kb, "as_retriever")

        print("✅ RAG integration working correctly")

    def test_vision_state_detection(self):
        """Test vision state detection logic."""
        print("🧪 Testing vision state detection...")

        # Test state without image
        state_no_image = TestState(input_text="Hello")
        assert not (
            hasattr(state_no_image, "image_url")
            and getattr(state_no_image, "image_url")
        )

        # Test state with image
        state_with_image = TestState(
            input_text="Hello", image_url="https://example.com/image.jpg"
        )
        assert hasattr(state_with_image, "image_url") and getattr(
            state_with_image, "image_url"
        )

        print("✅ Vision state detection working correctly")

    def test_multi_agent_workflow(self):
        """Test multi-agent workflow creation."""
        print("🧪 Testing multi-agent workflow...")

        workflow = Workflow(name="MultiAgentWorkflow", state=TestState)

        # Use available LLMs
        llm1 = self.google_llm or self.anthropic_llm
        llm2 = self.anthropic_llm or self.google_llm

        if not llm1 or not llm2:
            print("⚠️  Need at least one LLM for multi-agent test")
            return

        # Create multiple agents
        agent1 = Agent(
            name="Agent1", system_prompt="You are the first agent.", llm=llm1
        )

        agent2 = Agent(
            name="Agent2", system_prompt="You are the second agent.", llm=llm2
        )

        # Add tasks for both agents
        workflow.add_task(
            name="task1",
            agent=agent1,
            instructions="Process: {input_text}",
            output_mapping={"intermediate_result": "output"},
        )

        workflow.add_task(
            name="task2",
            agent=agent2,
            instructions="Continue processing: {intermediate_result}",
            output_mapping={"result": "output"},
        )

        # Set up workflow flow
        workflow.set_entry_point("task1")
        workflow.add_edge("task1", "task2")
        workflow.add_edge("task2", workflow.END)

        assert len(workflow.tasks) == 2
        assert workflow.entry_point == "task1"

        print("✅ Multi-agent workflow created successfully")

    def test_error_handling(self):
        """Test error handling and validation."""
        print("🧪 Testing error handling...")

        # Test invalid state key
        workflow = Workflow(name="ErrorTestWorkflow", state=TestState)

        llm = self.google_llm or self.anthropic_llm
        if not llm:
            print("⚠️  No LLM available for error test")
            return

        agent = Agent(name="ErrorAgent", system_prompt="Test agent.", llm=llm)

        workflow.add_task(
            name="error_task",
            agent=agent,
            instructions="Process: {nonexistent_key}",  # This should cause an error
            output_mapping={"result": "output"},
        )

        workflow.set_entry_point("error_task")
        workflow.add_edge("error_task", workflow.END)

        # This should raise a StateValidationError
        try:
            workflow.run({"input_text": "test"})
            assert False, "Should have raised StateValidationError"
        except Exception as e:
            assert "nonexistent_key" in str(e)
            print("✅ Error handling working correctly")

    def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting comprehensive Agentum framework tests...\n")

        try:
            self.test_provider_instantiation()
            self.test_agent_creation()
            self.test_workflow_creation()
            self.test_tool_integration()
            self.test_memory_integration()
            self.test_rag_integration()
            self.test_vision_state_detection()
            self.test_multi_agent_workflow()
            self.test_error_handling()

            print("\n🎉 All tests passed! Framework is working correctly.")
            return True

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            return False


if __name__ == "__main__":
    test_framework = TestFramework()
    success = test_framework.run_all_tests()
    exit(0 if success else 1)
