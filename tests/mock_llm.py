"""
Mock LLM classes for testing that properly inherit from BaseLLM.
"""
from unittest.mock import AsyncMock, MagicMock
from typing import Any, List

from langchain_core.messages import BaseMessage

from agentum.providers.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM that properly inherits from BaseLLM for testing."""
    
    def __init__(self, responses=None, **kwargs):
        super().__init__()
        self.responses = responses or []
        self.response_index = 0
        self.ainvoke_mock = AsyncMock()
        self.bind_tools_mock = MagicMock()
        
    async def ainvoke(self, messages: List[BaseMessage]) -> Any:
        """Mock ainvoke method."""
        if self.responses:
            if self.response_index < len(self.responses):
                response = self.responses[self.response_index]
                self.response_index += 1
                return response
            else:
                # Return the last response if we've exhausted the list
                return self.responses[-1]
        
        # Default mock response
        mock_response = MagicMock()
        mock_response.content = "Mock response"
        mock_response.tool_calls = []
        return mock_response
    
    def bind_tools(self, tools: List[Any]) -> "MockLLM":
        """Mock bind_tools method."""
        return self


class MockAsyncLLM(BaseLLM):
    """Mock async LLM for testing async workflows."""
    
    def __init__(self, side_effect=None, **kwargs):
        super().__init__()
        self.ainvoke_mock = AsyncMock(side_effect=side_effect)
        self.bind_tools_mock = MagicMock()
        
    async def ainvoke(self, messages: List[BaseMessage]) -> Any:
        """Mock ainvoke method."""
        return await self.ainvoke_mock(messages)
    
    def bind_tools(self, tools: List[Any]) -> "MockAsyncLLM":
        """Mock bind_tools method."""
        self.bind_tools_mock.return_value = self
        return self.bind_tools_mock(tools)
    
    # Expose mock methods for assertions
    @property
    def ainvoke(self):
        """Expose the mock ainvoke for assertions."""
        return self.ainvoke_mock
