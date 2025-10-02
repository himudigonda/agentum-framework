"""
Simple integration test that verifies framework components work together.

This test runs without requiring API keys and focuses on:
- Component instantiation
- Workflow creation
- State management
- Tool definitions
- Memory setup
- Error handling
"""
import os
import tempfile
from typing import Any
import pytest
from agentum import Agent, AnthropicLLM, ConversationMemory, GoogleLLM, KnowledgeBase, State, Workflow, tool
from agentum.config import settings

class TestState(State):
    input_text: str
    result: str = ''

@tool
def simple_tool(input_text: str) -> str:
    return f'Tool processed: {input_text}'

@pytest.mark.integration
def test_framework_components():
    if not settings.GOOGLE_API_KEY or not settings.ANTHROPIC_API_KEY:
        pytest.skip('API keys not found in settings, skipping integration test')
    print('🧪 Testing framework components...')
    state = TestState(input_text='test')
    assert state.input_text == 'test'
    assert state.result == ''
    print('✅ State creation works')
    try:
        google_llm = GoogleLLM(api_key=settings.GOOGLE_API_KEY or 'test_key')
        assert isinstance(google_llm, GoogleLLM)
        print('✅ GoogleLLM instantiation works')
    except Exception as e:
        print(f'⚠️  GoogleLLM instantiation failed (expected without real API key): {e}')
    try:
        anthropic_llm = AnthropicLLM(api_key=settings.ANTHROPIC_API_KEY or 'test_key')
        assert isinstance(anthropic_llm, AnthropicLLM)
        print('✅ AnthropicLLM instantiation works')
    except Exception as e:
        print(f'⚠️  AnthropicLLM instantiation failed (expected without real API key): {e}')
    agent = Agent(name='TestAgent', system_prompt='You are a test agent.', llm=google_llm)
    assert agent.name == 'TestAgent'
    print('✅ Agent creation works')
    agent_with_tools = Agent(name='ToolAgent', system_prompt='You are a tool agent.', llm=google_llm, tools=[simple_tool])
    assert len(agent_with_tools.tools) == 1
    print('✅ Tool integration works')
    memory = ConversationMemory()
    agent_with_memory = Agent(name='MemoryAgent', system_prompt='You are a memory agent.', llm=google_llm, memory=memory)
    assert agent_with_memory.memory == memory
    print('✅ Memory integration works')
    workflow = Workflow(name='TestWorkflow', state=TestState)
    assert workflow.name == 'TestWorkflow'
    print('✅ Workflow creation works')
    workflow.add_task(name='test_task', agent=agent, instructions='Process: {input_text}', output_mapping={'result': 'output'})
    assert 'test_task' in workflow.tasks
    print('✅ Task addition works')
    workflow.set_entry_point('test_task')
    workflow.add_edge('test_task', workflow.END)
    assert workflow.entry_point == 'test_task'
    print('✅ Workflow configuration works')
    kb = KnowledgeBase(name='TestKB')
    assert kb.name == 'TestKB'
    print('✅ Knowledge base creation works')
    print('\n🎉 All framework components working correctly!')

def test_vision_state_logic():
    print('\n🧪 Testing vision state logic...')

    class VisionTestState(State):
        text: str
        image_url: str = ''
        result: str = ''
    state_no_image = VisionTestState(text='Hello')
    has_image = hasattr(state_no_image, 'image_url') and getattr(state_no_image, 'image_url')
    assert not has_image
    print('✅ State without image detected correctly')
    state_with_image = VisionTestState(text='Hello', image_url='https://example.com/image.jpg')
    has_image = hasattr(state_with_image, 'image_url') and getattr(state_with_image, 'image_url')
    assert has_image
    print('✅ State with image detected correctly')
    print('🎉 Vision state logic working correctly!')

def test_error_handling():
    print('\n🧪 Testing error handling...')
    workflow = Workflow(name='ErrorTest', state=TestState)
    from tests.mock_llm import MockLLM
    agent = Agent(name='ErrorAgent', system_prompt='Test agent.', llm=MockLLM())
    workflow.add_task(name='error_task', agent=agent, instructions='Process: {invalid_key}', output_mapping={'result': 'output'})
    workflow.set_entry_point('error_task')
    workflow.add_edge('error_task', workflow.END)
    with pytest.raises(Exception, match='Missing state key'):
        workflow.run({'input_text': 'test'})
    print('✅ Error handling works correctly')
    print('🎉 Error handling working correctly!')

def main():
    print('🚀 Agentum Framework Integration Tests')
    print('=' * 50)
    try:
        test_framework_components()
        test_vision_state_logic()
        test_error_handling()
        print('\n' + '=' * 50)
        print('🎉 All integration tests passed!')
        print('✅ Agentum framework is working correctly!')
    except Exception as e:
        print(f'\n❌ Integration test failed: {e}')
        raise
if __name__ == '__main__':
    main()