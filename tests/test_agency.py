import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from agentum import Agent, ConversationMemory, GoogleLLM, State, Workflow, tool
from tests.mock_llm import MockLLM, MockAsyncLLM

class AgencyState(State):
    request: str
    response: str = ''

class TestAgency:

    def test_tool_decorator_schema_generation(self):

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city"""
            return f'Weather in {city}: 72°F'
        assert hasattr(get_weather, '__name__')
        assert get_weather.__name__ == 'get_weather'
        assert 'Get weather for a city' in get_weather.__doc__

    def test_agent_with_tools(self):

        @tool
        def test_tool(query: str) -> str:
            return f'Result: {query}'
        agent = Agent(name='TestAgent', system_prompt='You are a test agent.', llm=MockLLM(), tools=[test_tool])
        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert agent.tools[0].__name__ == 'test_tool'

    def test_agent_with_memory(self):
        agent = Agent(name='TestAgent', system_prompt='You are a test agent.', llm=MockLLM(), memory=ConversationMemory())
        assert agent.memory is not None
        assert isinstance(agent.memory, ConversationMemory)

    def test_agent_retry_configuration(self):
        agent = Agent(name='TestAgent', system_prompt='You are a test agent.', llm=MockLLM(), max_retries=5)
        assert agent.max_retries == 5

    @pytest.mark.asyncio
    async def test_autonomous_tool_usage(self):
        workflow = Workflow(name='AgencyTest', state=AgencyState)

        @tool
        def get_weather(city: str) -> str:
            return f'Weather in {city}: 72°F and sunny'
        tool_call_response = MagicMock()
        tool_call_response.content = ''
        tool_call_response.tool_calls = [{'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_123'}]
        final_response = MagicMock()
        final_response.content = 'Based on the weather, I recommend packing light layers.'
        final_response.tool_calls = []
        mock_llm = MockAsyncLLM()
        mock_llm.ainvoke_mock.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools_mock.return_value = mock_llm
        agent = Agent(name='TravelAgent', system_prompt='You are a travel assistant. Use weather tools when needed.', llm=mock_llm, tools=[get_weather])
        workflow.add_task(name='plan_trip', agent=agent, instructions='Plan a trip to: {request}', output_mapping={'response': 'output'})
        workflow.set_entry_point('plan_trip')
        workflow.add_edge('plan_trip', workflow.END)
        result = await workflow.arun({'request': 'San Francisco'})
        assert 'response' in result
        assert 'layers' in result['response'].lower()
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_memory_persistence(self):
        workflow = Workflow(name='MemoryTest', state=AgencyState)
        mock_response = MagicMock()
        mock_response.content = 'Hello Alice! Nice to meet you.'
        mock_response.tool_calls = []
        mock_llm = MockAsyncLLM()
        mock_llm.ainvoke_mock.return_value = mock_response
        agent = Agent(name='ChatAgent', system_prompt='You are a helpful assistant.', llm=mock_llm, memory=ConversationMemory())
        workflow.add_task(name='chat', agent=agent, instructions='Respond to: {request}', output_mapping={'response': 'output'})
        workflow.set_entry_point('chat')
        workflow.add_edge('chat', workflow.END)
        result1 = await workflow.arun({'request': 'My name is Alice'})
        mock_response.content = 'Hello Alice! How can I help you today?'
        result2 = await workflow.arun({'request': "What's my name?"})
        assert 'Alice' in result2['response']

    @pytest.mark.asyncio
    async def test_event_emission(self):
        workflow = Workflow(name='EventTest', state=AgencyState)
        events = []

        @workflow.on('agent_start')
        async def on_agent_start(agent_name: str, state: dict):
            events.append(f'agent_start:{agent_name}')

        @workflow.on('agent_tool_call')
        async def on_tool_call(tool_name: str, tool_args: dict):
            events.append(f'tool_call:{tool_name}')

        @workflow.on('agent_end')
        async def on_agent_end(agent_name: str, final_response: str):
            events.append(f'agent_end:{agent_name}')

        @tool
        def test_tool(query: str) -> str:
            return f'Tool result: {query}'
        tool_call_response = MagicMock()
        tool_call_response.content = ''
        tool_call_response.tool_calls = [{'name': 'test_tool', 'args': {'query': 'test'}, 'id': 'call_123'}]
        final_response = MagicMock()
        final_response.content = 'Final response'
        final_response.tool_calls = []
        mock_llm = MockAsyncLLM()
        mock_llm.ainvoke_mock.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools_mock.return_value = mock_llm
        agent = Agent(name='EventAgent', system_prompt='You are a test agent.', llm=mock_llm, tools=[test_tool])
        workflow.add_task(name='test_task', agent=agent, instructions='Process: {request}', output_mapping={'response': 'output'})
        workflow.set_entry_point('test_task')
        workflow.add_edge('test_task', workflow.END)
        await workflow.arun({'request': 'test'})
        assert 'agent_start:EventAgent' in events
        assert 'tool_call:test_tool' in events
        assert 'agent_end:EventAgent' in events

    @pytest.mark.asyncio
    async def test_multi_tool_usage(self):
        workflow = Workflow(name='MultiToolTest', state=AgencyState)

        @tool
        def get_weather(city: str) -> str:
            return f'Weather in {city}: 72°F'

        @tool
        def get_restaurants(city: str) -> str:
            return f'Best restaurants in {city}: Italian, Mexican, Asian'
        weather_call = MagicMock()
        weather_call.content = ''
        weather_call.tool_calls = [{'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_1'}]
        restaurant_call = MagicMock()
        restaurant_call.content = ''
        restaurant_call.tool_calls = [{'name': 'get_restaurants', 'args': {'city': 'San Francisco'}, 'id': 'call_2'}]
        final_response = MagicMock()
        final_response.content = "Here's your complete travel guide for San Francisco."
        final_response.tool_calls = []
        mock_llm = MockAsyncLLM()
        mock_llm.ainvoke_mock.side_effect = [weather_call, restaurant_call, final_response]
        mock_llm.bind_tools_mock.return_value = mock_llm
        agent = Agent(name='TravelAgent', system_prompt='You are a comprehensive travel assistant.', llm=mock_llm, tools=[get_weather, get_restaurants])
        workflow.add_task(name='plan_trip', agent=agent, instructions='Create a travel guide for: {request}', output_mapping={'response': 'output'})
        workflow.set_entry_point('plan_trip')
        workflow.add_edge('plan_trip', workflow.END)
        result = await workflow.arun({'request': 'San Francisco'})
        assert 'response' in result
        assert 'travel guide' in result['response'].lower()
        assert mock_llm.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        workflow = Workflow(name='ErrorTest', state=AgencyState)

        @tool
        def failing_tool(query: str) -> str:
            raise Exception('Tool failed')
        tool_call_response = MagicMock()
        tool_call_response.content = ''
        tool_call_response.tool_calls = [{'name': 'failing_tool', 'args': {'query': 'test'}, 'id': 'call_123'}]
        final_response = MagicMock()
        final_response.content = 'I encountered an error, but I can still help.'
        final_response.tool_calls = []
        mock_llm = MockAsyncLLM()
        mock_llm.ainvoke_mock.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools_mock.return_value = mock_llm
        agent = Agent(name='ErrorAgent', system_prompt='You are a helpful agent.', llm=mock_llm, tools=[failing_tool])
        workflow.add_task(name='test_task', agent=agent, instructions='Process: {request}', output_mapping={'response': 'output'})
        workflow.set_entry_point('test_task')
        workflow.add_edge('test_task', workflow.END)
        result = await workflow.arun({'request': 'test'})
        assert 'response' in result
        assert 'error' in result['response'].lower()

    def test_workflow_with_conditional_agency(self):
        workflow = Workflow(name='ConditionalAgency', state=AgencyState)

        @tool
        def analyze_sentiment(text: str) -> str:
            return 'positive' if 'good' in text.lower() else 'negative'
        mock_llm = MockAsyncLLM()
        mock_response = MagicMock()
        mock_response.content = 'Analysis complete'
        mock_response.tool_calls = []
        mock_llm.ainvoke_mock.return_value = mock_response
        mock_llm.bind_tools_mock.return_value = mock_llm
        agent = Agent(name='SentimentAgent', system_prompt='You analyze sentiment.', llm=mock_llm, tools=[analyze_sentiment])
        workflow.add_task(name='analyze', agent=agent, instructions='Analyze: {request}', output_mapping={'response': 'output'})
        workflow.set_entry_point('analyze')

        def should_continue(state: AgencyState) -> str:
            return 'continue' if 'good' in state.response.lower() else 'stop'
        workflow.add_conditional_edges(source='analyze', path=should_continue, paths={'continue': 'analyze', 'stop': workflow.END})
        from agentum.engine import GraphCompiler
        compiler = GraphCompiler(workflow)
        compiled_graph = compiler.compile()
        assert compiled_graph is not None