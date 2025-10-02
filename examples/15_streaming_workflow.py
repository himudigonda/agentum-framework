import asyncio
import os
from dotenv import load_dotenv
from agentum import Agent, GoogleLLM, State, Workflow, search_web_tavily
from agentum.config import settings
load_dotenv()

class StreamingState(State):
    topic: str
    research_data: str = ''
    summary: str = ''
researcher = Agent(name='StreamResearcher', system_prompt='You are an expert researcher. Find a detailed article on a topic.', llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model='gemini-2.5-flash-lite'), tools=[search_web_tavily])
summarizer = Agent(name='StreamSummarizer', system_prompt='You are an expert summarizer. Create a concise, one-paragraph summary of the provided text.', llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model='gemini-2.5-flash-lite'))
streaming_workflow = Workflow(name='Streaming_Pipeline', state=StreamingState)
streaming_workflow.add_task(name='research', agent=researcher, instructions='Find information on: {topic}', output_mapping={'research_data': 'output'})
streaming_workflow.add_task(name='summarize', agent=summarizer, instructions='Summarize this text: {research_data}', output_mapping={'summary': 'output'})
streaming_workflow.set_entry_point('research')
streaming_workflow.add_edge('research', 'summarize')
streaming_workflow.add_edge('summarize', streaming_workflow.END)

async def main():
    initial_state = {'topic': 'The history of the internet'}
    print('--------------------------------------------------------------------------------')
    print('🚀 Running the stream via the CLI for a rich, real-time trace.')
    print('   Please run the following command in your terminal for the full experience:')
    print('   [bold]agentum run examples/15_streaming_workflow.py --stream[/bold]')
    print('--------------------------------------------------------------------------------')
    print('\n[dim]Running simple programmatic stream...[/dim]')
    async for event in streaming_workflow.astream(initial_state):
        if 'research' in event:
            print('✅ RESEARCH TASK COMPLETE')
        if 'summarize' in event:
            print('✅ SUMMARIZE TASK COMPLETE')
    print('🏁 Programmatic stream finished.')
if __name__ == '__main__':
    asyncio.run(main())