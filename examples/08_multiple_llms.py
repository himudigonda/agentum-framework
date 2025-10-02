from dotenv import load_dotenv
from agentum import Agent, AnthropicLLM, GoogleLLM, OpenAILLM, State, Workflow
from agentum.config import settings
load_dotenv()

class StoryState(State):
    topic: str
    story: str = ''
llm_provider = GoogleLLM(api_key=settings.GOOGLE_API_KEY, temperature=0.8, model='gemini-2.5-flash-lite')
storyteller = Agent(name='Storyteller', system_prompt='You are a master storyteller. You write a short, captivating story (3-4 paragraphs) based on a topic.', llm=llm_provider)
story_workflow = Workflow(name='Story_Writing_Workflow', state=StoryState)
story_workflow.add_task(name='write_story', agent=storyteller, instructions='Write a story about: {topic}', output_mapping={'story': 'output'})
story_workflow.set_entry_point('write_story')
story_workflow.add_edge('write_story', story_workflow.END)
if __name__ == '__main__':
    initial_state = {'topic': 'a lost astronaut who finds a mysterious alien artifact'}
    print(f'🚀 Starting story generation with: {llm_provider.__class__.__name__}')
    final_state = story_workflow.run(initial_state)
    print('\n--- Generated Story ---')
    print(final_state['story'])
    print('\n✅ Workflow complete.')