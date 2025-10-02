import os
from dotenv import load_dotenv
from pydantic import Field
from agentum import Agent, GoogleLLM, State, Workflow, tool
from agentum.config import settings
load_dotenv()

class ResearchState(State):
    topic: str
    conduct_research: str = Field(default='')
    save_to_file: str = Field(default='')

@tool
def save_summary(summary: str):
    print('\n--- TOOL EXECUTING ---')
    print(f"Saving summary: '{summary[:60]}...'")
    print('--- TOOL FINISHED ---\n')
    return f"Summary for topic '{summary.split()[0]}' saved successfully."
google_llm = GoogleLLM(api_key=settings.GOOGLE_API_KEY, model='gemini-2.5-flash-lite')
researcher = Agent(name='Researcher', system_prompt='You are a helpful research assistant. Keep your answers concise and to the point (2-3 sentences).', llm=google_llm)
research_workflow = Workflow(name='Basic_Research_Pipeline', state=ResearchState)
research_workflow.add_task(name='conduct_research', agent=researcher, instructions='Please research the topic: {topic}')
research_workflow.add_task(name='save_to_file', tool=save_summary, inputs={'summary': '{conduct_research}'})
research_workflow.set_entry_point('conduct_research')
research_workflow.add_edge('conduct_research', 'save_to_file')
research_workflow.add_edge('save_to_file', research_workflow.END)
if __name__ == '__main__':
    initial_state = {'topic': 'The Future of AI'}
    final_state = research_workflow.run(initial_state)
    print('\nFinal State:')
    print(final_state)