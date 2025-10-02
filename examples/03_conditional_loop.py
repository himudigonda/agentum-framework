import os
from dotenv import load_dotenv
from pydantic import Field
from agentum import Agent, GoogleLLM, State, Workflow
load_dotenv()

class EditorState(State):
    draft: str
    critique_draft: str = ''
    revision_count: int = 0
llm = GoogleLLM(api_key=os.getenv('GOOGLE_API_KEY'), model='gemini-2.5-flash-lite')
editor = Agent(name='Editor', system_prompt='You are a creative editor. You rewrite text based on feedback.', llm=llm)
critic = Agent(name='Critic', system_prompt="You are a tough but fair critic. Review the text. If it is perfect, respond with only the word 'APPROVED'. Otherwise, give one concrete suggestion for improvement.", llm=llm)
editing_workflow = Workflow(name='Revision_Loop_Pipeline', state=EditorState)
editing_workflow.add_task(name='edit_draft', agent=editor, instructions='Please revise this draft: {draft}\nBased on feedback: {critique_draft}', output_mapping={'draft': 'output'})
editing_workflow.add_task(name='critique_draft', agent=critic, instructions='Is this draft good enough? {draft}', output_mapping={'critique_draft': 'output'})
editing_workflow.set_entry_point('critique_draft')
editing_workflow.add_edge('edit_draft', 'critique_draft')

def check_approval(state: EditorState) -> str:
    critique_result = getattr(state, 'critique_draft', '')
    print(f"DEBUG: critique_draft = '{critique_result}'")
    if 'APPROVED' in critique_result:
        print('--- Reviewer approved! Ending loop. ---')
        return 'end'
    else:
        print('--- Revisions needed. Looping back. ---')
        return 'revise'
editing_workflow.add_conditional_edges(source='critique_draft', path=check_approval, paths={'end': editing_workflow.END, 'revise': 'edit_draft'})
if __name__ == '__main__':
    initial_state = {'draft': 'This is the first version of the article.'}
    final_state = editing_workflow.run(initial_state)
    print('\nFinal State:')
    print(final_state)
    assert 'APPROVED' in final_state['critique_draft']
    print('✅ Conditional loop example completed successfully!')