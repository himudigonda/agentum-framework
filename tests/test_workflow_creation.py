from agentum import State, Workflow

class SimpleState(State):
    value: int

def test_workflow_initialization():
    wf = Workflow(name='TestWorkflow', state=SimpleState)
    assert wf.name == 'TestWorkflow'
    assert wf.state_model == SimpleState
    assert wf.tasks == {}