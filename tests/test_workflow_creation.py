# tests/test_workflow_creation.py
from agentum import State, Workflow


class SimpleState(State):
    value: int


def test_workflow_initialization():
    """Tests that a Workflow can be initialized without errors."""
    wf = Workflow(name="TestWorkflow", state=SimpleState)
    assert wf.name == "TestWorkflow"
    assert wf.state_model == SimpleState
    assert wf.tasks == {}
