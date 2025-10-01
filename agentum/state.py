# agentum/state.py
from pydantic import BaseModel


class State(BaseModel):
    """
    The base class for defining the data structure of a workflow.

    By inheriting from this class, you get automatic type validation,
    IDE autocompletion, and a clear definition of the data that flows
    through your agentic system.
    """

    pass
