# agentum/state.py
from pydantic import BaseModel


class State(BaseModel):
    """
    The base class for defining the data structure of a workflow.

    By inheriting from this class, you get automatic type validation,
    IDE autocompletion, and a clear definition of the data that flows
    through your agentic system.

    State classes define the schema for data that flows between tasks
    in a workflow. Each field represents a piece of data that can be
    passed between agents and tools.

    Attributes:
        All fields are defined as class attributes with type annotations.
        Default values can be provided for optional fields.

    Example:
        ```python
        class ResearchState(State):
            topic: str
            research_data: str = ""
            summary: str = ""
            confidence_score: float = 0.0
        ```
    """

    pass
