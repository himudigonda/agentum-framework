# agentum/agent.py
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict


class Agent(BaseModel):
    """
    Represents an AI agent with a specific role, configuration, and capabilities.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    system_prompt: str
    llm: Any  # For now, we'll keep this generic.
    tools: Optional[List[Any]] = None  # Changed from callable to Any
    memory: Optional[Any] = None  # NEW: memory support
    # Future additions: max_retries, structured_output
