# agentum/agent.py
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict


class Agent(BaseModel):
    """
    Represents an AI agent with a specific role, configuration, and capabilities.

    An Agent is the core building block of agentum workflows. It encapsulates
    an LLM with a specific personality, tools, and memory capabilities.

    Attributes:
        name: A unique identifier for this agent
        system_prompt: The system prompt that defines the agent's role and behavior
        llm: The language model instance (e.g., GoogleLLM, OpenAILLM)
        tools: Optional list of tools this agent can use
        memory: Optional conversation memory for multi-turn interactions
        max_retries: Maximum number of retry attempts on failure (default: 3)

    Example:
        ```python
        researcher = Agent(
            name="Researcher",
            system_prompt="You are an expert researcher who finds accurate information.",
            llm=GoogleLLM(api_key="your_key"),
            tools=[search_web_tavily, read_file],
            memory=ConversationMemory()
        )
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    system_prompt: str
    llm: Any  # For now, we'll keep this generic.
    tools: Optional[List[Any]] = None  # Changed from callable to Any
    memory: Optional[Any] = None  # NEW: memory support
    max_retries: int = 3  # NEW: Default to 3 retries
    # Future additions: structured_output
