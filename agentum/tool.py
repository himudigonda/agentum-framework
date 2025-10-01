# agentum/tool.py
import functools
import inspect

from pydantic import create_model


def tool(func):
    """
    A decorator that marks a Python function as a tool and attaches its schema.

    This decorator transforms any Python function into an agentum tool by:
    1. Introspecting the function's signature to create a Pydantic schema
    2. Attaching metadata for the LLM to understand the tool's capabilities
    3. Preserving the original function's behavior

    The decorated function can then be used by agents in workflows.

    Args:
        func: The function to convert into a tool

    Returns:
        The decorated function with tool metadata attached

    Example:
        ```python
        @tool
        def search_web(query: str) -> str:
            \"\"\"
            Search the web for information about a given query.

            Args:
                query: The search query string

            Returns:
                Search results as a formatted string
            \"\"\"
            # Implementation here
            return results

        # Now this can be used in an agent
        agent = Agent(
            name="Researcher",
            llm=llm,
            tools=[search_web]
        )
        ```
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # --- The Magic: Introspect the function to build a Pydantic model ---
    sig = inspect.signature(func)
    fields = {param.name: (param.annotation, ...) for param in sig.parameters.values()}
    # Create a dynamic Pydantic model representing the function's arguments
    tool_schema = create_model(f"{func.__name__}Schema", **fields)

    # Attach metadata for the engine to use
    wrapper._is_agentum_tool = True
    wrapper._tool_schema = tool_schema
    wrapper._tool_description = func.__doc__ or "No description provided."

    return wrapper
