# agentum/tool.py
import functools
import inspect

from pydantic import create_model


def tool(func):
    """
    A decorator that marks a Python function as a tool and attaches its schema.
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
