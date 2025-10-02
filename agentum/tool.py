import functools
import inspect

from pydantic import create_model


def tool(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    sig = inspect.signature(func)
    fields = {param.name: (param.annotation, ...) for param in sig.parameters.values()}
    tool_schema = create_model(f"{func.__name__}Schema", **fields)

    wrapper._is_agentum_tool = True
    wrapper._tool_schema = tool_schema
    wrapper._tool_description = func.__doc__ or "No description provided."

    return wrapper
