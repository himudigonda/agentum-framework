import functools
import inspect
from pydantic import create_model

def tool(func=None, *, name=None):

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        tool_name = name or f.__name__
        sig = inspect.signature(f)
        fields = {param.name: (param.annotation, ...) for param in sig.parameters.values()}
        tool_schema = create_model(f'{tool_name}Schema', **fields)
        wrapper._is_agentum_tool = True
        wrapper._tool_schema = tool_schema
        wrapper._tool_description = f.__doc__ or 'No description provided.'
        wrapper.__name__ = tool_name
        return wrapper
    if func is None:
        return decorator
    else:
        return decorator(func)