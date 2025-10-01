# agentum/tool.py
import functools


def tool(func):
    """
    A decorator that marks a Python function as a tool that can be used by an Agent.

    In the future, this will inspect the function's signature and docstring
    to make it available to the agent executor.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # We can attach metadata to the function for later use
    wrapper._is_agentum_tool = True
    return wrapper
