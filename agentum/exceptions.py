# agentum/exceptions.py
"""
Custom exceptions for the agentum framework.
"""


class AgentumError(Exception):
    """Base exception for all agentum-related errors."""
    pass


class WorkflowDefinitionError(AgentumError):
    """Raised when there's an error in workflow definition."""
    pass


class TaskConfigurationError(AgentumError):
    """Raised when a task is configured incorrectly."""
    pass


class StateValidationError(AgentumError):
    """Raised when state validation fails."""
    pass


class CompilationError(AgentumError):
    """Raised when workflow compilation fails."""
    pass


class ExecutionError(AgentumError):
    """Raised when workflow execution fails."""
    pass


class ToolError(AgentumError):
    """Raised when a tool execution fails."""
    pass


class MemoryError(AgentumError):
    """Raised when memory operations fail."""
    pass


class RAGError(AgentumError):
    """Raised when RAG operations fail."""
    pass
