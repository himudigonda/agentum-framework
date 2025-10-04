class AgentumError(Exception):
    pass

class WorkflowDefinitionError(AgentumError):
    pass

class TaskConfigurationError(AgentumError):
    pass

class StateValidationError(AgentumError):
    pass

class CompilationError(AgentumError):
    pass

class ExecutionError(AgentumError):
    pass

class ToolError(AgentumError):
    pass

class MemoryError(AgentumError):
    pass

class RAGError(AgentumError):
    pass