from enum import Enum


class WorkflowEvents(str, Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_FINISH = "workflow_finish"
    TASK_START = "task_start"
    TASK_FINISH = "task_finish"
    AGENT_START = "agent_start"
    AGENT_LLM_START = "agent_llm_start"
    AGENT_LLM_END = "agent_llm_end"
    AGENT_TOOL_CALL = "agent_tool_call"
    AGENT_TOOL_RESULT = "agent_tool_result"
    AGENT_END = "agent_end"
