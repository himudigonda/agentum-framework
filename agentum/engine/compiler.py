from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..core.exceptions import WorkflowDefinitionError
from ..workflow.workflow import Workflow
from .nodes import create_agent_node, create_tool_node


class GraphCompiler:

    def __init__(self, workflow: Workflow):
        self.workflow = workflow

    def compile(self) -> CompiledStateGraph:
        workflow_graph = StateGraph(self.workflow.state_model)
        for task_name, task_details in self.workflow.tasks.items():
            if task_details["agent"]:
                node_func = create_agent_node(task_name, task_details, self.workflow)
            elif task_details["tool"]:
                node_func = create_tool_node(task_name, task_details, self.workflow)
            else:
                continue
            workflow_graph.add_node(task_name, node_func)
        if not self.workflow.entry_point:
            raise WorkflowDefinitionError("Workflow entry point is not set.")
        workflow_graph.set_entry_point(self.workflow.entry_point)
        for edge in self.workflow.edges:
            if isinstance(edge, tuple):
                source, target = edge
                workflow_graph.add_edge(source, target)
            elif isinstance(edge, dict):
                source = edge["source"]
                path_func = edge["path"]
                paths_map = edge["paths"]
                workflow_graph.add_conditional_edges(source, path_func, paths_map)
        return workflow_graph.compile()
