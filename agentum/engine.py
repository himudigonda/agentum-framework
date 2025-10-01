# agentum/engine.py
import asyncio
import inspect
from typing import Any, Dict

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console

from .state import State
from .workflow import Workflow

console = Console()


class GraphCompiler:
    """
    Compiles an agentum Workflow into an executable LangGraph StateGraph.
    """

    def __init__(self, workflow: Workflow):
        self.workflow = workflow

    def _create_agent_node(self, task_name: str, task_details: Dict):
        """Creates a runnable node for an agent task."""
        agent = task_details["agent"]
        instructions_template = task_details["instructions"]

        async def agent_node(state: State) -> Dict[str, Any]:  # Now async
            console.print(
                f"  Executing Agent Task: [bold magenta]{task_name}[/bold magenta]"
            )
            formatted_instructions = instructions_template.format(**state.model_dump())

            # Use the async 'ainvoke' method
            response = await agent.llm.ainvoke(
                f"{agent.system_prompt}\n\n{formatted_instructions}"
            )

            # Store the result in a specific state field based on task name
            if task_name == "critique_draft":
                return {"critique_result": response.content}
            elif task_name == "edit_draft":
                return {"draft": response.content}
            else:
                return {task_name: response.content}

        return agent_node

    def _create_tool_node(self, task_name: str, task_details: Dict):
        """Creates a runnable node for a tool task."""
        tool_func = task_details["tool"]
        input_mapping = task_details["inputs"] or {}

        async def tool_node(state: State) -> Dict[str, Any]:  # Now async
            console.print(f"  Executing Tool Task: [bold cyan]{task_name}[/bold cyan]")
            resolved_inputs = {
                key: template.format(**state.model_dump())
                for key, template in input_mapping.items()
            }

            # Handle both sync and async tools gracefully
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**resolved_inputs)
            else:
                # Run synchronous tools in a separate thread to avoid blocking the event loop
                result = await asyncio.to_thread(tool_func, **resolved_inputs)

            return {task_name: result}

        return tool_node

    def compile(self) -> CompiledStateGraph:
        """Builds the LangGraph StateGraph from the workflow definition."""
        # The state graph is typed with our Pydantic state model
        workflow_graph = StateGraph(self.workflow.state_model)

        # 1. Add nodes for each task (same as before)
        for task_name, task_details in self.workflow.tasks.items():
            if task_details["agent"]:
                node_func = self._create_agent_node(task_name, task_details)
            elif task_details["tool"]:
                node_func = self._create_tool_node(task_name, task_details)
            else:
                continue
            workflow_graph.add_node(task_name, node_func)

        # 2. Set the entry point (same as before)
        if not self.workflow.entry_point:
            raise ValueError("Workflow entry point is not set.")
        workflow_graph.set_entry_point(self.workflow.entry_point)

        # 3. Add edges between nodes (THIS IS THE NEW, UPGRADED LOGIC)
        for edge in self.workflow.edges:
            if isinstance(edge, tuple):
                # This is a simple, direct edge
                source, target = edge
                workflow_graph.add_edge(source, target)
            elif isinstance(edge, dict):
                # This is a conditional edge
                source = edge["source"]
                path_func = edge["path"]
                paths_map = edge["paths"]
                workflow_graph.add_conditional_edges(source, path_func, paths_map)

        # 4. Compile the graph
        # Note: LangGraph's state updates are managed implicitly now.
        # Each node's returned dictionary is merged into the main state object.
        return workflow_graph.compile()
