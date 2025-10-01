# agentum/engine.py
import asyncio
import inspect
from typing import Any, Dict

from langgraph.graph import StateGraph
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

        def agent_node(state: State) -> Dict[str, Any]:
            console.print(
                f"  Executing Agent Task: [bold magenta]{task_name}[/bold magenta]"
            )
            # Format the instructions with the current state
            formatted_instructions = instructions_template.format(**state.model_dump())

            # This is a simplified invocation. We will add tool support later.
            response = agent.llm.invoke(
                f"{agent.system_prompt}\n\n{formatted_instructions}"
            )

            # Return the updated state with the task output
            return {task_name: response.content}

        return agent_node

    def _create_tool_node(self, task_name: str, task_details: Dict):
        """Creates a runnable node for a tool task."""
        tool_func = task_details["tool"]
        input_mapping = task_details["inputs"] or {}

        def tool_node(state: State) -> Dict[str, Any]:
            console.print(f"  Executing Tool Task: [bold cyan]{task_name}[/bold cyan]")
            # Resolve inputs from state
            resolved_inputs = {
                key: template.format(**state.model_dump())
                for key, template in input_mapping.items()
            }

            # Execute the tool synchronously
            result = tool_func(**resolved_inputs)

            return {task_name: result}

        return tool_node

    def compile(self) -> StateGraph:
        """Builds the LangGraph StateGraph from the workflow definition."""

        # Define a reducer function for state updates
        def reducer(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
            """Merges state updates from nodes."""
            return {**left, **right}

        # The state graph is typed with our Pydantic state model
        workflow_graph = StateGraph(self.workflow.state_model, reducer=reducer)

        # 1. Add nodes for each task
        for task_name, task_details in self.workflow.tasks.items():
            if task_details["agent"]:
                node_func = self._create_agent_node(task_name, task_details)
            elif task_details["tool"]:
                node_func = self._create_tool_node(task_name, task_details)
            else:
                # This could be a placeholder for future task types like sub-workflows
                continue

            workflow_graph.add_node(task_name, node_func)

        # 2. Set the entry point
        if not self.workflow.entry_point:
            raise ValueError("Workflow entry point is not set.")
        workflow_graph.set_entry_point(self.workflow.entry_point)

        # 3. Add edges between nodes
        for source, target in self.workflow.edges:
            workflow_graph.add_edge(source, target)

        # 4. Compile the graph
        return workflow_graph.compile()
