# agentum/workflow.py
import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Type

from rich.console import Console

# Import moved to _compile method to avoid circular import
from .state import State

console = Console()


class Workflow:
    """
    The main orchestrator for defining, compiling, and running an agentic graph.
    """

    END = "__end__"

    def __init__(
        self, name: str, state: Type[State], persistence: Optional[str] = None
    ):
        self.name = name
        self.state_model = state
        self.persistence = persistence
        self.tasks = {}
        self.edges = []
        self.entry_point = None
        self._compiled_graph = None  # Add a cache for the compiled graph
        self.event_listeners = {}  # NEW: dictionary to hold event listeners
        console.print(f"✨ Workflow '{self.name}' initialized.", style="bold green")

    def on(self, event: str):
        """A decorator to register a listener for a workflow event."""

        def decorator(func: Callable):
            if event not in self.event_listeners:
                self.event_listeners[event] = []
            self.event_listeners[event].append(func)
            return func

        return decorator

    async def _emit(self, event: str, **kwargs):
        """Asynchronously calls all registered listeners for an event."""
        if event in self.event_listeners:
            for listener in self.event_listeners[event]:
                await listener(**kwargs)

    def add_task(
        self,
        name: str,
        agent: Optional[Any] = None,
        tool: Optional[Callable] = None,
        instructions: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        output_mapping: Optional[Dict[str, str]] = None,  # NEW PARAMETER
    ):
        """Adds a node (a unit of work) to the workflow."""
        if name in self.tasks:
            raise ValueError(f"Task '{name}' already exists.")

        # Validate that either agent or tool is provided, but not both
        if not agent and not tool:
            raise ValueError(f"Task '{name}' must have either an agent or a tool.")
        if agent and tool:
            raise ValueError(f"Task '{name}' cannot have both an agent and a tool.")

        # Validate agent-specific requirements
        if agent and not instructions:
            raise ValueError(f"Agent task '{name}' must have instructions.")

        # Validate tool-specific requirements
        if tool and not inputs:
            console.print(
                f"[yellow]Warning: Tool task '{name}' has no input mapping.[/yellow]"
            )

        self.tasks[name] = {
            "agent": agent,
            "tool": tool,
            "instructions": instructions,
            "inputs": inputs,
            "output_mapping": output_mapping or {name: "output"},  # Default mapping
        }
        console.print(f"  - Task added: [cyan]{name}[/cyan]")

    def add_edge(self, source: str, target: str):
        """Defines a direct connection from one task to another."""
        # Validate that source task exists
        if source not in self.tasks and source != self.END:
            raise ValueError(f"Source task '{source}' does not exist.")

        # Validate that target task exists
        if target not in self.tasks and target != self.END:
            raise ValueError(f"Target task '{target}' does not exist.")

        self.edges.append((source, target))
        console.print(f"  - Edge added: [cyan]{source}[/cyan] -> [cyan]{target}[/cyan]")

    def add_conditional_edges(self, source: str, path: Callable, paths: Dict[str, str]):
        """Defines a branching point based on the output of a path function."""
        # Validate that source task exists
        if source not in self.tasks:
            raise ValueError(f"Source task '{source}' does not exist.")

        # Validate that all path targets exist
        for target in paths.values():
            if target not in self.tasks and target != self.END:
                raise ValueError(f"Path target '{target}' does not exist.")

        console.print(f"  - Conditional Edge added from [cyan]{source}[/cyan]")
        # Store this complex edge information for the compiler
        self.edges.append({"source": source, "path": path, "paths": paths})

    def set_entry_point(self, task_name: str):
        """Sets the starting task for the workflow."""
        if task_name not in self.tasks:
            raise ValueError(f"Entry point task '{task_name}' does not exist.")
        self.entry_point = task_name
        console.print(f"  - Entry point set to: [cyan]{task_name}[/cyan]")

    def _compile(self):
        """Compiles the workflow into a runnable graph, caching the result."""
        if not self._compiled_graph:
            console.print(
                "\n🔧 [bold]Compiling workflow into an executable graph...[/bold]",
                style="blue",
            )
            from .engine import GraphCompiler

            compiler = GraphCompiler(self)
            self._compiled_graph = compiler.compile()
            console.print("✅ [bold]Compilation successful.[/bold]", style="green")
        return self._compiled_graph

    def run(self, initial_state: Dict) -> Dict:
        """
        Synchronous wrapper for the async execution of the workflow.
        """
        return asyncio.run(self.arun(initial_state))

    async def arun(self, initial_state: Dict, thread_id: Optional[str] = None) -> Dict:
        """
        Asynchronously executes the workflow with the given initial state.
        """
        await self._emit("workflow_start", workflow_name=self.name, state=initial_state)

        console.print(
            f"\n🚀 [bold]Running workflow '{self.name}'...[/bold]", style="yellow"
        )

        runnable_graph = self._compile()

        config = {}
        if self.persistence and thread_id:
            from langgraph.checkpoint.redis import RedisSaver

            checkpointer = RedisSaver.from_url(self.persistence)
            config = {"configurable": {"thread_id": thread_id}}
            runnable_graph = runnable_graph.with_checkpoints(checkpointer)

        # Use the async 'ainvoke' to run the graph
        final_state = await runnable_graph.ainvoke(initial_state, config=config)

        console.print("\n🏁 [bold]Workflow finished.[/bold]", style="yellow")

        await self._emit("workflow_finish", workflow_name=self.name, state=final_state)
        return final_state

    async def astream(
        self, initial_state: Dict, thread_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Asynchronously streams the output of each step in the workflow.
        """
        console.print(
            f"\n🚀 [bold]Streaming workflow '{self.name}'...[/bold]", style="yellow"
        )
        runnable_graph = self._compile()

        config = {}
        if self.persistence and thread_id:
            from langgraph.checkpoint.redis import RedisSaver

            checkpointer = RedisSaver.from_url(self.persistence)
            config = {"configurable": {"thread_id": thread_id}}
            runnable_graph = runnable_graph.with_checkpoints(checkpointer)

        # Use the async 'astream' to get real-time events from the graph
        async for event in runnable_graph.astream(initial_state, config=config):
            yield event

        console.print("\n🏁 [bold]Workflow stream finished.[/bold]", style="yellow")
