import asyncio
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Type

from rich.console import Console

from .exceptions import TaskConfigurationError, WorkflowDefinitionError
from .state import State

console = Console()


class Workflow:

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
        self._compiled_graph = None
        self.event_listeners = {}
        console.print(f"✨ Workflow '{self.name}' initialized.", style="bold green")

    def on(self, event: str):

        def decorator(func: Callable):
            if event not in self.event_listeners:
                self.event_listeners[event] = []
            self.event_listeners[event].append(func)
            return func

        return decorator

    async def _emit(self, event: str, **kwargs):
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
        output_mapping: Optional[Dict[str, str]] = None,
    ):
        if name in self.tasks:
            raise TaskConfigurationError(f"Task '{name}' already exists.")

        if not agent and not tool:
            raise TaskConfigurationError(
                f"Task '{name}' must have either an agent or a tool."
            )
        if agent and tool:
            raise TaskConfigurationError(
                f"Task '{name}' cannot have both an agent and a tool."
            )

        if agent and not instructions:
            raise TaskConfigurationError(f"Agent task '{name}' must have instructions.")

        if tool and not inputs:
            console.print(
                f"[yellow]Warning: Tool task '{name}' has no input mapping.[/yellow]"
            )

        self.tasks[name] = {
            "agent": agent,
            "tool": tool,
            "instructions": instructions,
            "inputs": inputs,
            "output_mapping": output_mapping,
        }
        console.print(f"  - Task added: [cyan]{name}[/cyan]")

    def add_edge(self, source: str, target: str):
        if source not in self.tasks and source != self.END:
            raise WorkflowDefinitionError(f"Source task '{source}' does not exist.")

        if target not in self.tasks and target != self.END:
            raise WorkflowDefinitionError(f"Target task '{target}' does not exist.")

        self.edges.append((source, target))
        console.print(f"  - Edge added: [cyan]{source}[/cyan] -> [cyan]{target}[/cyan]")

    def add_conditional_edges(self, source: str, path: Callable, paths: Dict[str, str]):
        if source not in self.tasks:
            raise WorkflowDefinitionError(f"Source task '{source}' does not exist.")

        for target in paths.values():
            if target not in self.tasks and target != self.END:
                raise WorkflowDefinitionError(f"Path target '{target}' does not exist.")

        console.print(f"  - Conditional Edge added from [cyan]{source}[/cyan]")
        self.edges.append({"source": source, "path": path, "paths": paths})

    def set_entry_point(self, task_name: str):
        if task_name not in self.tasks:
            raise WorkflowDefinitionError(
                f"Entry point task '{task_name}' does not exist."
            )
        self.entry_point = task_name
        console.print(f"  - Entry point set to: [cyan]{task_name}[/cyan]")

    def _compile(self):
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
        return asyncio.run(self.arun(initial_state))

    async def arun(self, initial_state: Dict, thread_id: Optional[str] = None) -> Dict:
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

        final_state = await runnable_graph.ainvoke(initial_state, config=config)

        console.print("\n🏁 [bold]Workflow finished.[/bold]", style="yellow")

        await self._emit("workflow_finish", workflow_name=self.name, state=final_state)
        return final_state

    async def astream(
        self, initial_state: Dict, thread_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
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

        async for event in runnable_graph.astream(initial_state, config=config):
            yield event

        console.print("\n🏁 [bold]Workflow stream finished.[/bold]", style="yellow")
