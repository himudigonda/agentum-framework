import asyncio
import base64
import inspect
import mimetypes
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.panel import Panel

from .exceptions import ExecutionError, StateValidationError, WorkflowDefinitionError
from .providers.google import GoogleLLM
from .state import State
from .workflow import Workflow

console = Console()

SAFE_BASE_DIR = Path.cwd().resolve()

try:
    from jinja2 import Environment, meta
except ImportError:

    class Environment:
        def __init__(self, *args, **kwargs):
            pass

        def from_string(self, *args, **kwargs):
            raise ImportError(
                "Jinja2 is required for state templating. Please run 'pip install jinja2'"
            )


jinja_env = Environment()


def _is_safe_path(path_str: str) -> bool:
    return (
        not path_str.startswith("/")
        and ".." not in path_str
        and (not path_str.startswith("~"))
        and (not path_str.startswith("\\"))
    )


def _safe_format(template: str, state_data: Dict) -> str:
    try:
        jinja_template = jinja_env.from_string(template)
        template_vars = meta.find_undeclared_variables(jinja_env.parse(template))

        context = {key: state_data[key] for key in template_vars if key in state_data}

        missing_keys = template_vars - context.keys()
        if missing_keys:
            raise KeyError(
                f"Missing keys required by template: {', '.join(missing_keys)}"
            )

        return jinja_template.render(context)
    except ImportError as e:
        raise e
    except KeyError as e:
        raise StateValidationError(f"Missing state key {e} required by template.")
    except Exception as e:
        raise ExecutionError(f"Failed to render template: {e}")


class GraphCompiler:

    def __init__(self, workflow: Workflow):
        self.workflow = workflow

    def _create_agent_node(self, task_name: str, task_details: Dict):
        agent = task_details["agent"]
        instructions_template = task_details["instructions"]
        output_mapping = task_details["output_mapping"]
        if agent.tools:
            console.print(
                f"    - Binding {len(agent.tools)} tools to LLM: {[t.__name__ for t in agent.tools]}"
            )
            llm_with_tools = agent.llm.bind_tools(agent.tools)
        else:
            console.print("    - No tools available for this agent")
            llm_with_tools = agent.llm

        async def agent_node(state: State) -> Dict[str, Any]:
            await self.workflow._emit("task_start", task_name=task_name, state=state)
            await self.workflow._emit("agent_start", agent_name=agent.name, state=state)
            console.print(
                f"  Executing Agent Task: [bold magenta]{task_name}[/bold magenta]"
            )
            try:
                formatted_instructions = _safe_format(
                    instructions_template, state.model_dump()
                )
            except Exception as e:
                raise StateValidationError(
                    f"Missing state key '{e}' required by task '{task_name}' instructions template."
                )
            prompt_text = f"{agent.system_prompt}\n\n{formatted_instructions}"
            message_content = [{"type": "text", "text": prompt_text}]
            if isinstance(agent.llm, GoogleLLM):
                if hasattr(state, "image_path") and getattr(state, "image_path"):
                    image_path_str = getattr(state, "image_path")

                    try:
                        user_path = SAFE_BASE_DIR / image_path_str
                        resolved_path = user_path.resolve()

                        if not resolved_path.is_relative_to(SAFE_BASE_DIR):
                            console.print(
                                f"[bold red]SECURITY ERROR: Path traversal detected in {image_path_str}. The path is outside the safe directory.[/bold red]"
                            )
                        elif not resolved_path.exists():
                            console.print(
                                f"[yellow]Warning: Image file not found at {resolved_path}. Skipping image.[/yellow]"
                            )
                        else:
                            console.print(
                                f"    - Attaching local image for analysis: {resolved_path}"
                            )
                            mime_type, _ = mimetypes.guess_type(resolved_path)
                            if mime_type:
                                with open(resolved_path, "rb") as image_file:
                                    base64_image = base64.b64encode(
                                        image_file.read()
                                    ).decode("utf-8")
                                message_content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_image}"
                                        },
                                    }
                                )
                            else:
                                console.print(
                                    f"[yellow]Warning: Could not determine MIME type for {resolved_path}. Skipping image.[/yellow]"
                                )
                    except Exception as e:
                        console.print(
                            f"[bold red]SECURITY ERROR: Could not process path {image_path_str}. Reason: {e}[/bold red]"
                        )
                elif hasattr(state, "image_url") and getattr(state, "image_url"):
                    image_url = getattr(state, "image_url")
                    console.print(
                        f"    - Attaching remote image for analysis: {image_url}"
                    )
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": image_url}}
                    )
            else:
                console.print(
                    "[yellow]Warning: Agent LLM does not support multi-modal input. Image logic skipped.[/yellow]"
                )
            human_message = HumanMessage(content=message_content)
            messages = []
            if agent.memory:
                messages.extend(agent.memory.load_messages(human_message))
            messages.append(human_message)
            response = None
            last_tool_result = None
            for attempt in range(agent.max_retries):
                try:
                    while True:
                        await self.workflow._emit("agent_llm_start", messages=messages)
                        response = await llm_with_tools.ainvoke(messages)
                        await self.workflow._emit("agent_llm_end", response=response)
                        if not response.tool_calls:
                            console.print(
                                f"    - Agent '{agent.name}' responded directly."
                            )
                            agent_output = response.content.strip()
                            console.print(
                                Panel(
                                    agent_output,
                                    title=f"[bold blue]{agent.name}[/bold blue] Output",
                                    border_style="blue",
                                    padding=(1, 2),
                                )
                            )
                            break
                        console.print(
                            f"    - Agent '{agent.name}' wants to call tools: {[tc['name'] for tc in response.tool_calls]}"
                        )
                        messages.append(response)
                        for tool_call in response.tool_calls:
                            await self.workflow._emit(
                                "agent_tool_call",
                                tool_name=tool_call["name"],
                                tool_args=tool_call["args"],
                            )
                            tool_func = next(
                                (
                                    t
                                    for t in agent.tools
                                    if t.__name__ == tool_call["name"]
                                ),
                                None,
                            )
                            if not tool_func:
                                messages.append(
                                    ToolMessage(
                                        content=f"Error: Tool '{tool_call['name']}' not found.",
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                                continue
                            try:
                                result = await asyncio.to_thread(
                                    tool_func, **tool_call["args"]
                                )
                                last_tool_result = result
                                tool_output = str(result).strip()
                                console.print(
                                    Panel(
                                        tool_output,
                                        title=f"[bold green]Tool '{tool_call['name']}'[/bold green] Result",
                                        border_style="green",
                                        padding=(1, 2),
                                    )
                                )
                                await self.workflow._emit(
                                    "agent_tool_result",
                                    tool_name=tool_call["name"],
                                    result=result,
                                )
                                messages.append(
                                    ToolMessage(
                                        content=str(result),
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                            except Exception as e:
                                last_tool_result = None
                                error_message = (
                                    f"Error executing tool '{tool_call['name']}': {e}"
                                )
                                console.print(
                                    f"[bold red]    - {error_message}[/bold red]"
                                )
                                messages.append(
                                    ToolMessage(
                                        content=error_message,
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                    break
                except Exception as e:
                    console.print(
                        f"[bold yellow]  - Attempt {attempt + 1}/{agent.max_retries} failed: {e}[/bold yellow]"
                    )
                    if attempt + 1 == agent.max_retries:
                        raise e
                    await asyncio.sleep(2**attempt)
            final_content = response.content
            if agent.memory:
                agent.memory.save_messages([human_message, response])
            await self.workflow._emit(
                "agent_end", agent_name=agent.name, final_response=final_content
            )
            state_update = {}
            if output_mapping:
                for state_key, source_key in output_mapping.items():
                    if source_key == "tool_result":
                        state_update[state_key] = last_tool_result
                    else:
                        state_update[state_key] = final_content
            await self.workflow._emit(
                "task_finish", task_name=task_name, state_update=state_update
            )
            return state_update

        return agent_node

    def _create_tool_node(self, task_name: str, task_details: Dict):
        tool_func = task_details["tool"]
        input_mapping = task_details["inputs"] or {}
        output_mapping = task_details["output_mapping"]

        async def tool_node(state: State) -> Dict[str, Any]:
            await self.workflow._emit("task_start", task_name=task_name, state=state)
            console.print(f"  Executing Tool Task: [bold cyan]{task_name}[/bold cyan]")
            try:
                resolved_inputs = {
                    key: _safe_format(template, state.model_dump())
                    for key, template in input_mapping.items()
                }
            except Exception as e:
                raise StateValidationError(
                    f"Error resolving inputs for tool '{task_name}': {e}"
                )
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**resolved_inputs)
            else:
                result = await asyncio.to_thread(tool_func, **resolved_inputs)
            tool_output = str(result).strip()
            console.print(
                Panel(
                    tool_output,
                    title=f"[bold cyan]Tool '{task_name}'[/bold cyan] Result",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            state_update = {}
            if output_mapping:
                state_update = {
                    state_key: result
                    for state_key, response_key in output_mapping.items()
                }
            await self.workflow._emit(
                "task_finish", task_name=task_name, state_update=state_update
            )
            return state_update

        return tool_node

    def compile(self) -> CompiledStateGraph:
        workflow_graph = StateGraph(self.workflow.state_model)
        for task_name, task_details in self.workflow.tasks.items():
            if task_details["agent"]:
                node_func = self._create_agent_node(task_name, task_details)
            elif task_details["tool"]:
                node_func = self._create_tool_node(task_name, task_details)
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
