# agentum/engine.py
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
from .providers.google import GoogleLLM  # Import for multi-modal capability check
from .state import State
from .workflow import Workflow

console = Console()


def _is_safe_path(path_str: str) -> bool:
    """Restrict to relative paths within the project and disallow path traversal."""
    return (
        not path_str.startswith("/")
        and ".." not in path_str
        and not path_str.startswith("~")
        and not path_str.startswith("\\")  # Windows absolute paths
    )


def _safe_format(template: str, state_data: Dict) -> str:
    """Safely formats a string template using a restricted subset of state keys."""
    import re

    # Find all keys in the template (e.g., {key_name})
    keys_in_template = re.findall(r"\{(\w+)\}", template)
    # Create a restricted dict with ONLY the keys present in the template
    # This prevents unintentional interpolation/KeyError from state values.
    restricted_data = {
        k: state_data.get(k, f"{{MISSING_KEY:{k}}}") for k in keys_in_template
    }
    # Perform the actual format. If a key is missing, it will now show up explicitly.
    return template.format(**restricted_data)


class GraphCompiler:
    """
    Compiles an agentum Workflow into an executable LangGraph StateGraph.
    """

    def __init__(self, workflow: Workflow):
        self.workflow = workflow

    def _create_agent_node(self, task_name: str, task_details: Dict):
        """Creates a runnable agent executor node that can use tools."""
        agent = task_details["agent"]
        instructions_template = task_details["instructions"]
        output_mapping = task_details["output_mapping"]

        # --- The New Logic: Bind Tools to the LLM ---
        if agent.tools:
            # This tells the LLM about the available tools and their schemas.
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

            # 1. Format the initial prompt text
            try:
                formatted_instructions = _safe_format(
                    instructions_template, state.model_dump()
                )
            except Exception as e:
                raise StateValidationError(
                    f"Missing state key '{e}' required by task '{task_name}' instructions template."
                )

            # --- UPGRADED MULTI-MODAL LOGIC ---
            # 2. Construct the message content, checking for URL or local path
            prompt_text = f"{agent.system_prompt}\n\n{formatted_instructions}"
            message_content = [{"type": "text", "text": prompt_text}]

            # Check if the LLM supports multi-modal input before attempting image logic
            if isinstance(agent.llm, GoogleLLM):
                # Priority 1: Check for a local image path
                if hasattr(state, "image_path") and getattr(state, "image_path"):
                    image_path_str = getattr(state, "image_path")

                    # Security check: prevent path traversal and arbitrary file access
                    if not _is_safe_path(image_path_str):
                        console.print(
                            f"[bold red]SECURITY ERROR: Path traversal detected in {image_path_str}. Skipping image.[/bold red]"
                        )
                    else:
                        image_path = Path(image_path_str)
                        if image_path.exists():
                            console.print(
                                f"    - Attaching local image for analysis: {image_path_str}"
                            )
                            # Read image, encode to base64, and determine MIME type
                            mime_type, _ = mimetypes.guess_type(image_path)
                            if mime_type:
                                with open(image_path, "rb") as image_file:
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
                                    f"[yellow]Warning: Could not determine MIME type for {image_path_str}. Skipping image.[/yellow]"
                                )
                        else:
                            console.print(
                                f"[yellow]Warning: Image file not found at {image_path_str}. Skipping image.[/yellow]"
                            )

                # Priority 2: Check for an image URL (if no local path was used)
                elif hasattr(state, "image_url") and getattr(state, "image_url"):
                    image_url = getattr(state, "image_url")
                    console.print(
                        f"    - Attaching remote image for analysis: {image_url}"
                    )
                    message_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        }
                    )
            else:
                # Non-multi-modal LLM - skip image processing to avoid token waste
                console.print(
                    "[yellow]Warning: Agent LLM does not support multi-modal input. Image logic skipped.[/yellow]"
                )

            human_message = HumanMessage(content=message_content)

            messages = []
            if agent.memory:
                agent.append_message_for_search(human_message)
                messages.extend(agent.memory.load_messages())

            messages.append(human_message)

            # --- NEW RESILIENCE LOGIC ---
            response = None  # Initialize response variable
            for attempt in range(agent.max_retries):
                try:
                    # The entire agentic loop is now wrapped in a try block
                    while True:
                        await self.workflow._emit("agent_llm_start", messages=messages)
                        response = await llm_with_tools.ainvoke(messages)
                        await self.workflow._emit("agent_llm_end", response=response)

                        if not response.tool_calls:
                            # If no tool calls, the agent has finished its thought process.
                            console.print(
                                f"    - Agent '{agent.name}' responded directly."
                            )

                            # Enhanced: Show the actual agent output in a beautiful panel
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

                        # The agent wants to use a tool!
                        console.print(
                            f"    - Agent '{agent.name}' wants to call tools: {[tc['name'] for tc in response.tool_calls]}"
                        )
                        messages.append(
                            response
                        )  # Add the AI's tool-calling request to history

                        # 3. Execute the requested tools
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
                                # This is a safety check; should rarely happen if the LLM is good
                                messages.append(
                                    ToolMessage(
                                        content=f"Error: Tool '{tool_call['name']}' not found.",
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                                continue

                            try:
                                # Execute the tool and get the result
                                result = await asyncio.to_thread(
                                    tool_func, **tool_call["args"]
                                )

                                # Enhanced: Show tool result in a beautiful panel
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

                                # Add the tool's result to the message history for the next loop
                                messages.append(
                                    ToolMessage(
                                        content=str(result),
                                        tool_call_id=tool_call["id"],
                                    )
                                )
                            except Exception as e:
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

                    # If the loop completes without error, break the retry loop
                    break
                except Exception as e:
                    console.print(
                        f"[bold yellow]  - Attempt {attempt + 1}/{agent.max_retries} failed: {e}[/bold yellow]"
                    )
                    if attempt + 1 == agent.max_retries:
                        # If all retries fail, re-raise the exception
                        raise e
                    # Exponential backoff: wait 1s, 2s, 4s, etc. before retrying
                    await asyncio.sleep(2**attempt)

            # The loop is finished, the final response is in `response.content`
            final_content = response.content

            # --- NEW MEMORY LOGIC ---
            if agent.memory:
                # Save the latest human input and the final AI response
                agent.memory.save_messages([human_message, response])

            await self.workflow._emit(
                "agent_end", agent_name=agent.name, final_response=final_content
            )

            state_update = {
                state_key: final_content
                for state_key, response_key in output_mapping.items()
            }
            await self.workflow._emit(
                "task_finish", task_name=task_name, state_update=state_update
            )
            return state_update

        return agent_node

    def _create_tool_node(self, task_name: str, task_details: Dict):
        """Creates a runnable node for a tool task."""
        tool_func = task_details["tool"]
        input_mapping = task_details["inputs"] or {}
        output_mapping = task_details["output_mapping"]

        async def tool_node(state: State) -> Dict[str, Any]:  # Now async
            await self.workflow._emit("task_start", task_name=task_name, state=state)

            console.print(f"  Executing Tool Task: [bold cyan]{task_name}[/bold cyan]")

            # Resolve inputs with error handling
            try:
                resolved_inputs = {
                    key: _safe_format(template, state.model_dump())
                    for key, template in input_mapping.items()
                }
            except Exception as e:  # Broader catch for potential _safe_format errors
                raise StateValidationError(
                    f"Error resolving inputs for tool '{task_name}': {e}"
                )

            # Handle both sync and async tools gracefully
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**resolved_inputs)
            else:
                # Run synchronous tools in a separate thread to avoid blocking the event loop
                result = await asyncio.to_thread(tool_func, **resolved_inputs)

            # Enhanced: Show tool result in a beautiful panel
            tool_output = str(result).strip()
            console.print(
                Panel(
                    tool_output,
                    title=f"[bold cyan]Tool '{task_name}'[/bold cyan] Result",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

            # Use the explicit mapping for tools as well
            state_update = {
                state_key: result for state_key, response_key in output_mapping.items()
            }
            await self.workflow._emit(
                "task_finish", task_name=task_name, state_update=state_update
            )
            return state_update

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
            raise WorkflowDefinitionError("Workflow entry point is not set.")
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
        # Note: LangGraph's state updates are managed implicitly now.
        # Each node's returned dictionary is merged into the main state object.
        return workflow_graph.compile()
        # Note: LangGraph's state updates are managed implicitly now.
        # Each node's returned dictionary is merged into the main state object.
        return workflow_graph.compile()
