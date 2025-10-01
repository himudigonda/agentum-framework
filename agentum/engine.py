# agentum/engine.py
import asyncio
import inspect
from typing import Any, Dict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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

            # 1. Format the initial prompt
            try:
                formatted_instructions = instructions_template.format(
                    **state.model_dump()
                )
            except KeyError as e:
                raise ValueError(
                    f"Missing state key '{e}' required by task '{task_name}' instructions template."
                )

            human_message = HumanMessage(
                content=f"{agent.system_prompt}\n\n{formatted_instructions}"
            )

            # --- NEW MEMORY LOGIC ---
            messages = []
            if agent.memory:
                messages.extend(agent.memory.load_messages())
            messages.append(human_message)

            # --- NEW RESILIENCE LOGIC ---
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
                    key: template.format(**state.model_dump())
                    for key, template in input_mapping.items()
                }
            except KeyError as e:
                raise ValueError(
                    f"Missing state key '{e}' required by tool '{task_name}' input mapping."
                )

            # Handle both sync and async tools gracefully
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**resolved_inputs)
            else:
                # Run synchronous tools in a separate thread to avoid blocking the event loop
                result = await asyncio.to_thread(tool_func, **resolved_inputs)

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
