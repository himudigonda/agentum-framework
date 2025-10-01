# examples/07_professional_research.py
"""
Professional Research Pipeline - A Real-World Agentum Example

This example demonstrates agentum's professional capabilities by building
a complete research workflow that:
1. Uses real web search (Tavily API) to gather information
2. Synthesizes findings into a structured report
3. Saves the report to the filesystem

This showcases the transition from "mock tools" to "mission-ready" tools
that developers can use immediately in production.

Key Learning Points:
- How to structure a multi-agent workflow
- Real tool integration (web search + filesystem)
- State management across multiple tasks
- Event-driven observability
- Error handling and validation
"""

import os

from dotenv import load_dotenv

from agentum import Agent, GoogleLLM, State, Workflow, search_web_tavily, write_file

# Load environment variables (API keys, etc.)
load_dotenv()


# 1. DEFINE THE STATE - The Data Contract
# ======================================
# The State class defines the schema for data flowing through our workflow.
# This provides type safety, IDE autocompletion, and clear documentation
# of what data each task expects and produces.
class ResearchReportState(State):
    topic: str  # Input: What to research
    research_data: str = ""  # Output from researcher agent
    report: str = ""  # Output from writer agent
    report_filepath: str = ""  # Where to save the report
    save_status: str = ""  # Confirmation of file save operation


# 2. DEFINE PROFESSIONAL AGENTS - Specialized AI Workers
# ======================================================
# Each agent has a specific role and the tools needed to fulfill it.
# This separation of concerns makes the system more maintainable and
# allows for easy testing and modification of individual components.

# The Researcher agent: Specialized in information gathering
# Uses real web search to find current, accurate information
researcher_agent = Agent(
    name="Researcher",
    system_prompt="You are an expert researcher. Your job is to use the search tool to find comprehensive, up-to-date information on a topic and then synthesize it into a coherent summary.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"),
    tools=[search_web_tavily],  # Real web search capability
)

# The Writer agent: Specialized in content formatting
# Takes research data and formats it into a professional report
writer_agent = Agent(
    name="Writer",
    system_prompt="You are a professional writer. Your job is to format research data into a well-structured markdown report.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"),
    # No tools needed - this agent focuses purely on content formatting
)

# 3. DEFINE THE WORKFLOW - Orchestrating the Process
# ==================================================
# The workflow defines how tasks are connected and how data flows between them.
# This is where we specify the sequence of operations and how outputs
# from one task become inputs to the next.

pro_research_workflow = Workflow(
    name="Professional_Research_Pipeline", state=ResearchReportState
)

# Task 1: Research Phase
# ----------------------
# The researcher agent uses web search to gather information.
# The {topic} placeholder gets filled from the initial state.
# The output is mapped to the 'research_data' field in our state.
pro_research_workflow.add_task(
    name="conduct_research",
    agent=researcher_agent,
    instructions="Perform a thorough web search and synthesize the findings for the topic: {topic}",
    output_mapping={"research_data": "output"},  # Maps agent output to state field
)

# Task 2: Writing Phase
# ---------------------
# The writer agent takes the research data and formats it into a report.
# Notice how {research_data} references the output from the previous task.
# This creates a data dependency chain: topic -> research_data -> report
pro_research_workflow.add_task(
    name="write_report",
    agent=writer_agent,
    instructions="""
    Format the following research data into a clean, well-structured markdown report.
    The report should have a title, an introduction, and sections for key findings.
    
    Research Data:
    {research_data}
    """,
    output_mapping={"report": "output"},  # Maps agent output to state field
)

# Task 3: File System Operation
# -----------------------------
# This demonstrates a direct tool task (not agentic).
# The write_file tool takes the report content and saves it to disk.
# This shows how agentum seamlessly integrates both AI agents and traditional tools.
pro_research_workflow.add_task(
    name="save_report_to_file",
    tool=write_file,  # Direct tool usage (no agent)
    inputs={
        "filepath": "{report_filepath}",
        "content": "{report}",
    },  # Maps state to tool parameters
    output_mapping={"save_status": "output"},  # Captures tool result
)

# 4. DEFINE THE CONTROL FLOW - The Execution Path
# ===============================================
# This defines the sequence of task execution. Each edge represents
# a transition from one task to another, creating a directed graph.
pro_research_workflow.set_entry_point("conduct_research")  # Start here
pro_research_workflow.add_edge(
    "conduct_research", "write_report"
)  # Research -> Writing
pro_research_workflow.add_edge(
    "write_report", "save_report_to_file"
)  # Writing -> Saving
pro_research_workflow.add_edge(
    "save_report_to_file", pro_research_workflow.END
)  # Done!

# 5. EXECUTION & OBSERVABILITY - Running and Monitoring
# ======================================================
# This section demonstrates how to run the workflow and observe its execution.
# The event listener shows agentum's observability capabilities in action.

if __name__ == "__main__":
    # Define the research topic and initial state
    topic = "the latest advancements in large language models"
    initial_state = {
        "topic": topic,
        "report_filepath": f"./{topic.replace(' ', '_')}_report.md",  # Auto-generate filename
    }

    # OBSERVABILITY: Event-Driven Monitoring
    # --------------------------------------
    # This demonstrates agentum's powerful observability system.
    # We can hook into workflow events to monitor execution in real-time.
    @pro_research_workflow.on("agent_tool_call")
    async def log_tool_call(tool_name: str, tool_args: dict):
        """
        Event listener that fires whenever an agent calls a tool.
        This provides real-time visibility into what the agent is doing.
        """
        print(
            f"🔧 Agent is calling a REAL tool: {tool_name} with query: '{tool_args.get('query')}'"
        )

    # EXECUTION: Run the Complete Workflow
    # ------------------------------------
    # This single call orchestrates the entire research pipeline:
    # 1. Researcher searches the web for information
    # 2. Writer formats the findings into a report
    # 3. File system tool saves the report to disk
    print("🚀 Starting Professional Research Pipeline...")
    final_state = pro_research_workflow.run(initial_state)

    # RESULTS: Verify and Display Output
    # ----------------------------------
    # Check that the workflow completed successfully and verify the output
    print("\n--- Workflow Complete ---")
    print(f"Research Status: {final_state['save_status']}")

    # Verify the file was actually created and show a preview
    report_path = final_state["report_filepath"]
    if os.path.exists(report_path):
        print(f"✅ Report successfully saved to: {report_path}")
        with open(report_path, "r") as f:
            print("\n--- Report Content (first 200 chars) ---")
            print(f.read(200) + "...")
    else:
        print(f"❌ Report file was not created at: {report_path}")

    # This example demonstrates the complete journey from idea to implementation:
    # - Real web search using Tavily API
    # - AI-powered content synthesis and formatting
    # - File system integration for persistent output
    # - Event-driven observability for debugging and monitoring
    # - Professional error handling and validation
