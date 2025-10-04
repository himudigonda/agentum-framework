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
from agentum.core.config import settings

load_dotenv()


class ResearchReportState(State):
    topic: str
    research_data: str = ""
    report: str = ""
    report_filepath: str = ""
    save_status: str = ""


researcher_agent = Agent(
    name="Researcher",
    system_prompt="You are an expert researcher. Your job is to use the search tool to find comprehensive, up-to-date information on a topic and then synthesize it into a coherent summary.",
    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"),
    tools=[search_web_tavily],
)
writer_agent = Agent(
    name="Writer",
    system_prompt="You are a professional writer. Your job is to format research data into a well-structured markdown report.",
    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"),
)
pro_research_workflow = Workflow(
    name="Professional_Research_Pipeline", state=ResearchReportState
)
pro_research_workflow.add_task(
    name="conduct_research",
    agent=researcher_agent,
    instructions="Perform a thorough web search and synthesize the findings for the topic: {topic}",
    output_mapping={"research_data": "output"},
)
pro_research_workflow.add_task(
    name="write_report",
    agent=writer_agent,
    instructions="\n    Format the following research data into a clean, well-structured markdown report.\n    The report should have a title, an introduction, and sections for key findings.\n    \n    Research Data:\n    {research_data}\n    ",
    output_mapping={"report": "output"},
)
pro_research_workflow.add_task(
    name="save_report_to_file",
    tool=write_file,
    inputs={"filepath": "{report_filepath}", "content": "{report}"},
    output_mapping={"save_status": "output"},
)
pro_research_workflow.set_entry_point("conduct_research")
pro_research_workflow.add_edge("conduct_research", "write_report")
pro_research_workflow.add_edge("write_report", "save_report_to_file")
pro_research_workflow.add_edge("save_report_to_file", pro_research_workflow.END)
if __name__ == "__main__":
    topic = "the latest advancements in large language models"
    initial_state = {
        "topic": topic,
        "report_filepath": f"./{topic.replace(' ', '_')}_report.md",
    }

    @pro_research_workflow.on("agent_tool_call")
    async def log_tool_call(tool_name: str, tool_args: dict):
        print(
            f"🔧 Agent is calling a REAL tool: {tool_name} with query: '{tool_args.get('query')}'"
        )

    print("🚀 Starting Professional Research Pipeline...")
    final_state = pro_research_workflow.run(initial_state)
    print("\n--- Workflow Complete ---")
    print(f"Research Status: {final_state['save_status']}")
    report_path = final_state["report_filepath"]
    if os.path.exists(report_path):
        print(f"✅ Report successfully saved to: {report_path}")
        with open(report_path, "r") as f:
            print("\n--- Report Content (first 200 chars) ---")
            print(f.read(200) + "...")
    else:
        print(f"❌ Report file was not created at: {report_path}")
