"""
Flagship Example: Dynamic Multi-Agent Planning (The Planner Agent Pattern)

This example showcases the most advanced form of agency in Agentum: a Planner Agent.

1.  **Planner Agent (The Brain):** Receives a complex goal and outputs the name of the NEXT best task to run.
2.  **Worker Agents (The Hands):** Execute the specific task (e.g., Research, Summarize).
3.  **Conditional Edge (The Conductor):** Uses the Planner's text output to dynamically route the workflow.

This demonstrates true autonomous task decomposition.

To run this, ensure GOOGLE_API_KEY is set in your .env file.
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import Field

from agentum import Agent, GoogleLLM, State, Workflow, tool
from agentum.config import settings

load_dotenv()


# 1. DEFINE THE STATE (What the whole team works on)
class ResearchPlanState(State):
    goal: str = Field(description="The complex goal the user wants to achieve.")
    plan: str = Field(description="The next task the Planner has decided to execute.")
    introduction: str = ""
    body_section: str = ""
    conclusion: str = ""
    final_report: str = ""


# 2. DEFINE THE WORKER AGENTS (The Hands)
# We will use the same LLM for all agents for consistency
llm = GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite")

researcher = Agent(
    name="Researcher",
    system_prompt="You are an expert market researcher. You write detailed, factual text.",
    llm=llm,
)

writer = Agent(
    name="Writer",
    system_prompt="You are an expert report writer. You take text and integrate it into a formal report.",
    llm=llm,
)


# 3. DEFINE THE PLANNER AGENT (The Brain)
# Crucially, its output format is restricted to one of the task names or 'DONE'.
planner = Agent(
    name="Planner",
    system_prompt="""You are the ultimate workflow orchestrator. Your job is to decide the single, next task to execute to complete the user's GOAL.
    
    You must output ONLY one of the following task names or 'DONE' if the GOAL is complete:
    - 'write_introduction'
    - 'write_body'
    - 'write_conclusion'
    - 'assemble_report'
    - 'DONE'
    
    Current State of Report:
    - Introduction: {introduction}
    - Body: {body_section}
    - Conclusion: {conclusion}
    
    GOAL: {goal}
    """,
    llm=llm,
)


# 4. DEFINE THE WORKFLOW
planner_workflow = Workflow(name="Dynamic_Planner_Workflow", state=ResearchPlanState)

# 4.1. Define Worker Tasks
planner_workflow.add_task(
    name="write_introduction",
    agent=researcher,
    instructions="Write a comprehensive 2-paragraph introduction for the report on the GOAL: {goal}",
    output_mapping={"introduction": "output"},
)

planner_workflow.add_task(
    name="write_body",
    agent=researcher,
    instructions="Write the main 3-paragraph body section for the report on the GOAL: {goal}",
    output_mapping={"body_section": "output"},
)

planner_workflow.add_task(
    name="write_conclusion",
    agent=researcher,
    instructions="Write a concise conclusion for the report on the GOAL: {goal}",
    output_mapping={"conclusion": "output"},
)

planner_workflow.add_task(
    name="assemble_report",
    agent=writer,
    instructions="""Assemble the final professional report from these parts:
    Introduction: {introduction}
    Body: {body_section}
    Conclusion: {conclusion}
    """,
    output_mapping={"final_report": "output"},
)


# 4.2. Define the Planner Task
planner_workflow.add_task(
    name="planning_step",
    agent=planner,
    # The planner's instructions template includes the current report state!
    instructions="Decide the next task based on the current report state to achieve the GOAL: {goal}",
    output_mapping={"plan": "output"},
)


# 4.3. Define the Dynamic Control Flow
# Our entry point is the planner.
planner_workflow.set_entry_point("planning_step")

# Our worker tasks ALL loop back to the planner.
planner_workflow.add_edge("write_introduction", "planning_step")
planner_workflow.add_edge("write_body", "planning_step")
planner_workflow.add_edge("write_conclusion", "planning_step")
planner_workflow.add_edge("assemble_report", "planning_step")


# 4.4. Define the Conditional Edge controlled by the Planner's output
def planner_decides_next_step(state: ResearchPlanState) -> str:
    """The path function: looks at the 'plan' state field to determine the next node."""
    next_task = getattr(state, "plan", "").strip().upper()
    print(f"🧠 Planner decided next action: {next_task}")

    # Use the next task name as the route key
    if next_task == "DONE":
        return "DONE"
    return next_task


planner_workflow.add_conditional_edges(
    source="planning_step",
    path=planner_decides_next_step,
    paths={
        "write_introduction": "write_introduction",
        "write_body": "write_body",
        "write_conclusion": "write_conclusion",
        "assemble_report": "assemble_report",
        "DONE": planner_workflow.END,
    },
)


# 5. RUN IT!
if __name__ == "__main__":
    initial_goal = "Write a complete, three-part professional report analyzing the impact of quantum computing on modern cryptography."

    initial_state = {"goal": initial_goal}

    print(f"🚀 Starting Dynamic Planner Workflow with GOAL: {initial_goal}")

    final_state = planner_workflow.run(initial_state)

    print("\n" + "=" * 80)
    print("🏁 FINAL ASSEMBLED REPORT")
    print("=" * 80)
    print(final_state["final_report"])

    # The plan field should contain 'DONE' if the planner correctly decided the final step
    assert "DONE" in final_state.get("plan", "").upper()
    print("\n✅ Dynamic planning and execution successful!")
