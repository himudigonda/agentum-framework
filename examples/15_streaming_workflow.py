"""
This example demonstrates how to get real-time feedback from a workflow
using the `astream()` method.

Streaming is crucial for:
- Building responsive user interfaces that show progress.
- Real-time debugging and logging of an agent's execution.
- Monitoring long-running workflows.

This workflow simulates a simple research and summarization task.
We will iterate through the events as they are produced by the graph.
"""

import asyncio
import os

from dotenv import load_dotenv

from agentum import Agent, GoogleLLM, State, Workflow, search_web_tavily

load_dotenv()


# 1. Define State and Agents
class StreamingState(State):
    topic: str
    research_data: str = ""
    summary: str = ""


researcher = Agent(
    name="StreamResearcher",
    system_prompt="You are an expert researcher. Find a detailed article on a topic.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"),
    tools=[search_web_tavily],
)
summarizer = Agent(
    name="StreamSummarizer",
    system_prompt="You are an expert summarizer. Create a concise, one-paragraph summary of the provided text.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"),
)

# 2. Define the Workflow
streaming_workflow = Workflow(name="Streaming_Pipeline", state=StreamingState)

streaming_workflow.add_task(
    name="research",
    agent=researcher,
    instructions="Find information on: {topic}",
    output_mapping={"research_data": "output"},
)
streaming_workflow.add_task(
    name="summarize",
    agent=summarizer,
    instructions="Summarize this text: {research_data}",
    output_mapping={"summary": "output"},
)

streaming_workflow.set_entry_point("research")
streaming_workflow.add_edge("research", "summarize")
streaming_workflow.add_edge("summarize", streaming_workflow.END)


# 3. Run and Process the Stream
async def main():
    initial_state = {"topic": "The history of the internet"}

    print("🚀 Starting workflow stream...")
    print("-" * 30)

    # astream() returns an async generator that yields events as they complete.
    async for event in streaming_workflow.astream(initial_state):
        # The event dictionary's key is the name of the node that just finished.
        if "research" in event:
            print("✅ RESEARCH TASK COMPLETE")
            data = event["research"]["research_data"]
            print(f"   -> Research data received ({len(data)} characters).")
            print("-" * 30)

        if "summarize" in event:
            print("✅ SUMMARIZE TASK COMPLETE")
            summary = event["summarize"]["summary"]
            print(f"   -> Summary received: '{summary[:100]}...'")
            print("-" * 30)

    print("🏁 Workflow stream finished.")
    print("\nPro Tip: Try running this from the CLI for a rich visual trace:")
    print("agentum run examples/15_streaming_workflow.py --stream")


if __name__ == "__main__":
    asyncio.run(main())
