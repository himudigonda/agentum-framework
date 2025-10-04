"""
Flagship Example: Advanced Agent Memory with Semantic Search (VectorStoreMemory)

This example demonstrates the power of VectorStoreMemory, which uses a vector store
to give the agent true long-term memory via semantic retrieval.

Key Learning Points:
1.  Agent remembers facts from prior, unconnected runs.
2.  Memory retrieval is semantic (based on meaning), not just keyword-based.
3.  The workflow is clean; the complexity is hidden in the memory module.
"""

import asyncio

from dotenv import load_dotenv

from agentum import Agent, GoogleLLM, State, Workflow
from agentum.core.config import settings
from agentum.memory import VectorStoreMemory

load_dotenv()


class ChatState(State):
    request: str
    response: str = ""


personal_assistant = Agent(
    name="PersonalAssistant",
    system_prompt="You are a personal assistant. You must use your memory to recall past information about the user.",
    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"),
    memory=VectorStoreMemory(),
)
memory_workflow = Workflow(name="Advanced_Memory_Assistant", state=ChatState)
memory_workflow.add_task(
    name="chat_task",
    agent=personal_assistant,
    instructions="User: {request}",
    output_mapping={"response": "output"},
)
memory_workflow.set_entry_point("chat_task")
memory_workflow.add_edge("chat_task", memory_workflow.END)


async def main():
    print("--- 1. First Run: Agent learns a fact ---")
    request_1 = "I am a Python developer and my favorite framework is Agentum."
    result_1 = await memory_workflow.arun({"request": request_1})
    print(f"Agent's response (1): {result_1['response']}")
    print("\n--- 2. Second Run: Agent handles unrelated query ---")
    request_2 = "What are the three most recent developments in AI ethics?"
    result_2 = await memory_workflow.arun({"request": request_2})
    print(f"Agent's response (2): {result_2['response']}")
    print("\n--- 3. Third Run: Semantic Recall Test (Across runs) ---")
    request_3 = "What is my primary job role and my preferred library?"
    result_3 = await memory_workflow.arun({"request": request_3})
    print(f"Agent's response (3): {result_3['response']}")
    if (
        "python developer" in result_3["response"].lower()
        and "agentum" in result_3["response"].lower()
    ):
        print("\n✅ SUCCESS: Agent demonstrated long-term, semantic memory!")
    else:
        print("\n⚠️  WARNING: Agent failed to recall the information semantically.")


if __name__ == "__main__":
    asyncio.run(main())
