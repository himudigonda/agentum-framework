# examples/03_conditional_loop.py
from pydantic import Field

from agentum import Agent, State, Workflow


# --- Mock LLM for predictable testing ---
class MockCriticLLM:
    _call_count = 0

    def invoke(self, prompt: str):
        class MockResponse:
            content = ""

        MockCriticLLM._call_count += 1
        print(f"MockCriticLLM call #{MockCriticLLM._call_count}")

        # Simple logic: approve after 3 calls
        if MockCriticLLM._call_count >= 3:
            MockResponse.content = "APPROVED"
            print("Returning APPROVED")
        else:
            MockResponse.content = (
                f"Feedback #{MockCriticLLM._call_count}: It needs more detail."
            )
            print(f"Returning feedback: {MockResponse.content}")

        return MockResponse()

    async def ainvoke(self, prompt: str):
        # For async compatibility, we can just call the sync version
        return self.invoke(prompt)


# 1. Define the state
class EditorState(State):
    draft: str
    review: str = ""
    revision_count: int = 0
    critique_result: str = ""  # Add this field to store the critique result


# 2. Define the agents
editor = Agent(name="Editor", system_prompt="You are an editor.", llm=MockCriticLLM())
critic = Agent(name="Critic", system_prompt="You are a critic.", llm=MockCriticLLM())

# 3. Define the workflow
editing_workflow = Workflow(name="Revision_Loop_Pipeline", state=EditorState)

editing_workflow.add_task(
    name="edit_draft",
    agent=editor,
    instructions="Please revise this draft: {draft}\nBased on feedback: {review}",
)
editing_workflow.add_task(
    name="critique_draft",
    agent=critic,
    instructions="Is this draft good enough? {draft}",
)

# 4. Define the control flow with a loop
editing_workflow.set_entry_point("critique_draft")
editing_workflow.add_edge("edit_draft", "critique_draft")


# This is our conditional logic!
def check_approval(state: EditorState) -> str:
    # Access the critique_result from the state
    critique_result = getattr(state, "critique_result", "")

    print(f"DEBUG: critique_result = '{critique_result}'")

    if "APPROVED" in critique_result:
        print("--- Reviewer approved! Ending loop. ---")
        return "end"
    else:
        print("--- Revisions needed. Looping back. ---")
        return "revise"


editing_workflow.add_conditional_edges(
    source="critique_draft",
    path=check_approval,
    paths={"end": editing_workflow.END, "revise": "edit_draft"},
)

# 5. Run it
if __name__ == "__main__":
    initial_state = {"draft": "This is the first version of the article."}

    # The sync 'run' method will handle the async execution for us.
    final_state = editing_workflow.run(initial_state)

    print("\nFinal State:")
    print(final_state)
    assert "APPROVED" in final_state["critique_result"]
    print("✅ Conditional loop example completed successfully!")
