# examples/02_test_without_api.py
from pydantic import Field

from agentum import Agent, State, Workflow, tool


# 1. Define the state model
class TestState(State):
    input_text: str
    process_text: str = Field(default="")
    save_result: str = Field(default="")


# 2. Define a tool
@tool
def save_text(text: str):
    """A simple tool to save text."""
    print(f"\n--- TOOL EXECUTING ---")
    print(f"Saving text: '{text[:50]}...'")
    print("--- TOOL FINISHED ---\n")
    return f"Text saved successfully: {text[:20]}..."


# 3. Create a mock LLM for testing
class MockLLM:
    def invoke(self, prompt: str):
        class MockResponse:
            content = f"Mock response to: {prompt[:50]}..."

        return MockResponse()


# 4. Define the workflow
test_workflow = Workflow(name="Test_Pipeline", state=TestState)

# Create a mock agent
mock_agent = Agent(
    name="MockAgent",
    system_prompt="You are a helpful assistant.",
    llm=MockLLM(),
)

test_workflow.add_task(
    name="process_text",
    agent=mock_agent,
    instructions="Process this text: {input_text}",
)
test_workflow.add_task(
    name="save_result",
    tool=save_text,
    inputs={"text": "{process_text}"},
)

test_workflow.set_entry_point("process_text")
test_workflow.add_edge("process_text", "save_result")
test_workflow.add_edge("save_result", test_workflow.END)

# 5. Run the workflow
if __name__ == "__main__":
    initial_state = {"input_text": "Hello, Agentum!"}

    # This will test the compilation and execution without requiring an API key
    final_state = test_workflow.run(initial_state)

    print("\nFinal State:")
    print(final_state)
