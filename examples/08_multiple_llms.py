"""
Demonstrates Agentum's extensible provider architecture.
Switch between different LLM providers with a single line change.
"""

import os

from dotenv import load_dotenv

from agentum import Agent, AnthropicLLM, GoogleLLM, State, Workflow

load_dotenv()


class StoryState(State):
    topic: str
    story: str = ""


# Choose your LLM Provider
llm_provider = AnthropicLLM(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.8,
)

# Alternative: Use Google Gemini 2.5 Flash
# llm_provider = GoogleLLM(
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.8,
# )

storyteller = Agent(
    name="Storyteller",
    system_prompt="You are a master storyteller. You write a short, captivating story (3-4 paragraphs) based on a topic.",
    llm=llm_provider,
)

story_workflow = Workflow(name="Story_Writing_Workflow", state=StoryState)

story_workflow.add_task(
    name="write_story",
    agent=storyteller,
    instructions="Write a story about: {topic}",
    output_mapping={"story": "output"},
)

story_workflow.set_entry_point("write_story")
story_workflow.add_edge("write_story", story_workflow.END)

if __name__ == "__main__":
    initial_state = {"topic": "a lost astronaut who finds a mysterious alien artifact"}

    print(f"🚀 Starting story generation with: {llm_provider.__class__.__name__}")

    final_state = story_workflow.run(initial_state)

    print("\n--- Generated Story ---")
    print(final_state["story"])
    print("\n✅ Workflow complete.")
