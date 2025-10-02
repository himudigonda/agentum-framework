"""
Demonstrates Agentum's extensible provider architecture.
Switch between different LLM providers with a single line change.

To run this example:
1. Make sure you have the necessary API key set in your .env file
   (e.g., GOOGLE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY).
2. Uncomment the provider you wish to use.
"""

from dotenv import load_dotenv

from agentum import Agent, AnthropicLLM, GoogleLLM, OpenAILLM, State, Workflow
from agentum.config import settings  # MODIFICATION: Import settings

load_dotenv()


class StoryState(State):
    topic: str
    story: str = ""


# Choose your LLM Provider
# ---------------------------

# Option A: Use Google Gemini (Default)
llm_provider = GoogleLLM(
    # MODIFICATION: Use the centralized settings object
    api_key=settings.GOOGLE_API_KEY,
    temperature=0.8,
    model="gemini-2.5-flash-lite",
)

# Option B: Use Anthropic Claude 3.5 Sonnet (uncomment to switch)
# llm_provider = AnthropicLLM(
#     # MODIFICATION: Use the centralized settings object
#     api_key=settings.ANTHROPIC_API_KEY,
#     temperature=0.8,
# )

# Option C: Use OpenAI GPT-4o-mini (uncomment to switch)
# llm_provider = OpenAILLM(
#     # MODIFICATION: Use the centralized settings object
#     api_key=settings.OPENAI_API_KEY,
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
