# examples/04_true_agency.py
import os

from dotenv import load_dotenv

from agentum import Agent, GoogleLLM, State, Workflow, tool

load_dotenv()


# 1. Define a tool that the agent can choose to use
@tool
def get_weather(city: str) -> str:
    """Provides the current weather for a specified city."""
    if city.lower() == "san francisco":
        return "It's 65°F and foggy."
    elif city.lower() == "tokyo":
        return "It's 82°F and sunny."
    else:
        return f"Sorry, I don't have the weather for {city}."


# 2. Define the State
class TravelState(State):
    request: str
    plan: str = ""


# 3. Define the Agent
# This agent has a goal and a tool. It must figure out how to use the tool to achieve the goal.
travel_agent = Agent(
    name="TravelPlanner",
    system_prompt="You are a helpful travel assistant. You create simple, bulleted travel plans. When creating packing recommendations, always check the weather for the destination to provide accurate advice.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[get_weather],  # Crucially, the agent is GIVEN the tool.
)

# 4. Define the Workflow
travel_workflow = Workflow(name="Travel_Planner", state=TravelState)

# The instructions are high-level. They do NOT tell the agent to use the tool.
# The agent must INFER that it needs to call get_weather.
travel_workflow.add_task(
    name="create_plan",
    agent=travel_agent,
    instructions="Create a travel plan based on the user's request: {request}",
    output_mapping={"plan": "output"},
)

travel_workflow.set_entry_point("create_plan")
travel_workflow.add_edge("create_plan", travel_workflow.END)

# 5. Run it
if __name__ == "__main__":
    initial_state = {
        "request": "I'm thinking of visiting San Francisco. What should I pack?"
    }

    final_state = travel_workflow.run(initial_state)

    print("\n--- Final Travel Plan ---")
    print(final_state["plan"])
    # We can assert that the plan includes weather details, proving the tool was called.
    assert "foggy" in final_state["plan"].lower()
    print("\n✅ Agent successfully used a tool to complete its task!")
