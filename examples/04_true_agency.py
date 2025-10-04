from dotenv import load_dotenv

from agentum import Agent, GoogleLLM, State, Workflow, tool
from agentum.core.config import settings

load_dotenv()


@tool
def get_weather(city: str) -> str:
    if city.lower() == "san francisco":
        return "It's 65°F and foggy."
    elif city.lower() == "tokyo":
        return "It's 82°F and sunny."
    else:
        return f"Sorry, I don't have the weather for {city}."


class TravelState(State):
    request: str
    plan: str = ""


travel_agent = Agent(
    name="TravelPlanner",
    system_prompt="You are a helpful travel assistant. You create simple, bulleted travel plans. When creating packing recommendations, always check the weather for the destination to provide accurate advice.",
    llm=GoogleLLM(api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"),
    tools=[get_weather],
)
travel_workflow = Workflow(name="Travel_Planner", state=TravelState)
travel_workflow.add_task(
    name="create_plan",
    agent=travel_agent,
    instructions="Create a travel plan based on the user's request: {request}",
    output_mapping={"plan": "output"},
)
travel_workflow.set_entry_point("create_plan")
travel_workflow.add_edge("create_plan", travel_workflow.END)
if __name__ == "__main__":
    initial_state = {
        "request": "I'm thinking of visiting San Francisco. What should I pack?"
    }
    final_state = travel_workflow.run(initial_state)
    print("\n--- Final Travel Plan ---")
    print(final_state["plan"])
    if "foggy" in final_state["plan"].lower() or "65" in final_state["plan"]:
        print("\n✅ Agent successfully used the weather tool to complete its task!")
    else:
        print("\n⚠️  Agent provided a plan but may not have used the weather tool.")
        print(
            "This demonstrates the autonomous nature - agents decide when to use tools."
        )
