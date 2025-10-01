# examples/05_testing_and_evaluation.py
import asyncio
import os

from dotenv import load_dotenv

from agentum import Agent, ConversationMemory, GoogleLLM, State, Workflow, tool
from agentum.testing import Evaluator, TestSuite

load_dotenv()


# 1. Define a tool for weather information
@tool
def get_weather(city: str) -> str:
    """Provides the current weather for a specified city."""
    weather_data = {
        "san francisco": "It's 65°F and foggy with light winds.",
        "tokyo": "It's 82°F and sunny with high humidity.",
        "london": "It's 55°F and rainy with overcast skies.",
        "paris": "It's 70°F and partly cloudy.",
    }
    return weather_data.get(
        city.lower(), f"Sorry, I don't have weather data for {city}."
    )


# 2. Define the State
class TravelState(State):
    request: str
    plan: str = ""


# 3. Define the Agent with Memory
travel_agent = Agent(
    name="TravelPlanner",
    system_prompt="You are a helpful travel assistant. You create detailed, bulleted travel plans. When creating packing recommendations, always check the weather for the destination to provide accurate advice.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[get_weather],
    memory=ConversationMemory(),  # Enable memory for stateful conversations
)

# 4. Define the Workflow
travel_workflow = Workflow(name="Travel_Planner", state=TravelState)

travel_workflow.add_task(
    name="create_plan",
    agent=travel_agent,
    instructions="Create a travel plan based on the user's request: {request}",
    output_mapping={"plan": "output"},
)

travel_workflow.set_entry_point("create_plan")
travel_workflow.add_edge("create_plan", travel_workflow.END)

# 5. Define Evaluators
quality_evaluator = Evaluator(
    name="Quality",
    evaluator_agent=Agent(
        name="QualityEvaluator",
        system_prompt="You are a quality evaluator for travel plans. Rate the quality of travel plans on a scale of 1-10.",
        llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    ),
    instructions="Rate the quality of this travel plan on a scale of 1-10. Consider completeness, practicality, and weather awareness. Travel plan: {plan}",
)

weather_evaluator = Evaluator(
    name="Weather_Awareness",
    evaluator_agent=Agent(
        name="WeatherEvaluator",
        system_prompt="You are a weather awareness evaluator. Check if travel plans properly consider weather conditions.",
        llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    ),
    instructions="Does this travel plan properly consider weather conditions? Answer 'YES' if weather is mentioned and incorporated, 'NO' otherwise. Travel plan: {plan}",
)

# 6. Define Test Dataset
test_dataset = [
    {"request": "I'm thinking of visiting San Francisco. What should I pack?"},
    {"request": "Planning a trip to Tokyo next month. What do I need to bring?"},
    {"request": "Going to London for business. What should I pack for the weather?"},
]


# 7. Create and Run Test Suite
async def main():
    test_suite = TestSuite(
        workflow=travel_workflow,
        dataset=test_dataset,
        evaluators=[quality_evaluator, weather_evaluator],
    )

    # Add event listeners for observability
    @travel_workflow.on("agent_tool_call")
    async def on_tool_call(tool_name: str, tool_args: dict):
        print(f"🔧 Agent called tool: {tool_name} with args: {tool_args}")

    @travel_workflow.on("agent_tool_result")
    async def on_tool_result(tool_name: str, result: str):
        print(f"📊 Tool {tool_name} returned: {result}")

    @travel_workflow.on("agent_end")
    async def on_agent_end(agent_name: str, final_response: str):
        print(
            f"✅ Agent {agent_name} finished with response length: {len(final_response)} chars"
        )

    results = await test_suite.arun()
    test_suite.summary(results)

    print("\n🎯 Testing completed! Check the summary table above for results.")


if __name__ == "__main__":
    asyncio.run(main())
