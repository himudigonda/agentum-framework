# examples/02_simple_tool_usage.py
import os

from dotenv import load_dotenv
from pydantic import Field

from agentum import Agent, GoogleLLM, State, Workflow, tool

load_dotenv()


# 1. Define the state model
class NewsState(State):
    company_name: str
    news_summary: str = Field(default="")
    sentiment: str = Field(default="")


# 2. Define a "real" tool
@tool
def get_latest_news(company: str) -> str:
    """A tool to fetch the latest news for a company."""
    print(f"\n--- TOOL: Fetching news for {company} ---")
    # This is a mock, but it simulates a real tool call.
    return f"Shares for {company} surged today after they announced a breakthrough in AI research."


# 3. Define agents
news_researcher = Agent(
    name="NewsResearcher",
    system_prompt="You are a financial news researcher.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[get_latest_news],  # Agent now knows about this tool
)

sentiment_analyst = Agent(
    name="SentimentAnalyst",
    system_prompt="You are a sentiment analysis expert. Analyze the text and respond with only 'Positive', 'Negative', or 'Neutral'.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
)

# 4. Define the workflow
news_workflow = Workflow(name="News_Analysis_Pipeline", state=NewsState)

news_workflow.add_task(
    name="fetch_news",
    # This is a tool-calling task now, where the agent decides which tool to use.
    # We will implement the logic for this in the next steps.
    # For now, we will assume it calls the tool directly.
    tool=get_latest_news,
    inputs={"company": "{company_name}"},
    output_mapping={"news_summary": "output"},
)
news_workflow.add_task(
    name="analyze_sentiment",
    agent=sentiment_analyst,
    instructions="Analyze the sentiment of this news: {news_summary}",
    output_mapping={"sentiment": "output"},
)

news_workflow.set_entry_point("fetch_news")
news_workflow.add_edge("fetch_news", "analyze_sentiment")
news_workflow.add_edge("analyze_sentiment", news_workflow.END)

# 5. Run it
if __name__ == "__main__":
    initial_state = {"company_name": "InnovateCorp"}
    final_state = news_workflow.run(initial_state)

    print("\nFinal State:")
    print(final_state)
