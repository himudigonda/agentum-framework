# examples/02_simple_tool_usage.py
import os

from dotenv import load_dotenv
from pydantic import Field

from agentum import Agent, GoogleLLM, State, Workflow, search_web_tavily
from agentum.config import settings  # MODIFICATION: Import settings

load_dotenv()


class NewsState(State):
    company_name: str
    news_summary: str = Field(default="")
    sentiment: str = Field(default="")


news_researcher = Agent(
    name="NewsResearcher",
    system_prompt="You are a financial news researcher. Your goal is to find the most relevant, recent news article about a company.",
    llm=GoogleLLM(
        api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"
    ),  # MODIFICATION: Use settings
    tools=[search_web_tavily],
)

sentiment_analyst = Agent(
    name="SentimentAnalyst",
    system_prompt="You are a sentiment analysis expert. Analyze the text and respond with only 'Positive', 'Negative', or 'Neutral'.",
    llm=GoogleLLM(
        api_key=settings.GOOGLE_API_KEY, model="gemini-2.5-flash-lite"
    ),  # MODIFICATION: Use settings
)

news_workflow = Workflow(name="News_Analysis_Pipeline", state=NewsState)

news_workflow.add_task(
    name="fetch_news",
    agent=news_researcher,
    instructions="Use your tools to find the latest news for the company: {company_name}",
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

if __name__ == "__main__":
    initial_state = {"company_name": "NVIDIA"}
    print("🚀 Running Real-World News Analysis Workflow...")
    final_state = news_workflow.run(initial_state)

    print("\n--- Final State ---")
    print(f"Company: {final_state['company_name']}")
    print(f"News Summary (first 100 chars): {final_state['news_summary'][:100]}...")
    print(f"Sentiment: {final_state['sentiment']}")
    print("\n✅ Real-world tool workflow completed successfully!")
