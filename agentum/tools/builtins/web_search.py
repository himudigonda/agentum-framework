from tavily import TavilyClient
from agentum import tool
from agentum.config import settings

@tool
def search_web_tavily(query: str) -> str:
    try:
        api_key = settings.TAVILY_API_KEY
        if not api_key:
            return 'Error: TAVILY_API_KEY environment variable is not set.'
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth='advanced')
        formatted_results = [f"Source URL: {res['url']}\nContent: {res['content']}" for res in response['results']]
        return '\n\n---\n\n'.join(formatted_results)
    except Exception as e:
        return f'Error performing web search: {e}'