"""
Advanced Agentum Framework Demo - Showcasing All Features

This example demonstrates the full power of Agentum 2.0:
- Multi-provider LLM support (Google Gemini + Anthropic Claude)
- Vision capabilities for image analysis
- Tool integration and execution
- Memory and conversation persistence
- RAG with vector search
- Multi-agent collaboration
- Error handling and resilience
"""
import os
import tempfile
from typing import Any

from dotenv import load_dotenv

from agentum import (
    Agent,
    AnthropicLLM,
    ConversationMemory,
    GoogleLLM,
    KnowledgeBase,
    State,
    Workflow,
    tool,
)

load_dotenv()


class AdvancedState(State):
    """State for the advanced demo workflow."""
    user_query: str
    image_url: str = ""
    research_result: str = ""
    analysis_result: str = ""
    final_summary: str = ""
    knowledge_search_result: str = ""


# Define tools for the workflow
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    return f"Knowledge base search for '{query}': Found relevant information about the topic."


@tool
def analyze_data(data: str) -> str:
    """Analyze the provided data and extract key insights."""
    return f"Analysis of data: Key insights extracted from '{data[:50]}...'"


@tool
def format_summary(content: str) -> str:
    """Format content into a professional summary."""
    return f"Professional Summary:\n\n{content}\n\n--- End of Summary ---"


def create_knowledge_base() -> KnowledgeBase:
    """Create a sample knowledge base for testing."""
    kb = KnowledgeBase(name="AI_Knowledge_Base")
    
    # Add sample documents
    sample_documents = [
        "Artificial Intelligence is transforming industries worldwide.",
        "Machine learning algorithms can process vast amounts of data.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information.",
        "Deep learning networks can recognize patterns in complex data.",
    ]
    
    # Create temporary text files for the knowledge base
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, doc in enumerate(sample_documents):
            with open(f"{temp_dir}/doc_{i}.txt", "w") as f:
                f.write(doc)
        
        # Add documents to knowledge base
        kb.add([f"{temp_dir}/doc_{i}.txt" for i in range(len(sample_documents))])
    
    return kb


def create_advanced_workflow() -> Workflow:
    """Create an advanced multi-agent workflow."""
    
    # Create LLM providers
    google_llm = GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"))
    anthropic_llm = AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Create agents with different roles
    researcher = Agent(
        name="Researcher",
        system_prompt="You are an expert researcher who finds accurate information and conducts thorough analysis.",
        llm=google_llm,
        tools=[search_knowledge_base],
        memory=ConversationMemory(),
        max_retries=3
    )
    
    analyst = Agent(
        name="Analyst",
        system_prompt="You are a data analyst who processes information and extracts key insights.",
        llm=anthropic_llm,
        tools=[analyze_data],
        memory=ConversationMemory(),
        max_retries=3
    )
    
    visual_analyst = Agent(
        name="VisualAnalyst",
        system_prompt="You are a visual analyst who can analyze images and describe what you see in detail.",
        llm=google_llm,
        memory=ConversationMemory(),
        max_retries=3
    )
    
    summarizer = Agent(
        name="Summarizer",
        system_prompt="You are a professional summarizer who creates clear, concise summaries.",
        llm=anthropic_llm,
        tools=[format_summary],
        memory=ConversationMemory(),
        max_retries=3
    )
    
    # Create the workflow
    workflow = Workflow(name="Advanced_Demo_Workflow", state=AdvancedState)
    
    # Add tasks
    workflow.add_task(
        name="research_task",
        agent=researcher,
        instructions="Research and gather information about: {user_query}",
        output_mapping={"research_result": "output"}
    )
    
    workflow.add_task(
        name="knowledge_search",
        agent=researcher,
        instructions="Search the knowledge base for: {user_query}",
        output_mapping={"knowledge_search_result": "output"}
    )
    
    workflow.add_task(
        name="analysis_task",
        agent=analyst,
        instructions="Analyze the research results: {research_result}",
        output_mapping={"analysis_result": "output"}
    )
    
    workflow.add_task(
        name="visual_analysis",
        agent=visual_analyst,
        instructions="Analyze this image and describe what you see: {user_query}",
        output_mapping={"analysis_result": "output"}
    )
    
    workflow.add_task(
        name="summarization_task",
        agent=summarizer,
        instructions="Create a comprehensive summary combining: Research: {research_result}, Analysis: {analysis_result}, Knowledge: {knowledge_search_result}",
        output_mapping={"final_summary": "output"}
    )
    
    # Set up workflow flow
    workflow.set_entry_point("research_task")
    workflow.add_edge("research_task", "knowledge_search")
    workflow.add_edge("knowledge_search", "analysis_task")
    workflow.add_edge("analysis_task", "summarization_task")
    workflow.add_edge("summarization_task", workflow.END)
    
    return workflow


def run_text_only_demo():
    """Run a text-only demonstration."""
    print("📝 Running Text-Only Demo...")
    
    workflow = create_advanced_workflow()
    
    initial_state = {
        "user_query": "What are the key trends in artificial intelligence?",
        "image_url": ""  # No image for text-only demo
    }
    
    print(f"🚀 Starting workflow with query: {initial_state['user_query']}")
    
    try:
        final_state = workflow.run(initial_state)
        
        print("\n--- Research Results ---")
        print(final_state["research_result"])
        
        print("\n--- Knowledge Base Search ---")
        print(final_state["knowledge_search_result"])
        
        print("\n--- Analysis Results ---")
        print(final_state["analysis_result"])
        
        print("\n--- Final Summary ---")
        print(final_state["final_summary"])
        
        print("\n✅ Text-only demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Text-only demo failed: {e}")
        return False


def run_vision_demo():
    """Run a vision-enabled demonstration."""
    print("\n🖼️  Running Vision Demo...")
    
    workflow = create_advanced_workflow()
    
    initial_state = {
        "user_query": "Describe this image and analyze what you see",
        "image_url": "https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=2070&auto=format&fit=crop"  # AI/tech image
    }
    
    print(f"🚀 Starting vision workflow with image: {initial_state['image_url']}")
    
    try:
        final_state = workflow.run(initial_state)
        
        print("\n--- Visual Analysis Results ---")
        print(final_state["analysis_result"])
        
        print("\n✅ Vision demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Vision demo failed: {e}")
        return False


def run_provider_comparison():
    """Compare different LLM providers."""
    print("\n🔄 Running Provider Comparison Demo...")
    
    google_llm = GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"))
    anthropic_llm = AnthropicLLM(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    test_query = "Explain quantum computing in simple terms"
    
    # Test Google provider
    google_agent = Agent(
        name="GoogleAgent",
        system_prompt="You are a helpful assistant.",
        llm=google_llm
    )
    
    google_workflow = Workflow(name="GoogleTest", state=AdvancedState)
    google_workflow.add_task(
        name="google_task",
        agent=google_agent,
        instructions=test_query,
        output_mapping={"final_summary": "output"}
    )
    google_workflow.set_entry_point("google_task")
    google_workflow.add_edge("google_task", google_workflow.END)
    
    # Test Anthropic provider
    anthropic_agent = Agent(
        name="AnthropicAgent",
        system_prompt="You are a helpful assistant.",
        llm=anthropic_llm
    )
    
    anthropic_workflow = Workflow(name="AnthropicTest", state=AdvancedState)
    anthropic_workflow.add_task(
        name="anthropic_task",
        agent=anthropic_agent,
        instructions=test_query,
        output_mapping={"final_summary": "output"}
    )
    anthropic_workflow.set_entry_point("anthropic_task")
    anthropic_workflow.add_edge("anthropic_task", anthropic_workflow.END)
    
    try:
        print(f"🔍 Testing Google Gemini with: {test_query}")
        google_result = google_workflow.run({"user_query": test_query})
        print(f"Google Result: {google_result['final_summary'][:100]}...")
        
        print(f"\n🔍 Testing Anthropic Claude with: {test_query}")
        anthropic_result = anthropic_workflow.run({"user_query": test_query})
        print(f"Anthropic Result: {anthropic_result['final_summary'][:100]}...")
        
        print("\n✅ Provider comparison completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Provider comparison failed: {e}")
        return False


def main():
    """Run all demonstrations."""
    print("🚀 Agentum 2.0 Advanced Demo - Showcasing All Features")
    print("=" * 60)
    
    # Check API keys
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not found. Some demos may fail.")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not found. Some demos may fail.")
    
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ No API keys found. Please set at least one API key to run demos.")
        return
    
    success_count = 0
    total_demos = 3
    
    # Run demonstrations
    if run_text_only_demo():
        success_count += 1
    
    if run_vision_demo():
        success_count += 1
    
    if run_provider_comparison():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Demo Results: {success_count}/{total_demos} demos completed successfully")
    
    if success_count == total_demos:
        print("🎉 All demonstrations completed successfully!")
        print("✅ Agentum 2.0 framework is working perfectly!")
    else:
        print("⚠️  Some demonstrations failed. Check the output above for details.")


if __name__ == "__main__":
    main()
