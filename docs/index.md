# Agentum Framework

A Python-native framework for building, orchestrating, and observing production-ready, multi-modal AI agent systems.

## 🚀 Quick Start

```python
from agentum import Agent, State, Workflow, GoogleLLM, search_web_tavily

# Define state
class ResearchState(State):
    topic: str
    research: str = ""
    report: str = ""

# Create agents
researcher = Agent(
    name="Researcher",
    system_prompt="You are an expert researcher.",
    llm=GoogleLLM(api_key="your-api-key"),
    tools=[search_web_tavily]
)

writer = Agent(
    name="Writer", 
    system_prompt="You are a professional writer.",
    llm=GoogleLLM(api_key="your-api-key")
)

# Create workflow
workflow = Workflow(name="Research_Pipeline", state=ResearchState)
workflow.add_task(
    name="research",
    agent=researcher,
    instructions="Research: {topic}",
    output_mapping={"research": "output"}
)
workflow.add_task(
    name="write",
    agent=writer,
    instructions="Write report: {research}",
    output_mapping={"report": "output"}
)

workflow.set_entry_point("research")
workflow.add_edge("research", "write")

# Run it
result = workflow.run({"topic": "The future of AI"})
print(result["report"])
```

## 📚 Documentation

### Getting Started
- [Getting Started Guide](getting_started.md) - Build your first multi-agent workflow with real tools

### Core Features
- [LLM Providers](features/providers.md) - Switch between Google, Anthropic, and OpenAI models
- [Multi-Modal Capabilities](features/multi_modality.md) - Add vision support for images and local files
- [Advanced RAG](features/advanced_rag.md) - Use rerankers for high-precision document retrieval
- [Speech Capabilities](features/speech.md) - Build voice assistants with STT and TTS

### Core Concepts
- [Concepts](concepts/) - Deep dive into Agents, State, and Workflows
- [Examples](../examples/) - Explore 14 real-world examples showcasing all features

## 🎯 Key Features

- **🤖 Multi-Modal Agents**: Build agents that can see, hear, and speak
- **🔄 Multi-Provider LLMs**: Swap between Google Gemini, Anthropic Claude, and OpenAI GPT
- **🗣️ End-to-End Voice Workflows**: Complete voice assistants that listen, think, and speak
- **📚 Advanced RAG**: KnowledgeBase with cross-encoder reranking for high-precision retrieval
- **🔍 Deep Observability**: Event-driven system for comprehensive debugging
- **🧪 Professional Testing**: TestSuite and Evaluator for systematic quality assurance
- **🛠️ Powerful CLI**: Validate, visualize, and run workflows from the terminal

## 🏗️ Architecture

Agentum is built on top of LangGraph and provides:

- **Workflow**: Declarative workflow definition with multi-agent orchestration
- **Agent**: AI agents with tools, memory, and multi-modal capabilities
- **State**: Type-safe state management with multi-modal data support
- **Events**: Comprehensive observability and debugging
- **Testing**: Built-in evaluation framework with mock LLMs
- **Providers**: Extensible LLM provider architecture

## 🚀 Examples

### Basic Examples
- [Hello Agentum](../examples/01_hello_agentum.py) - Simple agent workflow
- [Tool Usage](../examples/02_simple_tool_usage.py) - Agent with real web search
- [Conditional Logic](../examples/03_conditional_loop.py) - Complex workflow control

### Advanced Examples
- [True Agency](../examples/04_true_agency.py) - Autonomous tool usage
- [Testing & Evaluation](../examples/05_testing_and_evaluation.py) - Quality assurance
- [Advanced RAG](../examples/06_advanced_rag.py) - Document retrieval
- [Professional Research](../examples/07_professional_research.py) - Multi-agent research

### Multi-Provider Examples
- [Multiple LLMs](../examples/08_multiple_llms.py) - Switch between providers
- [Vision Analysis](../examples/09_vision_analysis.py) - Image analysis with URLs
- [Advanced Demo](../examples/10_advanced_demo.py) - Comprehensive showcase

### Speech Examples
- [Speech-to-Text](../examples/11_speech_to_text.py) - Audio transcription
- [Voice Assistant](../examples/12_voice_assistant.py) - Complete voice workflow

### Advanced Features
- [RAG with Reranking](../examples/13_advanced_rag_reranking.py) - High-precision retrieval
- [Local Image Analysis](../examples/14_local_image_analysis.py) - Local file processing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.
