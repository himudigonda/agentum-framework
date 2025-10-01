# Agentum Framework

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/agentum.svg)](https://badge.fury.io/py/agentum)

**A Python-native framework for building, orchestrating, and observing multi-agent systems with RAG capabilities.**

Agentum empowers developers to create sophisticated AI workflows that can reason, act, and learn autonomously. Built on top of LangGraph and LangChain, it provides a clean, intuitive API for orchestrating complex multi-agent systems with built-in observability, memory, and retrieval-augmented generation (RAG) capabilities.

## 🚀 Key Features

### 🤖 **True Agentic Behavior**
- **Autonomous Tool Usage**: Agents decide when and how to use tools
- **Multi-Step Reasoning**: Think → Act → Observe → Think again loops
- **Tool Binding**: Automatic LLM tool integration with schema generation

### 🔍 **Complete Observability**
- **Event-Driven Debugging**: Hook into every step of agent execution
- **Rich Logging**: Beautiful console output with Rich
- **Real-time Monitoring**: Stream workflow execution as it happens

### 🧠 **Stateful Conversations**
- **Memory Management**: Persistent conversation history across interactions
- **Context Awareness**: Agents remember previous conversations
- **Multi-Turn Interactions**: Build conversational AI applications

### 📚 **RAG Integration**
- **Knowledge Bases**: Easy document ingestion and vector storage
- **Semantic Search**: Built-in vector search capabilities
- **Document Processing**: Support for PDFs, text files, and web content

### 🧪 **Professional Testing**
- **Automated Evaluation**: Agent-based testing and evaluation
- **Regression Testing**: Ensure agent behavior consistency
- **Test Suites**: Systematic testing of complex workflows

### 🛡️ **Production Ready**
- **Automatic Retries**: Resilient to transient failures
- **Error Handling**: Comprehensive error management
- **State Persistence**: Redis-based checkpointing for long-running workflows
- **CLI Tools**: Complete command-line interface

## 📦 Installation

```bash
pip install agentum
```

## 🎯 Quick Start

### Basic Workflow

```python
from agentum import Agent, State, Workflow, GoogleLLM
import os

# Define your state
class ResearchState(State):
    topic: str
    summary: str = ""

# Create an agent
researcher = Agent(
    name="Researcher",
    system_prompt="You are a helpful research assistant.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"))
)

# Build the workflow
workflow = Workflow(name="Research_Pipeline", state=ResearchState)

workflow.add_task(
    name="research",
    agent=researcher,
    instructions="Research the topic: {topic}",
    output_mapping={"summary": "output"}
)

workflow.set_entry_point("research")
workflow.add_edge("research", workflow.END)

# Run it
result = workflow.run({"topic": "The Future of AI"})
print(result["summary"])
```

### RAG-Powered Agent

```python
from agentum import Agent, State, Workflow, GoogleLLM, KnowledgeBase, create_vector_search_tool

# Create a knowledge base
kb = KnowledgeBase(name="company_docs")
kb.add(sources=["documents/report.pdf", "documents/policies.txt"])

# Create a RAG-enabled agent
vector_search_tool = create_vector_search_tool(kb)
analyst = Agent(
    name="DocumentAnalyst",
    system_prompt="Answer questions using only the provided documents.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[vector_search_tool]
)

# Build RAG workflow
workflow = Workflow(name="Document_QA", state=State)
workflow.add_task(
    name="answer",
    agent=analyst,
    instructions="Answer: {question}",
    output_mapping={"answer": "output"}
)
workflow.set_entry_point("answer")
workflow.add_edge("answer", workflow.END)

# Query your documents
result = workflow.run({"question": "What are our company policies?"})
```

### Observability & Testing

```python
from agentum import Agent, State, Workflow, GoogleLLM, TestSuite, Evaluator

# Set up event listeners
@workflow.on("agent_tool_call")
async def on_tool_call(tool_name: str, tool_args: dict):
    print(f"🔧 Agent called tool: {tool_name} with args: {tool_args}")

# Create evaluators
quality_evaluator = Evaluator(
    name="Quality",
    evaluator_agent=Agent(name="QualityJudge", llm=llm),
    instructions="Rate the quality of this response: {output}"
)

# Run test suite
test_suite = TestSuite(workflow, test_dataset, [quality_evaluator])
results = await test_suite.arun()
test_suite.summary(results)
```

## 🛠️ CLI Tools

Agentum comes with a powerful CLI for workflow management:

```bash
# Run a workflow
agentum run my_workflow.py --state '{"input": "Hello World"}'

# Stream execution in real-time
agentum run my_workflow.py --stream

# Validate workflow without running
agentum validate my_workflow.py

# Generate workflow visualization
agentum graph my_workflow.py --output workflow.png

# Initialize new project
agentum init MyProject --output ./projects/

# Check version
agentum version
```

## 📖 Documentation

- **[Getting Started Guide](docs/getting_started.md)** - Complete tutorial for new users
- **[Concepts](docs/concepts/)** - Core concepts and architecture
- **[Features](docs/features/)** - Detailed feature documentation
- **[Examples](examples/)** - Comprehensive example workflows

## 🏗️ Architecture

Agentum is built on solid foundations:

- **LangGraph**: Graph-based workflow orchestration
- **LangChain**: LLM integration and tool ecosystem
- **Pydantic**: Type-safe state management
- **Rich**: Beautiful console output
- **ChromaDB**: Vector storage for RAG
- **Redis**: State persistence and checkpointing

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Inspired by modern agent frameworks and multi-agent systems research
- Community feedback and contributions

---

**Ready to build the future of AI? Get started with Agentum today!** 🚀