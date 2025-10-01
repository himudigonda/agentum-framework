# Agentum Framework

[![PyPI version](https://badge.fury.io/py/agentum.svg)](https://badge.fury.io/py/agentum)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python-native framework for building, orchestrating, and observing production-ready AI agent systems.**

Agentum provides a clean, developer-centric API to build sophisticated AI workflows that can reason, use tools, access knowledge, and learn from interactions. It abstracts away the complexity of the agentic loop, state management, and observability, letting you focus on your application's logic.

## 🚀 Key Features

- **🤖 True Agentic Behavior:** Build agents that can autonomously use tools to solve problems through a `think -> act -> observe` loop.
- **📚 Built-in RAG:** A powerful, easy-to-use `KnowledgeBase` for creating agents that can reason over your own documents and data.
- **🔍 Deep Observability:** An event-driven system (`@workflow.on(...)`) to hook into every step of an agent's reasoning process for unparalleled debugging.
- **🧠 Stateful Memory:** Equip agents with `ConversationMemory` to enable rich, multi-turn conversational applications.
- **🧪 Professional Testing Harness:** A first-class `TestSuite` and `Evaluator` system to systematically test and score your agents' performance.
- **🛡️ Production Ready:** Automatic retries, asynchronous core, state persistence via Redis, and a complete CLI for managing your workflows.

## 📦 Installation

```bash
pip install agentum
```
You will also need an LLM provider library, for example:
```bash
pip install langchain-google-genai tavily-python
```

## 🎯 Quick Start: A RAG-Powered Analyst

In less than 30 lines of code, build an agent that can read a document and answer questions about it.

```python
# filename: analyst_workflow.py
import os
from dotenv import load_dotenv
from agentum import Agent, State, Workflow, GoogleLLM, KnowledgeBase, create_vector_search_tool

load_dotenv()

# 1. Ingest your data into a KnowledgeBase
kb = KnowledgeBase(name="company_docs")
kb.add(sources=["./path/to/your/report.txt"])

# 2. Create a RAG-enabled tool and agent
rag_tool = create_vector_search_tool(kb)
analyst = Agent(
    name="DocAnalyst",
    system_prompt="You are an analyst. Answer questions using only the provided documents.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[rag_tool]
)

# 3. Define the workflow
class AnalystState(State):
    question: str
    answer: str = ""

workflow = Workflow(name="Document_QA", state=AnalystState)
workflow.add_task(
    name="answer_question",
    agent=analyst,
    instructions="Use your tools to find the answer to: {question}",
    output_mapping={"answer": "output"}
)
workflow.set_entry_point("answer_question")
workflow.add_edge("answer_question", workflow.END)

# 4. Run it!
if __name__ == "__main__":
    result = workflow.run({"question": "What was the main driver of growth?"})
    print(result["answer"])
```

## 🛠️ Powerful CLI

Agentum comes with a complete CLI to supercharge your development workflow.

```bash
# Run a workflow with an initial state
agentum run analyst_workflow.py --state '{"question": "What are our new initiatives?"}'

# Validate a workflow's structure without running it
agentum validate analyst_workflow.py

# Generate a visual graph of your workflow
agentum graph analyst_workflow.py --output workflow.png

# Scaffold a new workflow file
agentum init my_new_workflow
```

## 📖 Documentation

- **[Getting Started Guide](docs/getting_started.md)** - Your first steps with Agentum.
- **[Core Concepts](docs/concepts/)** - Deep dive into Agents, State, and Workflows.
- **[Advanced Features](docs/features/)** - Learn about RAG, Observability, Memory, and Testing.
- **[Examples](examples/)** - Explore a rich set of real-world examples.

## 🤝 Contributing

We welcome contributions of all kinds! Please see our **[Contributing Guide](CONTRIBUTING.md)** to get started.

## 📄 License

Agentum is licensed under the **[MIT License](LICENSE)**.
