# Agentum Framework

[![PyPI version](https://badge.fury.io/py/agentum.svg)](https://badge.fury.io/py/agentum)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Python-native framework for building, orchestrating, and observing production-ready, multi-modal AI agent systems.**

Agentum provides a clean, developer-centric API to build sophisticated AI workflows that can see, hear, speak, and reason. It abstracts away the complexity of the agentic loop, multi-modal data handling, and state management, letting you focus on your application's logic.

## 🚀 Key Features

- **🤖 Multi-Modal Agents:** Build agents that can see, hear, and speak. Natively integrate vision, speech-to-text, and text-to-speech.
- **🔄 Multi-Provider LLMs:** Swap between Google Gemini, Anthropic Claude, and more with a single line of code, thanks to a provider-agnostic architecture.
- **🗣️ End-to-End Voice Workflows:** Create complete voice assistants that can listen to a user's request, think, and generate a spoken response.
- **📚 Advanced RAG:** A powerful, easy-to-use `KnowledgeBase` for creating agents that can reason over your own documents and data.
- **🔍 Deep Observability:** An event-driven system (`@workflow.on(...)`) to hook into every step of an agent's reasoning process for unparalleled debugging.
- **🧪 Professional Testing Harness:** A first-class `TestSuite` and `Evaluator` system to systematically test and score your agents' performance.
- **🛠️ Powerful CLI:** Validate, visualize, scaffold, and run your agentic workflows directly from the terminal.

## 📦 Installation

```bash
pip install agentum
```
You will also need libraries for the specific providers you want to use:
```bash
# For Google (Gemini, Vision, Speech)
pip install langchain-google-genai google-cloud-texttospeech langchain-google-community

# For Anthropic (Claude)
pip install langchain-anthropic

# For high-quality web search
pip install tavily-python
```

## 🎯 Quick Start: A Voice Assistant in ~40 Lines of Code

Build a complete voice assistant that listens to an audio file, thinks, and speaks a response.

```python
# filename: voice_assistant.py
import os
from dotenv import load_dotenv
from agentum import Agent, GoogleLLM, State, Workflow, text_to_speech, transcribe_audio

load_dotenv()

# 1. Define the State for our voice interaction
class VoiceAssistantState(State):
    input_audio_path: str
    output_audio_path: str
    question_text: str = ""
    answer_text: str = ""

# 2. Define the Agent that will answer the question
assistant_agent = Agent(
    name="HelpfulAssistant",
    system_prompt="You are a friendly assistant. Provide concise answers.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
)

# 3. Define the end-to-end voice workflow
workflow = Workflow(name="Voice_Assistant", state=VoiceAssistantState)

# Task 1: Listen - Transcribe the input audio to text
workflow.add_task(
    name="listen",
    tool=transcribe_audio,
    inputs={
        "audio_filepath": "{input_audio_path}",
        "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
    },
    output_mapping={"question_text": "output"},
)

# Task 2: Think - Generate a text answer
workflow.add_task(
    name="think",
    agent=assistant_agent,
    instructions="Answer this question: {question_text}",
    output_mapping={"answer_text": "output"},
)

# Task 3: Speak - Convert the text answer back to an audio file
workflow.add_task(
    name="speak",
    tool=text_to_speech,
    inputs={
        "text_to_speak": "{answer_text}",
        "output_filepath": "{output_audio_path}",
    },
)

# 4. Define the control flow
workflow.set_entry_point("listen")
workflow.add_edge("listen", "think")
workflow.add_edge("think", "speak")
workflow.add_edge("speak", workflow.END)

# 5. Run it!
if __name__ == "__main__":
    # You'll need a 'request.wav' file for this to work.
    result = workflow.run({
        "input_audio_path": "request.wav",
        "output_audio_path": "response.mp3",
    })
    print(f"User asked: '{result['question_text']}'")
    print(f"Agent responded: '{result['answer_text']}' (Audio saved to response.mp3)")
```

## 🛠️ Powerful CLI

Agentum comes with a complete CLI to supercharge your development workflow.

```bash
# Run a workflow with an initial JSON state
agentum run voice_assistant.py --state '{"input_audio_path": "request.wav", "output_audio_path": "response.mp3"}'

# Validate a workflow's structure without running it
agentum validate voice_assistant.py

# Generate a visual graph of your workflow
agentum graph voice_assistant.py --output workflow.png

# Scaffold a new workflow file from a template
agentum init my_new_workflow
```

## 📖 Documentation

- **[Getting Started Guide](docs/getting_started.md)** - Build your first multi-agent workflow with real tools.
- **[LLM Providers](docs/features/providers.md)** - Switch between Google, Anthropic, and OpenAI models.
- **[Multi-Modal Capabilities](docs/features/multi_modality.md)** - Add vision support for images and local files.
- **[Advanced RAG](docs/features/advanced_rag.md)** - Use rerankers for high-precision document retrieval.
- **[Speech Capabilities](docs/features/speech.md)** - Build voice assistants with STT and TTS.
- **[Core Concepts](docs/concepts/)** - Deep dive into Agents, State, and Workflows.
- **[Examples](examples/)** - Explore 14 real-world examples showcasing all features.

## 🤝 Contributing

We welcome contributions of all kinds! Please see our **[Contributing Guide](CONTRIBUTING.md)** to get started.

## 📄 License

Agentum is licensed under the **[MIT License](LICENSE)**.
