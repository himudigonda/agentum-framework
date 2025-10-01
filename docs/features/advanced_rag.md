# Advanced RAG with Rerankers

Standard vector search is good at finding a broad set of potentially relevant documents. However, for high-precision question-answering, you need to ensure the *most* relevant context is given to the LLM. This is where a **reranker** comes in.

A reranker takes a larger list of documents from the initial search and uses a more sophisticated model to re-order them based on true semantic relevance to the query.

## Using the Reranking KnowledgeBase

Agentum provides built-in reranking capabilities through the `KnowledgeBase` class. Simply enable reranking when creating your knowledge base, and Agentum will automatically use cross-encoder models to improve retrieval quality.

### Example

This workflow uses a reranker to answer a nuanced question that requires distinguishing between different types of documents.

```python
import os
from dotenv import load_dotenv
from agentum import Agent, GoogleLLM, KnowledgeBase, State, Workflow, create_vector_search_tool

load_dotenv()

# 1. Create a Knowledge Base with reranking enabled
kb = KnowledgeBase(name="company_docs", enable_reranking=True)

# Add some documents to the knowledge base
kb.add([
    "Our company's main strategic priority is to expand into European markets by 2025.",
    "Q3 earnings showed a 15% increase in revenue compared to last year.",
    "The new product launch strategy focuses on mobile-first customer acquisition.",
    "Employee satisfaction surveys indicate 92% approval rating for remote work policies."
])

# 2. Create the vector search tool with reranking
vector_tool = create_vector_search_tool(kb)

# 3. Create the agent
analyst = Agent(
    name="StrategyAnalyst",
    system_prompt="You are an expert analyst. Answer questions using only the highly relevant information found by your tool.",
    llm=GoogleLLM(model="gemini-2.5-flash-lite", api_key=os.getenv("GOOGLE_API_KEY")),
    tools=[vector_tool],
)

# 4. Define the workflow
class RerankingState(State):
    question: str
    answer: str = ""

workflow = Workflow(name="Reranking_RAG", state=RerankingState)
workflow.add_task(
    name="answer",
    agent=analyst,
    instructions="Use your tool to find the answer: {question}",
    output_mapping={"answer": "output"},
)
workflow.set_entry_point("answer")
workflow.add_edge("answer", workflow.END)

# 5. Run it
# This question is designed to see if the reranker can find the "strategy" document
# instead of just the "earnings" document.
result = workflow.run({
    "question": "What is the company's main strategic priority?"
})
print(result["answer"])
```

### How it Works

1. **Broad Fetch:** The base retriever fetches a larger set of documents (default: 10).
2. **Intelligent Reranking:** The cross-encoder model scores each document for relevance to the specific query.
3. **Precise Context:** Only the top-ranked documents are returned and formatted into the final context.

This process dramatically reduces noise and improves the accuracy of your RAG agents.

## Configuration Options

### Customizing Reranking Behavior

You can customize the reranking behavior by modifying the retriever parameters:

```python
# Create a knowledge base with reranking
kb = KnowledgeBase(name="custom_kb", enable_reranking=True)

# Get the retriever with custom parameters
retriever = kb.as_retriever(
    search_kwargs={"k": 15}  # Fetch 15 documents, then rerank to top 5
)

# Create the tool with the custom retriever
vector_tool = create_vector_search_tool(kb, retriever=retriever)
```

### Disabling Reranking

If you want to use standard vector search without reranking:

```python
# Create a knowledge base without reranking
kb = KnowledgeBase(name="standard_kb", enable_reranking=False)
```

## Performance Considerations

- **Reranking adds computational overhead** but significantly improves result quality
- **Use reranking for complex queries** where precision is more important than speed
- **Consider disabling reranking** for simple keyword-based searches
- **The cross-encoder model** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is automatically downloaded on first use

## Real-World Use Cases

Reranking is particularly effective for:

- **Complex analytical questions** requiring synthesis of multiple concepts
- **Factual queries** where accuracy is critical
- **Domain-specific knowledge** where context matters
- **Multi-step reasoning** that depends on finding the most relevant information

See `examples/13_advanced_rag_reranking.py` for a complete working example.
