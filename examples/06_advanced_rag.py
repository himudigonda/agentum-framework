# examples/06_advanced_rag.py
import os

from dotenv import load_dotenv

from agentum import (
    Agent,
    GoogleLLM,
    KnowledgeBase,
    State,
    Workflow,
    create_vector_search_tool,
)

load_dotenv()

# 1. SETUP: Create and populate a Knowledge Base with our report.
print("--- Creating Knowledge Base ---")
company_kb = KnowledgeBase(name="company_reports")
company_kb.add(sources=["examples/earnings_report.txt"])
print("---------------------------\n")


# 2. DEFINE STATE
class AnalystState(State):
    question: str
    answer: str = ""


# 3. DEFINE A RAG-POWERED AGENT
# Create a vector search tool bound to our knowledge base
vector_search_tool = create_vector_search_tool(company_kb)

analyst = Agent(
    name="FinancialAnalyst",
    system_prompt="You are an expert financial analyst. You must answer questions using only the information found by your tools.",
    llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash-lite"),
    tools=[vector_search_tool],
)

# 4. DEFINE THE RAG WORKFLOW
rag_workflow = Workflow(name="RAG_Analyst", state=AnalystState)

rag_workflow.add_task(
    name="answer_question",
    agent=analyst,
    instructions="""
    Use your tools to find information to answer the user's question.
    
    User Question: {question}
    """,
    output_mapping={"answer": "output"},
)

rag_workflow.set_entry_point("answer_question")
rag_workflow.add_edge("answer_question", rag_workflow.END)

# 5. RUN IT
if __name__ == "__main__":
    initial_state = {
        "question": "What was the main driver of growth for InnovateCorp?",
    }

    final_state = rag_workflow.run(initial_state)

    print("\n--- Final Answer ---")
    print(final_state["answer"])
    assert "quantumleap" in final_state["answer"].lower()
    print("\n✅ RAG workflow completed successfully!")
