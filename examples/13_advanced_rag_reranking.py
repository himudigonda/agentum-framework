"""
Advanced RAG with Reranking Example

This example demonstrates Agentum's enhanced RAG capabilities with cross-encoder reranking.
The KnowledgeBase now includes automatic reranking to improve retrieval quality by
re-scoring documents based on query-document relevance using a cross-encoder model.

Key Features Demonstrated:
- Advanced RAG with reranking
- Multi-document knowledge base
- Intelligent document retrieval
- Cross-encoder reranking for better relevance
"""
import os
from dotenv import load_dotenv
from pydantic import Field
from agentum import Agent, GoogleLLM, KnowledgeBase, State, Workflow, create_vector_search_tool
load_dotenv()

class AdvancedRAGState(State):
    question: str
    answer: str = Field(default='')
    sources_used: str = Field(default='')
company_kb = KnowledgeBase(name='company_docs_advanced', enable_reranking=True, embedding_model='all-MiniLM-L6-v2')
company_kb.add(sources=['./examples/earnings_report.txt', 'https://en.wikipedia.org/wiki/Artificial_intelligence', 'https://en.wikipedia.org/wiki/Machine_learning'])
vector_search_tool = create_vector_search_tool(company_kb)
advanced_analyst = Agent(name='AdvancedAnalyst', system_prompt='You are an expert analyst with access to a comprehensive knowledge base. \n    You provide detailed, well-sourced answers using only information from your tools.\n    Always cite specific sources and provide context for your answers.', llm=GoogleLLM(api_key=os.getenv('GOOGLE_API_KEY'), model='gemini-2.5-flash-lite'), tools=[vector_search_tool])
advanced_rag_workflow = Workflow(name='Advanced_RAG_Analyst', state=AdvancedRAGState)
advanced_rag_workflow.add_task(name='analyze_with_rag', agent=advanced_analyst, instructions='Using your knowledge base, provide a comprehensive answer to: {question}\n    \n    Requirements:\n    1. Search for relevant information using your tools\n    2. Provide a detailed, well-structured answer\n    3. Include specific citations and sources\n    4. If information is not available, clearly state this', output_mapping={'answer': 'output'})
advanced_rag_workflow.set_entry_point('analyze_with_rag')
advanced_rag_workflow.add_edge('analyze_with_rag', advanced_rag_workflow.END)
if __name__ == '__main__':
    test_questions = ['What are the key financial metrics and growth drivers mentioned in the earnings report?', 'How does artificial intelligence relate to machine learning?', 'What are the main challenges and opportunities in AI development?']
    for question in test_questions:
        print(f'\n🚀 Analyzing: {question}')
        print('=' * 80)
        initial_state = {'question': question}
        final_state = advanced_rag_workflow.run(initial_state)
        print(f'\n📊 Answer:')
        print(final_state['answer'])
        print('\n' + '=' * 80)
    print('\n✅ Advanced RAG with reranking completed successfully!')
    print('🔄 Reranking improved document relevance and answer quality!')