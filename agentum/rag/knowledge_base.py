# agentum/rag/knowledge_base.py
from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console

console = Console()


class KnowledgeBase:
    """A class that handles the ingestion and storage of documents for RAG."""
    
    def __init__(self, name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.name = name
        self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = Chroma(
            collection_name=name, embedding_function=self.embedding_function
        )
        console.print(f"📚 KnowledgeBase '{self.name}' initialized.", style="bold blue")

    def add(self, sources: List[str]):
        """Loads, splits, and embeds documents from a list of file paths or URLs."""
        docs = []
        for source in sources:
            loader = None
            if source.startswith("http"):
                loader = WebBaseLoader(source)
            elif source.endswith(".pdf"):
                loader = PyPDFLoader(source)
            elif source.endswith(".txt"):
                loader = TextLoader(source)
            else:
                console.print(
                    f"[yellow]Warning: Skipping unsupported source type: {source}[/yellow]"
                )
                continue

            console.print(f"  - Loading from source: {source}")
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        self.vector_store.add_documents(documents=splits)
        console.print(
            f"  ✅ Added {len(splits)} document chunks to the '{self.name}' knowledge base."
        )

    def as_retriever(self, **kwargs: Any):
        """Exposes the vector store as a LangChain retriever object."""
        return self.vector_store.as_retriever(**kwargs)
