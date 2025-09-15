#!/usr/bin/env python3
"""
RAG Example with VertexAI Embeddings and Pinecone.
Interactive dialog interface for semantic search.
"""

import os
import sys

os.environ["LANGCHAIN_TRACING_V2"] = "false"
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vertex_ai_embeddings import VertexAIEmbeddings

load_dotenv()

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = "europe-west1"
TEXT_EMBEDDING_MODEL_ID = "text-embedding-005"
LLM_MODEL = "gemini-2.0-flash-lite"
INDEX_NAME = "profiles-df-full"


class RAGSystem:
    """RAG System with VertexAI Embeddings and Pinecone."""

    def __init__(self):
        """Initialize the RAG system."""
        self.project_id = PROJECT_ID
        self.location = LOCATION
        self.index_name = INDEX_NAME

        # Initialize components
        self._setup_embeddings()
        self._setup_pinecone()
        self._setup_llm()
        self._setup_qa_chain()

    def _setup_embeddings(self):
        """Setup VertexAI embeddings."""
        print("🔧 Setting up VertexAI embeddings...")
        self.embedding_model = VertexAIEmbeddings(
            project_id=self.project_id, location=self.location, model=TEXT_EMBEDDING_MODEL_ID
        )
        print("✅ Embeddings ready!")

    def _setup_pinecone(self):
        """Setup Pinecone vector store."""
        print("🔧 Setting up Pinecone vector store...")
        pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc_client.Index(self.index_name)

        # Check index stats
        stats = self.index.describe_index_stats()
        print(f"📊 Index stats: {stats.total_vector_count} vectors")

        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model)
        print("✅ Pinecone ready!")

    def _setup_llm(self):
        """Setup Google Generative AI LLM."""
        print("🔧 Setting up LLM...")
        self.llm = GoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        print("✅ LLM ready!")

    def _setup_qa_chain(self):
        """Setup QA chain."""
        print("🔧 Setting up QA chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}), return_source_documents=True
        )
        print("✅ QA chain ready!")

    def ask_question(self, question: str):
        """Ask a question to the RAG system."""
        try:
            result = self.qa_chain.invoke({"query": question})
            return {"answer": result["result"], "source_documents": result["source_documents"]}
        except Exception as e:
            return {"error": str(e)}

    def similarity_search(self, query: str, k: int = 5):
        """Perform similarity search."""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            return {"error": str(e)}


def interactive_dialog():
    """Interactive dialog interface."""
    print("🚀 Initializing RAG System...")
    print("=" * 50)

    try:
        rag_system = RAGSystem()
        print("\n🎉 RAG System ready!")
        print("=" * 50)

        while True:
            print("\n📝 Options:")
            print("1. Ask a question (RAG)")
            print("2. Similarity search")
            print("3. Exit")

            choice = input("\n🔍 Choose an option (1-3): ").strip()

            if choice == "1":
                question = input("\n❓ Enter your question: ").strip()
                if question:
                    print("\n🤔 Thinking...")
                    result = rag_system.ask_question(question)

                    if "error" in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        print(f"\n💡 Answer: {result['answer']}")
                        print(f"\n📚 Sources used: {len(result['source_documents'])} documents")

                        # Show first source document preview
                        if result["source_documents"]:
                            first_doc = result["source_documents"][0]
                            print("\n📄 First source preview:")
                            print(f"   {first_doc.page_content[:200]}...")

            elif choice == "2":
                query = input("\n🔍 Enter search query: ").strip()
                if query:
                    k = input("📊 Number of results (default 5): ").strip()
                    k = int(k) if k.isdigit() else 5

                    print("\n🔍 Searching...")
                    results = rag_system.similarity_search(query, k=k)

                    if "error" in results:
                        print(f"❌ Error: {results['error']}")
                    else:
                        print(f"\n📋 Found {len(results)} results:")
                        for i, doc in enumerate(results, 1):
                            print(f"\n{i}. {doc.page_content[:150]}...")

            elif choice == "3":
                print("\n👋 Goodbye!")
                break

            else:
                print("❌ Invalid option. Please choose 1, 2, or 3.")

    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("Please check your environment variables and credentials.")


def main():
    """Main function."""
    print("🤖 RAG System with VertexAI Embeddings and Pinecone")
    print("=" * 60)

    # Check environment variables
    required_vars = ["GCP_PROJECT_ID", "PINECONE_API_KEY", "GOOGLE_API_KEY", "CREDENTIALS_PATH_EMBEDDINGS"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file.")
        return

    interactive_dialog()


if __name__ == "__main__":
    main()
