import os
from dotenv import load_dotenv
from src.rag.chain import create_rag_chain
from src.graph.state import create_research_graph
from langchain_core.messages import HumanMessage

def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize RAG chain
        print("Initializing RAG chain...")
        rag_chain = create_rag_chain("data/raw/apple_10k.pdf")
        
        # Create research graph
        print("Creating research graph...")
        chain = create_research_graph(rag_chain)
        
        # Create LCEL chain
        research_chain = enter_chain | chain
        
        # Example queries
        test_queries = [
            "What are Apple's main revenue streams?",
            "What are the key risk factors in the latest 10-K?",
            "How has the company's R&D spending changed?",
        ]
        
        # Process each query with streaming
        for query in test_queries:
            print(f"\nProcessing query: {query}")
            print("Stream of responses:")
            for s in research_chain.stream(query, {"recursion_limit": 10}):
                if "__end__" not in s:
                    print("\nStep output:")
                    print(s)
                    print("---")
            print("\nQuery completed\n")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()