## multi agent cli deployment for financial consumer complaints chatbot
import sys
import os
import logging
from utils import setup_logging, Message
from langgraph.graph import StateGraph, START, END
from multi_agent.state import State
from multi_agent.coordinator_agent import CoordinatorAgent
from multi_agent.trend_agent import TrendAgent
from multi_agent.rag_agent import RAGAgent
from multi_agent.response_agent import ResponseAgent
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Financial-Consumer-Chatbot"
os.environ["LANGSMITH_TIMEOUT_MS"] = "5000"
os.environ["LANGSMITH_MAX_RETRIES"] = "2"


def route_coordinator(state: dict) -> str:
    """Route based on coordinator's decision."""
    routing = state.get("routing_decision", "RAGAgent")
    return routing

def build_graph() -> StateGraph:
    """Build the multi-agent graph with proper routing."""
    graph = StateGraph(State)
    
    # Initialize agents
    coordinator = CoordinatorAgent()
    trend = TrendAgent()
    rag = RAGAgent()
    response = ResponseAgent()
    
    # Add nodes
    graph.add_node("CoordinatorAgent", coordinator.process)
    graph.add_node("TrendAgent", trend.process)
    graph.add_node("RAGAgent", rag.process)
    graph.add_node("ResponseAgent", response.process)
    
    # Add edges
    graph.add_edge(START, "CoordinatorAgent")
    
    # Conditional routing from Coordinator
    graph.add_conditional_edges(
        "CoordinatorAgent",
        route_coordinator,
        {
            "TrendAgent": "TrendAgent",
            "RAGAgent": "RAGAgent"
        }
    )
    
    # Both agents route to ResponseAgent
    graph.add_edge("TrendAgent", "ResponseAgent")
    graph.add_edge("RAGAgent", "ResponseAgent")
    
    # ResponseAgent to END
    graph.add_edge("ResponseAgent", END)
    
    # Compile the graph
    return graph.compile()

def main():
    """Main application loop."""
    # Setup
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    setup_logging("logs/multi_agent.log")
    
    # Build graph
    graph = build_graph()
    
    print("💬 Financial Complaints Chatbot Ready. Type 'exit' to quit.")
    print("   Try queries like:")
    print("   - 'What are the top complaint categories?'")
    print("   - 'Tell me about credit card issues'")
    print("   - 'Search for mortgage information'\n")
    
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            sys.exit(0)
        
        # Create initial message
        msg = Message("User", "CoordinatorAgent", {"query": q})
        
        # Create initial state - store Message as dict
        initial_state = {
            "messages": [msg.to_dict()],  # Convert to dict
            "query": q,
            "final_response": "",
            "routing_decision": ""
        }
        
        try:
            # Run the graph
            final_state = graph.invoke(initial_state)
            
            # Display response
            response = final_state.get("final_response", "No response generated")
            print(f"\n🤖 Assistant:\n{response}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            logging.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()
