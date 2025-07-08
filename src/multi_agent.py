import os
import sys
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

# Configure environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Financial-Consumer-Chatbot"
os.environ["LANGSMITH_TIMEOUT_MS"] = "5000"
os.environ["LANGSMITH_MAX_RETRIES"] = "2"

# Import our components
from multi_agent.state import State
from multi_agent.coordinator_agent import CoordinatorAgent
from multi_agent.trend_agent import TrendAgent
from multi_agent.rag_agent import RAGAgent
from multi_agent.response_agent import ResponseAgent
from multi_agent.human_feedback_agent import HumanFeedbackAgent
from multi_agent.document_loader_agent import DocumentLoaderAgent
from utils import setup_logging, Message


class MultiAgentSystem:
    """Main multi-agent system for financial complaints analysis."""
    
    def __init__(self, enable_human_in_loop: bool = True):
        """Initialize the multi-agent system.
        
        Args:
            enable_human_in_loop: Whether to enable human feedback agent
        """
        self.enable_human_in_loop = enable_human_in_loop
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        setup_logging("logs/multi_agent.log")
        
    def _route_coordinator(self, state: Dict[str, Any]) -> str:
        """Route based on coordinator's decision."""
        return state.get("routing_decision", "RAGAgent")
    
    def _route_response(self, state: Dict[str, Any]) -> str:
        """Route from response agent based on human-in-loop setting."""
        if self.enable_human_in_loop and state.get("needs_review", False):
            return "HumanFeedbackAgent"
        return END
    
    def _build_graph(self) -> StateGraph:
        """Build the multi-agent graph with proper routing."""
        graph = StateGraph(State)
        
        # Initialize agents
        coordinator = CoordinatorAgent()
        trend = TrendAgent()
        rag = RAGAgent()
        response = ResponseAgent()
        human_feedback = HumanFeedbackAgent()
        document_loader = DocumentLoaderAgent()
        
        # Add nodes
        graph.add_node("CoordinatorAgent", coordinator.process)
        graph.add_node("TrendAgent", trend.process)
        graph.add_node("RAGAgent", rag.process)
        graph.add_node("ResponseAgent", response.process)
        graph.add_node("DocumentLoaderAgent", document_loader.process)
        
        if self.enable_human_in_loop:
            graph.add_node("HumanFeedbackAgent", human_feedback.process)
        
        # Add edges
        graph.add_edge(START, "CoordinatorAgent")
        
        # Conditional routing from Coordinator
        graph.add_conditional_edges(
            "CoordinatorAgent",
            self._route_coordinator,
            {
                "TrendAgent": "TrendAgent",
                "RAGAgent": "RAGAgent",
                "DocumentLoaderAgent": "DocumentLoaderAgent"
            }
        )
        
        # All agents route to ResponseAgent
        graph.add_edge("TrendAgent", "ResponseAgent")
        graph.add_edge("RAGAgent", "ResponseAgent")
        graph.add_edge("DocumentLoaderAgent", "ResponseAgent")
        
        # Conditional routing from ResponseAgent
        if self.enable_human_in_loop:
            graph.add_conditional_edges(
                "ResponseAgent",
                self._route_response,
                {
                    "HumanFeedbackAgent": "HumanFeedbackAgent",
                    END: END
                }
            )
            graph.add_edge("HumanFeedbackAgent", END)
        else:
            graph.add_edge("ResponseAgent", END)
        
        # Compile with checkpointer for memory
        return graph.compile(checkpointer=self.memory)
    
    def process_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Process a single query through the multi-agent system.
        
        Args:
            query: User query to process
            session_id: Session ID for conversation memory
            
        Returns:
            Final state with response
        """
        # Create initial message
        msg = Message("User", "CoordinatorAgent", {"query": query})
        
        # Create initial state
        initial_state = {
            "messages": [msg.to_dict()],
            "query": query,
            "final_response": "",
            "routing_decision": "",
            "needs_review": False,
            "human_feedback": "",
            "loader_command": self._determine_loader_command(query),
            "loader_result": None
        }
        
        # Run the graph with memory
        config = {"configurable": {"thread_id": session_id}}
        final_state = self.graph.invoke(initial_state, config)
        
        return final_state
    
    def _determine_loader_command(self, query: str) -> str:
        """Determine the appropriate loader command based on query."""
        query_lower = query.lower()
        
        if "load" in query_lower or "ingest" in query_lower:
            return "load_data"
        elif "update" in query_lower or "refresh" in query_lower:
            return "update_vectorstore"
        elif "optimize" in query_lower or "chunk" in query_lower:
            return "optimize_chunks"
        else:
            return "check_status"


def main():
    """Main CLI application loop."""
    # Setup
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Ask user about human-in-loop preference
    print("🤖 Financial Complaints Multi-Agent System")
    print("-" * 50)
    enable_human = input("Enable human-in-the-loop review? (y/n): ").lower() == 'y'
    
    # Initialize system
    system = MultiAgentSystem(enable_human_in_loop=enable_human)
    
    print("\n💬 System Ready! Type 'exit' to quit.")
    print("   Try queries like:")
    print("   - 'What are the top complaint categories?'")
    print("   - 'Tell me about credit card issues'")
    print("   - 'Search for mortgage information'")
    print("   - 'Analyze trends in bank complaints'\n")
    
    session_id = f"session_{os.getpid()}"
    
    while True:
        query = input("\nYou: ")
        if query.strip().lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        
        try:
            # Process query
            result = system.process_query(query, session_id)
            
            # Display response
            response = result.get("final_response", "No response generated")
            print(f"\n🤖 Assistant:\n{response}")
            
            # Show if human feedback was incorporated
            if result.get("human_feedback"):
                print(f"\n📝 Human feedback incorporated: {result['human_feedback']}")
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logging.error(f"Error processing query: {e}", exc_info=True)


if __name__ == "__main__":
    main()
