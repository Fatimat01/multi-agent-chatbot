from utils import Message, timer
import logging

class CoordinatorAgent:
    @timer
    def process(self, state: dict) -> dict:
        """Process incoming query and route to appropriate agent."""
        # Get the latest message
        messages = state.get("messages", [])
        if not messages:
            logging.error("No messages in state")
            return state
        
        latest_msg = messages[-1]
        # Handle both Message objects and dicts
        if isinstance(latest_msg, dict):
            query = latest_msg.get("content", {}).get("query", "")
        else:
            query = latest_msg.content.get("query", "")
        
        # Determine routing
        if any(k in query.lower() for k in ["trend", "top", "most frequent", "statistics", "analysis"]):
            routing_decision = "TrendAgent"
            logging.info(f"Routing to TrendAgent for query: {query}")
        else:
            routing_decision = "RAGAgent"
            logging.info(f"Routing to RAGAgent for query: {query}")
        
        # Update state
        state["query"] = query
        state["routing_decision"] = routing_decision
        
        return state