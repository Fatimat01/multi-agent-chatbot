from typing import Dict, Any, List
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import Message, timer


class CoordinatorAgent:
    """Enhanced Coordinator Agent with intelligent routing capabilities."""
    
    def __init__(self):
        """Initialize the coordinator with LLM-based routing."""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.routing_prompt = ChatPromptTemplate.from_template("""
        Analyze the following user query and determine the best agent to handle it.
        
        Available agents:
        - TrendAgent: Handles statistical analysis, trends, top categories, frequency analysis
        - RAGAgent: Handles specific searches, company inquiries, detailed complaint information
        - DocumentLoaderAgent: Handles data loading, vectorstore management, ingestion status
        
        Query: {query}
        
        Respond with ONLY the agent name (TrendAgent, RAGAgent, or DocumentLoaderAgent).
        """)
        
        # Keywords for fallback routing
        self.trend_keywords = [
            "trend", "top", "most frequent", "statistics", "analysis",
            "pattern", "distribution", "summary", "overview", "ranking"
        ]
        
        self.loader_keywords = [
            "load", "ingest", "update", "vectorstore", "chunk", "index",
            "data status", "refresh", "import"
        ]
        
    @timer
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming query and route to appropriate agent."""
        messages = state.get("messages", [])
        if not messages:
            logging.error("No messages in state")
            return state
        
        # Extract query
        latest_msg = messages[-1]
        if isinstance(latest_msg, dict):
            query = latest_msg.get("content", {}).get("query", "")
        else:
            query = latest_msg.content.get("query", "")
        
        # Intelligent routing
        routing_decision = self._determine_routing(query)
        
        logging.info(f"Routing query '{query[:50]}...' to {routing_decision}")
        
        # Update state
        state["query"] = query
        state["routing_decision"] = routing_decision
        
        # Add routing message
        routing_msg = Message(
            "CoordinatorAgent", 
            routing_decision,
            {
                "query": query,
                "routing_reason": f"Query matches {routing_decision} capabilities"
            }
        )
        state["messages"].append(routing_msg.to_dict())
        
        return state
    
    def _determine_routing(self, query: str) -> str:
        """Determine routing using LLM with fallback to keyword matching."""
        try:
            # Try LLM-based routing first
            response = self.llm.invoke(
                self.routing_prompt.format(query=query)
            )
            decision = response.content.strip()
            
            if decision in ["TrendAgent", "RAGAgent", "DocumentLoaderAgent"]:
                return decision
                
        except Exception as e:
            logging.warning(f"LLM routing failed, using fallback: {e}")
        
        # Fallback to keyword-based routing
        query_lower = query.lower()
        
        # Check for loader keywords first
        if any(keyword in query_lower for keyword in self.loader_keywords):
            return "DocumentLoaderAgent"
        
        # Check for trend keywords
        if any(keyword in query_lower for keyword in self.trend_keywords):
            return "TrendAgent"
        
        return "RAGAgent"
