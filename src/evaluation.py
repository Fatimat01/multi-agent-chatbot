import time
import logging
import os
from utils import setup_logging, Message
from langgraph.graph import StateGraph, START, END
from multi_agent.coordinator_agent import CoordinatorAgent
from multi_agent.trend_agent import TrendAgent
from multi_agent.rag_agent import RAGAgent
from multi_agent.response_agent import ResponseAgent
from multi_agent.state import State

class Evaluator:
    """Benchmarks average end‐to‐end latency across queries."""
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        setup_logging("logs/evaluation.log")
        self.graph = self._build_graph()
    
    def _route_coordinator(self, state: dict) -> str:
        """Route based on coordinator's decision."""
        return state.get("routing_decision", "RAGAgent")
    
    def _build_graph(self) -> StateGraph:
        """Build the evaluation graph."""
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
        
        # Conditional routing
        graph.add_conditional_edges(
            "CoordinatorAgent",
            self._route_coordinator,
            {
                "TrendAgent": "TrendAgent",
                "RAGAgent": "RAGAgent"
            }
        )
        
        graph.add_edge("TrendAgent", "ResponseAgent")
        graph.add_edge("RAGAgent", "ResponseAgent")
        graph.add_edge("ResponseAgent", END)
        
        return graph.compile()
    
    def benchmark(self, queries: list[str]):
        """Run benchmark on provided queries."""
        latencies = []
        successful = 0
        
        for i, q in enumerate(queries, 1):
            msg = Message("User", "CoordinatorAgent", {"query": q})
            initial_state = {
                "messages": [msg],
                "query": q,
                "final_response": "",
                "routing_decision": ""
            }
            
            try:
                start = time.time()
                final_state = self.graph.invoke(initial_state)
                elapsed = time.time() - start
                latencies.append(elapsed)
                successful += 1
                logging.info(f"Query {i}/{len(queries)}: {elapsed:.2f}s - '{q[:50]}...'")
            except Exception as e:
                logging.error(f"Query {i} failed: {e}")
        
        if latencies:
            avg = sum(latencies) / len(latencies)
            logging.info(f"Benchmark complete: {successful}/{len(queries)} successful")
            logging.info(f"Average latency: {avg:.2f}s")
            logging.info(f"Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s")
            print(f"\n📊 Benchmark Results:")
            print(f"  • Successful: {successful}/{len(queries)}")
            print(f"  • Average latency: {avg:.2f}s")
            print(f"  • Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s")
        else:
            print("❌ No successful queries")

# Example usage
if __name__ == "__main__":
    evaluator = Evaluator()
    test_queries = [
        "What are the top complaint categories?",
        "Show me trends in customer complaints",
        "Search for credit card issues",
        "Tell me about mortgage problems",
        "What are the most frequent issues?"
    ]
    evaluator.benchmark(test_queries)