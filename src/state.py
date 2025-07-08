from typing import TypedDict, List, Dict, Any, Optional


class State(TypedDict):
    """Enhanced state definition for multi-agent system.
    
    This state is passed between all agents and contains:
    - Messages: Communication history between agents
    - Query: The original user query
    - Routing decisions and processing flags
    - Results from different agents
    - Human feedback when applicable
    - Document loader status and results
    """
    messages: List[Dict[str, Any]]
    query: str
    final_response: str
    routing_decision: str
    needs_review: bool
    human_feedback: str
    confidence_score: Optional[float]
    tool_results: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, float]]  # For evaluation tracking
    loader_command: Optional[str]  # Command for document loader
    loader_result: Optional[Dict[str, Any]]  # Results from document loader
