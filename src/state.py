from typing_extensions import TypedDict, Annotated
from typing import List, Any, Dict

class State(TypedDict):
    messages: List[Dict[str, Any]]  # Changed from using add_messages
    query: str
    final_response: str
    routing_decision: str
