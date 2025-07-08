from utils import Message, timer
import logging

class ResponseAgent:
    """Aggregates TrendAgent & RAGAgent outputs into a final reply."""
    @timer
    def process(self, state: dict) -> dict:
        """Process and aggregate responses from other agents."""
        messages = state.get("messages", [])
        response_parts = []
        
        # Find messages from TrendAgent and RAGAgent
        for msg in messages:
            # Handle dict messages
            if isinstance(msg, dict):
                sender = msg.get("sender", "")
                receiver = msg.get("receiver", "")
                content = msg.get("content", {})
                
                if sender == "TrendAgent" and receiver == "ResponseAgent":
                    response_parts.append("📊 **Trend Analysis Results:**")
                    response_parts.append(content.get("summary", ""))
                    details = content.get("details", {})
                    for k, v in details.items():
                        response_parts.append(f"  • {k}: {v}")
                    
                elif sender == "RAGAgent" and receiver == "ResponseAgent":
                    response_parts.append("\n🗒️ **RAG Answer:**")
                    response_parts.append(content.get("vector_answer", "No vector answer available"))
                    response_parts.append("\n🌐 **Web Fallback:**")
                    response_parts.append(content.get("web_fallback", "No web results available"))
        
        # Create final response
        if response_parts:
            final_response = "\n".join(response_parts)
        else:
            final_response = "No response generated. Please try again."
        
        state["final_response"] = final_response
        
        # Add final message as dict
        msg = Message("ResponseAgent", "User", {"response": final_response})
        state["messages"].append(msg.to_dict())
        
        return state