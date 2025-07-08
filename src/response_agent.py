from typing import Dict, Any, List
import logging
from utils import Message, timer


class ResponseAgent:
    """Enhanced Response Agent that aggregates and formats final responses."""
    
    def __init__(self):
        """Initialize response agent with formatting capabilities."""
        self.response_templates = {
            "trend": self._format_trend_response,
            "rag": self._format_rag_response,
            "mixed": self._format_mixed_response
        }
        
    @timer
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process and aggregate responses from other agents."""
        messages = state.get("messages", [])
        
        # Collect responses from different agents
        trend_response = None
        rag_response = None
        loader_response = None
        
        for msg in messages:
            if isinstance(msg, dict):
                sender = msg.get("sender", "")
                content = msg.get("content", {})
                
                if sender == "TrendAgent":
                    trend_response = content
                elif sender == "RAGAgent":
                    rag_response = content
                elif sender == "DocumentLoaderAgent":
                    loader_response = content
        
        # Determine response type and format
        if loader_response:
            response_type = "loader"
            final_response = self._format_loader_response(loader_response)
        elif trend_response and not rag_response:
            response_type = "trend"
            final_response = self._format_trend_response(trend_response)
        elif rag_response and not trend_response:
            response_type = "rag"
            final_response = self._format_rag_response(rag_response)
        else:
            response_type = "mixed"
            final_response = self._format_mixed_response(trend_response, rag_response)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(trend_response, rag_response)
        
        # Determine if human review is needed
        needs_review = confidence < 0.7 or self._needs_quality_review(final_response)
        
        # Add metadata
        final_response += self._add_metadata(state, confidence)
        
        # Update state
        state["final_response"] = final_response
        state["confidence_score"] = confidence
        state["needs_review"] = needs_review
        
        # Track metrics
        state.setdefault("metrics", {}).update({
            "response_type": response_type,
            "response_length": len(final_response),
            "confidence_score": confidence
        })
        
        # Add final message
        msg = Message(
            "ResponseAgent", 
            "User",
            {
                "response": final_response,
                "confidence": confidence,
                "needs_review": needs_review
            }
        )
        state["messages"].append(msg.to_dict())
        
        logging.info(f"Generated {response_type} response with confidence {confidence:.2f}")
        
        return state
    
    def _format_trend_response(self, trend_data: Dict[str, Any]) -> str:
        """Format trend analysis response."""
        if not trend_data:
            return "No trend analysis available."
        
        parts = []
        
        # Add summary
        summary = trend_data.get("summary", "")
        if summary:
            parts.append(f"📊 **{summary}**\n")
        
        # Add details
        details = trend_data.get("details", {})
        if details:
            for key, value in details.items():
                parts.append(f"• **{key}**: {value}")
        
        # Add confidence indicator
        confidence = trend_data.get("confidence", 0.5)
        if confidence >= 0.9:
            parts.append("\n✅ High confidence analysis")
        elif confidence >= 0.7:
            parts.append("\n⚠️ Moderate confidence analysis")
        else:
            parts.append("\n❓ Low confidence - additional data may be needed")
        
        return "\n".join(parts)
    
    def _format_loader_response(self, loader_data: Dict[str, Any]) -> str:
        """Format document loader response."""
        if not loader_data:
            return "No loader data available."
        
        parts = ["📚 **Data Management Response:**\n"]
        
        status = loader_data.get("status", "unknown")
        
        if status == "success":
            parts.append("✅ Operation completed successfully!")
            
            # Add specific details based on operation
            if loader_data.get("documents_loaded"):
                parts.append(f"\n📄 Documents loaded: {loader_data['documents_loaded']}")
            if loader_data.get("chunks_created"):
                parts.append(f"🧩 Chunks created: {loader_data['chunks_created']}")
            if loader_data.get("document_count"):
                parts.append(f"📊 Total documents in store: {loader_data['document_count']}")
                
        elif status == "ready":
            parts.append("✅ Vectorstore is operational")
            parts.append(f"\n📊 Current status:")
            parts.append(f"• Documents: {loader_data.get('document_count', 'N/A')}")
            parts.append(f"• Path: {loader_data.get('vectorstore_path', 'N/A')}")
            
            if loader_data.get("sample_metadata_keys"):
                parts.append(f"• Metadata fields: {', '.join(loader_data['sample_metadata_keys'][:5])}")
                
        elif status == "no_update":
            parts.append("ℹ️ No updates needed - data is current")
            
        else:
            parts.append(f"⚠️ Status: {status}")
            if loader_data.get("message"):
                parts.append(f"Message: {loader_data['message']}")
        
        # Add analysis results if present
        if loader_data.get("analysis"):
            analysis = loader_data["analysis"]
            parts.append("\n📊 **Data Analysis:**")
            parts.append(f"• Average document length: {analysis.get('avg_doc_length', 'N/A')} chars")
            parts.append(f"• Recommended chunk size: {analysis.get('recommended_chunk_size', 'N/A')}")
        
        return "\n".join(parts)
    
    def _format_rag_response(self, rag_data: Dict[str, Any]) -> str:
        """Format RAG search response."""
        if not rag_data:
            return "No search results available."
        
        parts = []
        
        # Add main answer
        answer = rag_data.get("answer", "")
        if answer:
            parts.append(f"🔍 **Search Results:**\n\n{answer}")
        
        # Add tools used
        tools = rag_data.get("tools_used", [])
        if tools:
            parts.append(f"\n🛠️ **Tools Used:** {', '.join(tools)}")
        
        return "\n".join(parts)
    
    def _format_mixed_response(self, trend_data: Dict, rag_data: Dict) -> str:
        """Format combined response from multiple agents."""
        parts = ["📊 **Comprehensive Analysis:**\n"]
        
        if trend_data:
            parts.append("**Statistical Analysis:**")
            parts.append(self._format_trend_response(trend_data))
            parts.append("")
        
        if rag_data:
            parts.append("**Detailed Search Results:**")
            parts.append(self._format_rag_response(rag_data))
        
        return "\n".join(parts)
    
    def _calculate_confidence(self, trend_data: Dict, rag_data: Dict, loader_data: Dict = None) -> float:
        """Calculate overall confidence score."""
        confidences = []
        
        if trend_data and "confidence" in trend_data:
            confidences.append(trend_data["confidence"])
        
        if rag_data and "confidence" in rag_data:
            confidences.append(rag_data["confidence"])
            
        if loader_data and loader_data.get("status") == "success":
            confidences.append(0.95)  # High confidence for successful data operations
        
        if not confidences:
            return 0.5
        
        # Weighted average
        return sum(confidences) / len(confidences)
    
    def _needs_quality_review(self, response: str) -> bool:
        """Check if response needs quality review."""
        # Check for quality indicators
        if len(response) < 50:
            return True
        
        error_indicators = [
            "error", "failed", "no data", "not available",
            "❌", "Error processing"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in error_indicators)
    
    def _add_metadata(self, state: Dict, confidence: float) -> str:
        """Add metadata footer to response."""
        metadata_parts = []
        
        # Add confidence
        metadata_parts.append(f"Confidence: {confidence:.0%}")
        
        # Add tool results if available
        tool_results = state.get("tool_results", {})
        if tool_results.get("tools_used"):
            tools_count = len(tool_results["tools_used"])
            metadata_parts.append(f"Tools used: {tools_count}")
        
        # Add routing info
        routing = state.get("routing_decision", "")
        if routing:
            metadata_parts.append(f"Processed by: {routing}")
        
        if metadata_parts:
            return f"\n\n---\n📊 " + " | ".join(metadata_parts)
        
        return ""
