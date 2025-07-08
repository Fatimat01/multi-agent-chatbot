# Multi-agent streamlit app for financial consumer complaints

import os
import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import plotly.graph_objects as go

from utils import setup_logging, Message
from multi_agent_app import MultiAgentSystem
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Logging and Environment Setup
load_dotenv()  # load OPENAI_API_KEY, SERPAPI_API_KEY, etc.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "financial-consumer-chatbot"
os.environ["LANGSMITH_TIMEOUT_MS"] = "5000"
os.environ["LANGSMITH_MAX_RETRIES"] = "2"
setup_logging(log_path="logs/app.log")

# Streamlit Page Setup
st.set_page_config(
    page_title="Financial Complaints Multi-Agent System", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "system" not in st.session_state:
    st.session_state.system = None
    st.session_state.human_in_loop = True
    st.session_state.session_id = f"session_{int(time.time())}"
    st.session_state.metrics = {
        "total_queries": 0,
        "avg_latency": 0,
        "routing_stats": {"TrendAgent": 0, "RAGAgent": 0, "DocumentLoaderAgent": 0}
    }

# Initialize Streamlit Chat History
msgs = StreamlitChatMessageHistory()
if not msgs.messages:
    msgs.add_ai_message("👋 Welcome! I'm a multi-agent system for analyzing financial consumer complaints. Ask me anything!")

# Sidebar Configuration
with st.sidebar:
    st.title("🛠️ System Configuration")
    
    # Human-in-loop toggle
    human_in_loop = st.checkbox(
        "Enable Human-in-the-Loop",
        value=st.session_state.human_in_loop,
        help="Enable human review for low-confidence responses"
    )
    
    if human_in_loop != st.session_state.human_in_loop:
        st.session_state.human_in_loop = human_in_loop
        st.session_state.system = MultiAgentSystem(enable_human_in_loop=human_in_loop)
        st.success("System updated!")
    
    # Initialize system if needed
    if st.session_state.system is None:
        st.session_state.system = MultiAgentSystem(enable_human_in_loop=human_in_loop)
    
    st.divider()
    
    # Data Management Section
    st.subheader("📚 Data Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Check Status", use_container_width=True):
            st.session_state.query_input = "Check vectorstore status"
    with col2:
        if st.button("🔄 Update Data", use_container_width=True):
            st.session_state.query_input = "Update vectorstore with new data"
    
    if st.button("📥 Load Data", use_container_width=True):
        st.session_state.query_input = "Load complaint data into vectorstore"
    
    st.divider()
    
    # System Metrics
    st.subheader("📊 System Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.metrics["total_queries"])
    with col2:
        st.metric("Avg Latency", f"{st.session_state.metrics['avg_latency']:.2f}s")
    
    # Routing Statistics
    if st.session_state.metrics["routing_stats"]:
        st.subheader("🔀 Agent Routing")
        routing_data = st.session_state.metrics["routing_stats"]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(routing_data.keys()), 
                y=list(routing_data.values()),
                text=list(routing_data.values()),
                textposition='auto',
            )
        ])
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            yaxis_title="Queries Routed"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Example Queries
    st.subheader("💡 Example Queries")
    
    # Analysis queries
    with st.expander("📊 Analysis Queries"):
        analysis_queries = [
            "What are the top complaint categories?",
            "Show trends in credit card complaints",
            "Which states have the most complaints?",
            "Analyze complaint patterns by company"
        ]
        for query in analysis_queries:
            if st.button(f"→ {query}", key=f"analysis_{query}"):
                st.session_state.query_input = query
    
    # Search queries
    with st.expander("🔍 Search Queries"):
        search_queries = [
            "Tell me about Bank of America mortgage issues",
            "Search for Wells Fargo complaints",
            "Find credit card billing disputes",
            "Show me student loan servicing problems"
        ]
        for query in search_queries:
            if st.button(f"→ {query}", key=f"search_{query}"):
                st.session_state.query_input = query
    
    # Data queries
    with st.expander("📚 Data Management"):
        data_queries = [
            "Check vectorstore status",
            "Load new complaint data",
            "Optimize chunking strategy",
            "Update vectorstore"
        ]
        for query in data_queries:
            if st.button(f"→ {query}", key=f"data_{query}"):
                st.session_state.query_input = query

# Main Content Area
st.title("🤖 Financial Complaints Multi-Agent System")
st.caption("Powered by 6 specialized agents: Coordinator, Trend, RAG, Response, Human Feedback, and Document Loader")

# Info box
with st.expander("ℹ️ About this System", expanded=False):
    st.markdown("""
    This multi-agent system analyzes consumer financial complaints using:
    
    **🎯 Coordinator Agent** - Routes queries to the appropriate specialist
    **📊 Trend Agent** - Performs statistical analysis and identifies patterns
    **🔍 RAG Agent** - Searches complaint database and web for specific information
    **📝 Response Agent** - Formats and quality-checks responses
    **👤 Human Feedback Agent** - Reviews low-confidence responses (when enabled)
    **📚 Document Loader Agent** - Manages data ingestion and vectorstore
    
    The system uses ChromaDB for vector storage, OpenAI for embeddings and LLM, 
    and multiple tools including web search and calculators.
    """)

# Chat Interface
chat_container = st.container()

with chat_container:
    # Display chat history
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.write(msg.content)

# Query Input
query = st.chat_input("💬 Ask about financial complaints, trends, or data management...")

# Handle example query button clicks
if "query_input" in st.session_state and st.session_state.query_input:
    query = st.session_state.query_input
    st.session_state.query_input = None

if query:
    # Add user message
    with st.chat_message("human"):
        st.write(query)
    msgs.add_user_message(query)
    
    # Process with multi-agent system
    with st.chat_message("assistant"):
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Track processing stages
            stages = [
                (0.2, "🔄 Routing query to appropriate agent..."),
                (0.5, "⚙️ Processing with specialized agent..."),
                (0.8, "✨ Formatting response..."),
                (1.0, "✅ Complete!")
            ]
            
            start_time = time.time()
            
            try:
                # Update progress
                for progress, status in stages[:-1]:
                    progress_bar.progress(progress)
                    status_text.text(status)
                    time.sleep(0.1)  # Brief pause for UI
                
                # Process query
                result = st.session_state.system.process_query(
                    query,
                    st.session_state.session_id
                )
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("✅ Complete!")
                time.sleep(0.2)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Get results
                response = result.get("final_response", "No response generated")
                elapsed = time.time() - start_time
                confidence = result.get("confidence_score", 0)
                routing = result.get("routing_decision", "Unknown")
                
                # Display response
                st.markdown(response)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.caption(f"⏱️ {elapsed:.2f}s")
                with col2:
                    st.caption(f"🎯 {confidence:.0%} confidence")
                with col3:
                    st.caption(f"🔀 {routing}")
                with col4:
                    tools = result.get("tool_results", {}).get("tools_used", [])
                    if tools:
                        st.caption(f"🛠️ {len(tools)} tools")
                
                # Show human feedback if incorporated
                if result.get("human_feedback"):
                    st.info(f"📝 Human feedback incorporated: {result['human_feedback']}")
                
                # Update metrics
                st.session_state.metrics["total_queries"] += 1
                st.session_state.metrics["avg_latency"] = (
                    (st.session_state.metrics["avg_latency"] * (st.session_state.metrics["total_queries"] - 1) + elapsed) 
                    / st.session_state.metrics["total_queries"]
                )
                
                # Update routing stats
                if routing in st.session_state.metrics["routing_stats"]:
                    st.session_state.metrics["routing_stats"][routing] += 1
                
                # Add to chat history
                msgs.add_ai_message(response)
                
            except Exception as e:
                # Clear progress on error
                progress_bar.empty()
                status_text.empty()
                
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                msgs.add_ai_message(error_msg)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔧 Built with LangGraph & LangChain")
with col2:
    st.caption("💾 Powered by ChromaDB & OpenAI")
with col3:
    st.caption("🎯 6 Specialized Agents")
