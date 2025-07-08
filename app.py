# multi aggent streamlit app for financial consumer complaints

import os
import time

import streamlit as st
from dotenv import load_dotenv

from utils import setup_logging, Message
from multi_agent_app import build_graph
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

#  Logging and Environment Setup
load_dotenv()  # load OPENAI_API_KEY, SERPAPI_API_KEY, etc.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "financial-consumer-chatbot"
os.environ["LANGSMITH_TIMEOUT_MS"] = "5000"
os.environ["LANGSMITH_MAX_RETRIES"] = "2"
setup_logging(log_path="logs/app.log")

# Multi‐Agent Graph 
graph = build_graph()

# initialize Streamlit Chat History
msgs = StreamlitChatMessageHistory()
if not msgs.messages:
    msgs.add_ai_message("👋 Hi there! Ask me anything about financial consumer complaints.")

# Streamlit Page Setup 
st.set_page_config(page_title="Consumer Complaint Explorer", layout="wide")
st.title("Consumer Complaint Explorer")
st.caption("Ask questions about real financial consumer complaints in natural language.")

# Render Existing Chat
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Accept User Query
query = st.chat_input("💬 Your question…")
if query:
    # Add user message
    st.chat_message("human").write(query)
    msgs.add_user_message(query)

    # Invoke multi‐agent workflow
    with st.chat_message("ai"):
        start = time.time()
        with st.spinner("🤖 Thinking…"):
            # Wrap user input into our MCP Message
            m = Message(sender="User", receiver="CoordinatorAgent", content={"query": query})
            result_msg = graph.run(m)
            response = result_msg.content["response"]

            # Append and display
            msgs.add_ai_message(response)
            st.markdown(response)

        # display timing
        elapsed = round(time.time() - start, 2)
        st.caption(f"🔍 Retrieved in {elapsed}s")

