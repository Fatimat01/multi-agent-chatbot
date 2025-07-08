from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools import Tool
from utils import Message, timer
from dotenv import load_dotenv

load_dotenv()


class RAGAgent:
    """Enhanced RAG Agent with multiple retrieval and search tools."""
    
    def __init__(self):
        """Initialize RAG components and tools."""
        self.tools = {}
        self._initialize_vector_store()
        self._initialize_search_tools()
        self._initialize_llm_chain()
        
    def _initialize_vector_store(self):
        """Initialize vector store for complaint data."""
        try:
            embedding = OpenAIEmbeddings()
            vectorstore_path = "chroma_db"
            
            if os.path.exists(vectorstore_path):
                self.vectorstore = Chroma(
                    persist_directory=vectorstore_path, 
                    embedding_function=embedding
                )
                
                # Test the collection
                collection = self.vectorstore._collection
                doc_count = collection.count()
                logging.info(f"Loaded vector store with {doc_count} documents")
                
                # Create retriever tool
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
                
                self.tools["vector_search"] = Tool(
                    name="complaint_database",
                    description="Search internal complaint database",
                    func=self._vector_search
                )
                
            else:
                raise FileNotFoundError(f"Vector store not found at {vectorstore_path}")
                
        except Exception as e:
            logging.error(f"Error initializing vector store: {e}")
            self.vectorstore = None
            self.retriever = None
    
    def _initialize_search_tools(self):
        """Initialize external search tools."""
        # Tool 1: SerpAPI (if available)
        if os.getenv("SERPAPI_API_KEY"):
            try:
                self.tools["web_search"] = Tool(
                    name="web_search",
                    description="Search the web for current information",
                    func=SerpAPIWrapper().run
                )
            except Exception as e:
                logging.warning(f"SerpAPI initialization failed: {e}")
        
        # Tool 2: DuckDuckGo as fallback
        try:
            self.tools["ddg_search"] = Tool(
                name="duckduckgo_search",
                description="Alternative web search using DuckDuckGo",
                func=DuckDuckGoSearchRun().run
            )
        except Exception as e:
            logging.warning(f"DuckDuckGo initialization failed: {e}")
        
        # Tool 3: Calculator tool for statistics
        self.tools["calculator"] = Tool(
            name="calculator",
            description="Perform calculations and statistics",
            func=self._calculate
        )
        
        # Tool 4: Date/Time tool
        self.tools["datetime"] = Tool(
            name="datetime",
            description="Get current date and time information",
            func=lambda x: f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def _initialize_llm_chain(self):
        """Initialize LLM and chains."""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Enhanced system prompt for financial complaints analysis
        self.system_prompt = """You are an expert assistant specializing in analyzing consumer financial complaints data.
        
        You have access to a comprehensive database of consumer complaints filed with the Consumer Financial Protection Bureau (CFPB).
        
        Context from the complaints database:
        {context}
        
        When answering questions:
        1. **Prioritize information from the complaint database** - this is your primary source
        2. **Be specific** - mention exact company names, products, issues, dates, and locations when available
        3. **Provide quantitative insights** - include numbers, percentages, and trends when relevant
        4. **Structure your response** clearly with:
           - A direct answer to the question
           - Supporting evidence from the complaints
           - Any relevant patterns or trends
        5. **Acknowledge limitations** - if the database doesn't contain relevant information, clearly state this
        
        Key fields in the complaint data:
        - Company: The financial institution involved
        - Product: Type of financial product (credit card, mortgage, student loan, etc.)
        - Issue: The specific problem reported
        - Consumer complaint narrative: Detailed description from the consumer
        - State/Zip code: Geographic information
        - Date received: When the complaint was filed
        - Company response: How the company addressed the complaint
        
        If the question requires current information not in the database (like today's date or recent events), 
        I will use supplementary tools to provide accurate, up-to-date information.
        
        Available supplementary tools: {tools}
        
        Remember: Your goal is to help users understand patterns in consumer complaints and provide actionable insights 
        about financial products and services based on real consumer experiences."""
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
        ])
        
        if self.retriever:
            # Create question-answer chain
            question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            
            # Create RAG chain
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
    
    def _vector_search(self, query: str) -> str:
        """Perform vector similarity search."""
        if not self.retriever:
            return "Vector database not available"
        
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found"
        
        # Format results
        results = []
        for i, doc in enumerate(docs[:5]):
            metadata = doc.metadata
            snippet = doc.page_content[:200]
            results.append(f"{i+1}. {snippet}... [Company: {metadata.get('Company', 'N/A')}]")
        
        return "\n".join(results)
    
    def _calculate(self, expression: str) -> str:
        """Safe calculator function."""
        try:
            # Remove any dangerous characters
            safe_expr = expression.replace("__", "").replace("import", "").replace("exec", "")
            result = eval(safe_expr, {"__builtins__": {}}, {})
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    @timer
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process RAG request using multiple tools."""
        query = state.get("query", "")
        
        # Track tool usage
        tool_results = {
            "tools_used": [],
            "tool_outputs": {}
        }
        
        try:
            # Step 1: Try vector search first
            if self.retriever:
                logging.info(f"Processing query with RAG: {query}")
                result = self.rag_chain.invoke({"input": query})
                
                raw_answer = result.get("answer", "")
                source_docs = result.get("context", [])
                
                tool_results["tools_used"].append("vector_search")
                tool_results["tool_outputs"]["vector_search"] = {
                    "doc_count": len(source_docs),
                    "found_relevant": "NO_RELEVANT_DATA_FOUND" not in raw_answer
                }
                
                # If found good results, enhance them
                if source_docs and "NO_RELEVANT_DATA_FOUND" not in raw_answer:
                    answer = self._enhance_answer(raw_answer, source_docs)
                else:
                    # Step 2: Use web search as fallback
                    answer = self._search_fallback(query, tool_results)
            else:
                # No vector store, use search tools
                answer = self._search_fallback(query, tool_results)
            
            # Step 3: Add current date/time if relevant
            if any(word in query.lower() for word in ["today", "current", "now", "latest"]):
                date_info = self.tools["datetime"].func("")
                answer += f"\n\n📅 {date_info}"
                tool_results["tools_used"].append("datetime")
            
            # Calculate confidence based on tool results
            confidence = self._calculate_confidence(tool_results)
            
        except Exception as e:
            logging.error(f"Error in RAG processing: {e}", exc_info=True)
            answer = f"Error processing query: {str(e)}"
            confidence = 0.0
        
        # Prepare response
        content = {
            "answer": answer,
            "tools_used": tool_results["tools_used"],
            "confidence": confidence
        }
        
        # Update state
        state["tool_results"] = tool_results
        state["confidence_score"] = confidence
        
        # Add message
        msg = Message("RAGAgent", "ResponseAgent", content)
        state["messages"].append(msg.to_dict())
        
        return state
    
    def _enhance_answer(self, answer: str, docs: List[Any]) -> str:
        """Enhance answer with metadata from documents."""
        # Extract unique metadata
        companies = set()
        products = set()
        issues = set()
        states = set()
        
        for doc in docs[:10]:
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                if 'Company' in metadata:
                    companies.add(metadata['Company'])
                if 'Product' in metadata:
                    products.add(metadata['Product'])
                if 'Issue' in metadata:
                    issues.add(metadata['Issue'])
                if 'State' in metadata:
                    states.add(metadata['State'])
        
        # Build enhanced answer
        enhanced = answer
        
        metadata_parts = []
        if companies:
            metadata_parts.append(f"**Companies:** {', '.join(list(companies)[:5])}")
        if products:
            metadata_parts.append(f"**Products:** {', '.join(list(products)[:5])}")
        if issues:
            metadata_parts.append(f"**Issues:** {', '.join(list(issues)[:3])}")
        if states:
            metadata_parts.append(f"**States:** {', '.join(list(states)[:5])}")
        
        if metadata_parts:
            enhanced += "\n\n📊 **Related Information:**\n" + "\n".join(metadata_parts)
            enhanced += f"\n\n📄 Based on {len(docs)} relevant complaints"
        
        return enhanced
    
    def _search_fallback(self, query: str, tool_results: Dict) -> str:
        """Use web search tools as fallback."""
        answer_parts = ["No relevant information found in complaint database.\n"]
        
        # Try web search tools
        for tool_name in ["web_search", "ddg_search"]:
            if tool_name in self.tools:
                try:
                    results = self.tools[tool_name].func(query)
                    if results:
                        answer_parts.append(f"\n🌐 **Web Search Results ({tool_name}):**")
                        answer_parts.append(self._format_search_results(results))
                        tool_results["tools_used"].append(tool_name)
                        tool_results["tool_outputs"][tool_name] = {"found_results": True}
                        break
                except Exception as e:
                    logging.error(f"Error with {tool_name}: {e}")
        
        return "\n".join(answer_parts)
    
    def _format_search_results(self, results: str) -> str:
        """Format search results for display."""
        # Limit length and format
        if isinstance(results, str):
            lines = results.split('\n')[:5]  # First 5 results
            formatted = []
            for i, line in enumerate(lines, 1):
                if len(line) > 200:
                    line = line[:197] + "..."
                formatted.append(f"{i}. {line}")
            return "\n".join(formatted)
        return str(results)[:500]
    
    def _calculate_confidence(self, tool_results: Dict) -> float:
        """Calculate confidence score based on tool results."""
        confidence = 0.5  # Base confidence
        
        # Boost for successful vector search
        if "vector_search" in tool_results["tool_outputs"]:
            if tool_results["tool_outputs"]["vector_search"].get("found_relevant"):
                confidence += 0.4
        
        # Small boost for web search
        if any(tool in tool_results["tools_used"] for tool in ["web_search", "ddg_search"]):
            confidence += 0.1
        
        return min(confidence, 0.95)  # Cap at 0.95
