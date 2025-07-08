from utils import Message, timer
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import logging
from dotenv import load_dotenv
import os
load_dotenv()


class RAGAgent:
    def __init__(self):
        try:
            embedding = OpenAIEmbeddings()
            vectorstore_path = "../../chroma_db"
            
            if os.path.exists(vectorstore_path):
                vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding)
                
                # Test the collection
                collection = vectorstore._collection
                doc_count = collection.count()
                logging.info(f"Loaded vector store from {vectorstore_path} with {doc_count} documents")
                
                # Create retriever
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
                
                # Initialize LLM
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                
                # Define system prompt for QA
                system_prompt = (
                    "You are an assistant for question-answering tasks analyzing consumer financial complaints data. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer or if the context doesn't "
                    "contain relevant information, say 'NO_RELEVANT_DATA_FOUND'. "
                    "Be specific and mention company names, products, and issues when relevant. "
                    "Use three sentences maximum and keep the answer concise."
                    "\n\n"
                    "{context}"
                )
                
                # Create prompt template
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )
                
                # Create question-answer chain
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                
                # Create RAG chain
                self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                self.vectorstore = vectorstore
                
            else:
                raise FileNotFoundError(f"Vector store not found at {vectorstore_path}")
                
        except Exception as e:
            logging.error(f"Error initializing RAG components: {e}")
            raise
        
        # Initialize web search only if API key is available
        if os.getenv("SERPAPI_API_KEY"):
            self.web = SerpAPIWrapper()
        else:
            self.web = None
            logging.warning("SERPAPI_API_KEY not found. Web search will be disabled.")

    def _format_web_results(self, raw_results: str) -> str:
        """Format web search results into clean text."""
        try:
            if raw_results.startswith('[') and raw_results.endswith(']'):
                import ast
                results_list = ast.literal_eval(raw_results)
                
                formatted = "**Web Search Results:**\n"
                for i, result in enumerate(results_list[:5], 1):
                    result = result.strip()
                    if len(result) > 200:
                        result = result[:197] + "..."
                    formatted += f"{i}. {result}\n"
                return formatted
            else:
                return raw_results
        except Exception as e:
            logging.error(f"Error formatting web results: {e}")
            return raw_results

    @timer
    def process(self, state: dict) -> dict:
        """Process RAG request."""
        query = state.get("query", "")
        
        # Initialize response structure
        found_in_db = False
        vector_ans = ""
        web_ans = ""
        
        try:
            logging.info(f"Processing query: {query}")
            
            # Use the RAG chain
            result = self.rag_chain.invoke({"input": query})
            
            # Extract answer and source documents
            raw_answer = result.get("answer", "")
            source_docs = result.get("context", [])
            
            logging.info(f"Retrieved {len(source_docs)} documents")
            
            # Check if we found relevant data
            if "NO_RELEVANT_DATA_FOUND" in raw_answer or not source_docs:
                found_in_db = False
                vector_ans = "No relevant information found in the complaints database."
                
                # Only use web search as fallback
                if self.web:
                    logging.info("No relevant data in vector store, falling back to web search")
                    try:
                        raw_web_results = self.web.run(query)
                        web_ans = self._format_web_results(raw_web_results)
                    except Exception as e:
                        logging.error(f"Error in web search: {e}")
                        web_ans = "Web search also failed to find relevant information."
                else:
                    web_ans = "Web search is not available."
            else:
                found_in_db = True
                vector_ans = raw_answer
                
                # Add source information
                if source_docs:
                    companies = set()
                    products = set()
                    issues = set()
                    
                    for doc in source_docs[:5]:
                        if hasattr(doc, 'metadata'):
                            metadata = doc.metadata
                            if 'company' in metadata and metadata['company'] != 'Unknown':
                                companies.add(metadata['company'])
                            if 'product' in metadata and metadata['product'] != 'Unknown':
                                products.add(metadata['product'])
                            if 'issue' in metadata and metadata['issue'] != 'Unknown':
                                issues.add(metadata['issue'])
                    
                    # Add metadata summary
                    metadata_parts = []
                    if companies:
                        metadata_parts.append(f"**Companies:** {', '.join(list(companies)[:5])}")
                    if products:
                        metadata_parts.append(f"**Products:** {', '.join(list(products)[:5])}")
                    if issues:
                        metadata_parts.append(f"**Issues:** {', '.join(list(issues)[:3])}")
                    
                    if metadata_parts:
                        vector_ans += "\n\n📊 " + " | ".join(metadata_parts)
                        vector_ans += f"\n📄 Based on {len(source_docs)} relevant complaints"
                
                # No web search needed when found in DB
                web_ans = ""
            
        except Exception as e:
            logging.error(f"Error in RAG processing: {e}", exc_info=True)
            found_in_db = False
            vector_ans = f"Error processing query: {str(e)}"
            web_ans = ""
        
        content = {
            "found_in_db": found_in_db,
            "vector_answer": vector_ans,
            "web_fallback": web_ans
        }
        
        # Add message as dict to state
        msg = Message("RAGAgent", "ResponseAgent", content)
        state["messages"].append(msg.to_dict())
        
        return state