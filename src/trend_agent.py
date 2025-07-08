from utils import Message, timer
import pandas as pd
import logging
import os

class TrendAgent:
    """Performs batch statistics on complaints via Pandas."""
    def __init__(self, data_path: str = "data/complaints.csv"):
        self.data_path = data_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load data with error handling."""
        try:
            if os.path.exists(self.data_path):
                self.df = pd.read_csv(self.data_path)
                logging.info(f"Loaded {len(self.df)} records from {self.data_path}")
            else:
                logging.warning(f"Data file not found: {self.data_path}")
                # Create dummy data for testing
                self.df = pd.DataFrame({
                    "Product": ["Credit card", "Mortgage", "Student loan", "Auto loan", "Bank account"],
                    "Consumer complaint narrative": ["Issue 1", "Issue 2", "Issue 3", "Issue 4", "Issue 5"]
                })
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()

    @timer
    def process(self, state: dict) -> dict:
        """Process trend analysis request."""
        query = state.get("query", "")
        
        if self.df.empty:
            content = {
                "summary": "❌ No data available for trend analysis",
                "details": {}
            }
        else:
            # Perform analysis based on query
            if "product" in query.lower():
                top = (
                    self.df.groupby("Product")
                    .size()
                    .nlargest(5)
                    .to_dict()
                )
                summary = "🔝 Top 5 products by complaint volume:"
            else:
                # Default analysis
                top = (
                    self.df.groupby("Product")
                    .size()
                    .nlargest(5)
                    .to_dict()
                )
                summary = "📊 Top 5 complaint categories:"
            
            content = {
                "summary": summary,
                "details": top
            }
        
        # Add message as dict to state
        msg = Message("TrendAgent", "ResponseAgent", content)
        state["messages"].append(msg.to_dict())
        
        return state