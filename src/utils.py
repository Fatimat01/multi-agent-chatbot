import logging
import hashlib
import time
import os
from typing import Dict, Any

class Message:
    """Simple MCP message for agent‐to‐agent communication."""
    def __init__(self, sender: str, receiver: str, content: dict):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp
        }

def setup_logging(log_path="logs/app.log"):
    """Sets up file + console logging."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="a",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger("").addHandler(console)

def compute_sha256(text: str) -> str:
    """Computes SHA256 hash of a string (for deduplication)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def format_source_metadata(doc) -> str:
    """Formats metadata from a document for UI display."""
    company = doc.metadata.get("Company", "Unknown")
    product = doc.metadata.get("Product", "Unknown")
    return f"**{company}** – {product}"

def count_tokens(text: str) -> int:
    """Approximate token count for rate‐limit debugging."""
    return len(text.split())

def timer(func):
    """Decorator to log execution time of agent methods."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{func.__qualname__} took {elapsed:.2f}s")
        return result
    return wrapper