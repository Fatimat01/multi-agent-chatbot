#!/usr/bin/env python3
"""
Enhanced data ingestion script using DocumentLoaderAgent.
Handles CSV loading, intelligent chunking, and vector store management.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent.document_loader_agent import DocumentLoaderAgent
from utils import setup_logging

# Load environment variables
load_dotenv()


class DataIngestionPipeline:
    """Enhanced data ingestion pipeline with progress tracking and validation."""
    
    def __init__(self, 
                 data_path: str = "data/complaints.csv",
                 vectorstore_path: str = "chroma_db",
                 log_path: str = "logs/ingestion.log"):
        """Initialize ingestion pipeline."""
        # Setup logging
        os.makedirs("logs", exist_ok=True)
        setup_logging(log_path)
        
        # Initialize document loader agent
        self.loader = DocumentLoaderAgent(
            data_path=data_path,
            vectorstore_path=vectorstore_path,
            chunk_size=2000,  # Optimal for complaint narratives
            chunk_overlap=500,  # Good overlap for context preservation
            batch_size=500
        )
        
        self.data_path = data_path
        self.vectorstore_path = vectorstore_path
        
    def run_full_ingestion(self):
        """Run complete data ingestion pipeline."""
        logging.info("="*60)
        logging.info("Starting Enhanced Data Ingestion Pipeline")
        logging.info(f"Timestamp: {datetime.now()}")
        logging.info("="*60)
        
        # Step 1: Check current status
        print("\n📊 Checking current vectorstore status...")
        status = self.loader.check_vectorstore_status()
        self._display_status(status)
        
        # Step 2: Analyze data for optimal chunking
        print("\n🔍 Analyzing data characteristics...")
        optimization = self.loader.optimize_chunking_strategy()
        self._display_optimization(optimization)
        
        # Step 3: Load and chunk data
        print("\n📥 Loading and chunking data...")
        result = self.loader.load_and_chunk_data()
        self._display_result(result)
        
        # Step 4: Verify ingestion
        print("\n✅ Verifying ingestion...")
        final_status = self.loader.check_vectorstore_status()
        self._display_final_status(final_status)
        
        # Step 5: Generate summary report
        self._generate_summary_report(status, result, final_status)
        
    def update_existing(self):
        """Update existing vectorstore with new data."""
        print("\n🔄 Checking for updates...")
        result = self.loader.update_vectorstore()
        
        if result["status"] == "no_update":
            print("ℹ️  No updates needed - data file hasn't changed")
        else:
            self._display_result(result)
    
    def validate_data(self):
        """Validate data file before ingestion."""
        print(f"\n🔍 Validating data file: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"❌ Error: Data file not found at {self.data_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(self.data_path) / (1024 * 1024)  # MB
        print(f"📁 File size: {file_size:.2f} MB")
        
        # Try to load a sample
        try:
            import pandas as pd
            df = pd.read_csv(self.data_path, nrows=5)
            print(f"📋 Columns found: {', '.join(df.columns)}")
            print(f"✅ Data validation successful!")
            return True
        except Exception as e:
            print(f"❌ Error validating data: {e}")
            return False
    
    def _display_status(self, status: dict):
        """Display current vectorstore status."""
        if status["status"] == "ready":
            print(f"✅ Vectorstore is ready")
            print(f"📄 Documents: {status['document_count']}")
            print(f"📁 Location: {status['vectorstore_path']}")
            if status.get('sample_metadata_keys'):
                print(f"🏷️  Metadata fields: {', '.join(status['sample_metadata_keys'][:5])}")
        else:
            print(f"⚠️  Vectorstore status: {status['status']}")
            print(f"   {status.get('message', '')}")
    
    def _display_optimization(self, optimization: dict):
        """Display chunking optimization results."""
        if optimization["status"] == "success":
            analysis = optimization["analysis"]
            print(f"📊 Document Analysis:")
            print(f"   - Average length: {analysis['avg_doc_length']} chars")
            print(f"   - Range: {analysis['min_doc_length']} - {analysis['max_doc_length']} chars")
            print(f"   - Current chunk size: {analysis['current_chunk_size']}")
            print(f"   - Recommended: {analysis['recommended_chunk_size']} "
                  f"(overlap: {analysis['recommended_overlap']})")
    
    def _display_result(self, result: dict):
        """Display ingestion result."""
        if result["status"] == "success":
            print(f"✅ Ingestion successful!")
            print(f"📄 Documents processed: {result.get('documents_loaded', 'N/A')}")
            print(f"🧩 Chunks created: {result.get('chunks_created', 'N/A')}")
        else:
            print(f"❌ Ingestion failed: {result.get('message', 'Unknown error')}")
    
    def _display_final_status(self, status: dict):
        """Display final vectorstore status."""
        if status["status"] == "ready":
            print(f"✅ Vectorstore verified")
            print(f"📊 Final document count: {status['document_count']}")
            
            stats = status.get("stats", {})
            if stats.get("duplicates_skipped", 0) > 0:
                print(f"🔁 Duplicates skipped: {stats['duplicates_skipped']}")
    
    def _generate_summary_report(self, initial_status: dict, result: dict, final_status: dict):
        """Generate and save summary report."""
        report_path = f"logs/ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("INGESTION SUMMARY REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Data file: {self.data_path}\n")
            f.write(f"Vectorstore: {self.vectorstore_path}\n\n")
            
            f.write("RESULTS:\n")
            f.write(f"- Initial documents: {initial_status.get('document_count', 0)}\n")
            f.write(f"- Documents loaded: {result.get('documents_loaded', 0)}\n")
            f.write(f"- Chunks created: {result.get('chunks_created', 0)}\n")
            f.write(f"- Final documents: {final_status.get('document_count', 0)}\n")
            
            if final_status.get("stats"):
                stats = final_status["stats"]
                f.write(f"\nSTATISTICS:\n")
                f.write(f"- Duplicates skipped: {stats.get('duplicates_skipped', 0)}\n")
                f.write(f"- Load time: {stats.get('load_time', 'N/A')}\n")
        
        print(f"\n📄 Summary report saved to: {report_path}")


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced data ingestion for financial complaints chatbot"
    )
    
    parser.add_argument(
        "--data-path",
        default="data/complaints.csv",
        help="Path to CSV data file"
    )
    
    parser.add_argument(
        "--vectorstore-path",
        default="chroma_db",
        help="Path to ChromaDB vectorstore"
    )
    
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing vectorstore instead of full reload"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data without ingestion"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DataIngestionPipeline(
        data_path=args.data_path,
        vectorstore_path=args.vectorstore_path
    )
    
    # Execute based on arguments
    if args.validate_only:
        pipeline.validate_data()
    elif args.update:
        pipeline.update_existing()
    else:
        # Validate first
        if pipeline.validate_data():
            pipeline.run_full_ingestion()
        else:
            print("\n❌ Data validation failed. Please check your data file.")
            sys.exit(1)


if __name__ == "__main__":
    main()
