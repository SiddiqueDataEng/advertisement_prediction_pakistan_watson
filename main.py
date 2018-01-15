# -*- coding: utf-8 -*-
"""
Main Entry Point for Advertisement Prediction System
Compatible with Python 3.6+ (2018 standards)
"""

import sys
import os
import argparse
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_generator import PakistaniAdDataGenerator
from data_processor import AdvertisementDataProcessor
from model_trainer import AdvertisementModelTrainer
from watson_integration import get_watson_client

def generate_data(num_records: int = 1000) -> str:
    """Generate Pakistani advertisement dataset"""
    print(f"Generating {num_records} records of Pakistani advertisement data...")
    
    generator = PakistaniAdDataGenerator(num_records)
    filename = generator.save_dataset()
    
    print(f"Dataset generated successfully: {filename}")
    return filename

def process_data(filepath: str) -> Dict[str, Any]:
    """Process and prepare data for machine learning"""
    print(f"Processing data from {filepath}...")
    
    processor = AdvertisementDataProcessor()
    results = processor.process_pipeline(filepath)
    
    if 'error' in results:
        print(f"Error processing data: {results['error']}")
        return {}
    
    print("Data processing completed successfully!")
    return results

def train_models(processed_data: Dict[str, Any], tune_hyperparameters: bool = False) -> str:
    """Train machine learning models"""
    print("Training machine learning models...")
    
    trainer = AdvertisementModelTrainer()
    
    # Train all models
    model_results = trainer.train_all_models(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_test'],
        processed_data['y_test'],
        tune_hyperparameters=tune_hyperparameters
    )
    
    # Save best model
    model_path = trainer.save_model()
    
    # Generate report
    report = trainer.generate_model_report()
    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    return model_path

def test_watson_integration(api_key: str = None) -> bool:
    """Test Watson ML integration"""
    print("Testing Watson ML integration...")
    
    client = get_watson_client(api_key)
    result = client.test_connection()
    
    if result['success']:
        print("✅ Watson ML integration test successful!")
        if 'mock' in result:
            print("ℹ️ Using mock service (API key not configured)")
        return True
    else:
        print(f"❌ Watson ML integration test failed: {result['error']}")
        return False

def run_full_pipeline(num_records: int = 1000, tune_hyperparameters: bool = False, 
                     api_key: str = None) -> Dict[str, str]:
    """Run the complete pipeline from data generation to model training"""
    print("Starting full advertisement prediction pipeline...")
    print("="*60)
    
    results = {}
    
    try:
        # Step 1: Generate data
        print("\n📊 Step 1: Generating Data")
        data_file = generate_data(num_records)
        results['data_file'] = data_file
        
        # Step 2: Process data
        print("\n🔧 Step 2: Processing Data")
        processed_data = process_data(data_file)
        if not processed_data:
            raise Exception("Data processing failed")
        
        # Step 3: Train models
        print("\n🤖 Step 3: Training Models")
        model_path = train_models(processed_data, tune_hyperparameters)
        results['model_path'] = model_path
        
        # Step 4: Test Watson integration
        print("\n🔗 Step 4: Testing Watson Integration")
        watson_success = test_watson_integration(api_key)
        results['watson_integration'] = 'success' if watson_success else 'failed'
        
        print("\n✅ Full pipeline completed successfully!")
        print(f"📁 Data file: {data_file}")
        print(f"🤖 Model file: {model_path}")
        print(f"🔗 Watson integration: {results['watson_integration']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return {'error': str(e)}

def run_streamlit_app():
    """Launch the Streamlit web application"""
    print("Launching Streamlit web application...")
    
    try:
        import streamlit.cli
        sys.argv = ["streamlit", "run", "streamlit_app.py"]
        streamlit.cli.main()
    except ImportError:
        print("Streamlit not installed. Please install with: pip install streamlit")
    except Exception as e:
        print(f"Error launching Streamlit app: {str(e)}")

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description="Advertisement Prediction System - Pakistan Market",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --generate-data 1000
  python main.py --process-data data/advertising_pakistan.csv
  python main.py --train-models --tune
  python main.py --test-watson --api-key YOUR_API_KEY
  python main.py --full-pipeline --records 2000 --tune
  python main.py --run-app
        """
    )
    
    # Command options
    parser.add_argument('--generate-data', type=int, metavar='N',
                       help='Generate N records of Pakistani advertisement data')
    
    parser.add_argument('--process-data', type=str, metavar='FILE',
                       help='Process data from specified CSV file')
    
    parser.add_argument('--train-models', action='store_true',
                       help='Train machine learning models')
    
    parser.add_argument('--test-watson', action='store_true',
                       help='Test Watson ML integration')
    
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete pipeline (generate, process, train, test)')
    
    parser.add_argument('--run-app', action='store_true',
                       help='Launch Streamlit web application')
    
    # Options
    parser.add_argument('--records', type=int, default=1000,
                       help='Number of records to generate (default: 1000)')
    
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    
    parser.add_argument('--api-key', type=str,
                       help='IBM Watson ML API key')
    
    parser.add_argument('--data-file', type=str, default='data/advertising_pakistan.csv',
                       help='Data file path (default: data/advertising_pakistan.csv)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Execute commands
    try:
        if args.generate_data:
            generate_data(args.generate_data)
        
        elif args.process_data:
            process_data(args.process_data)
        
        elif args.train_models:
            # Need processed data
            if not os.path.exists(args.data_file):
                print(f"Data file not found: {args.data_file}")
                print("Generate data first with --generate-data")
                return
            
            processed_data = process_data(args.data_file)
            if processed_data:
                train_models(processed_data, args.tune)
        
        elif args.test_watson:
            test_watson_integration(args.api_key)
        
        elif args.full_pipeline:
            run_full_pipeline(args.records, args.tune, args.api_key)
        
        elif args.run_app:
            run_streamlit_app()
        
        else:
            print("No valid command specified. Use --help for options.")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()