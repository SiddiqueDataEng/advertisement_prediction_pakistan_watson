# -*- coding: utf-8 -*-
"""
Utility Functions for Advertisement Prediction System
Compatible with Python 3.6+ (2018 standards)
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileManager:
    """Handle file operations and directory management"""
    
    @staticmethod
    def ensure_directory(filepath: str) -> str:
        """Ensure directory exists for the given filepath"""
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        return directory
    
    @staticmethod
    def save_json(data: Dict, filepath: str) -> bool:
        """Save dictionary as JSON file"""
        try:
            FileManager.ensure_directory(filepath)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"JSON saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load JSON file as dictionary"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"JSON loaded from: {filepath}")
            return data
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {filepath}")
            return {}
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {str(e)}")
            return {}
    
    @staticmethod
    def save_pickle(obj: Any, filepath: str) -> bool:
        """Save object as pickle file"""
        try:
            FileManager.ensure_directory(filepath)
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            logger.info(f"Pickle saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving pickle to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """Load pickle file"""
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            logger.info(f"Pickle loaded from: {filepath}")
            return obj
        except FileNotFoundError:
            logger.warning(f"Pickle file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error loading pickle from {filepath}: {str(e)}")
            return None

class DataValidator:
    """Validate data integrity and format"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, str]:
        """Validate DataFrame structure and content"""
        if df is None or df.empty:
            return False, "DataFrame is empty or None"
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
        
        # Check for all NaN columns
        all_nan_columns = df.columns[df.isnull().all()].tolist()
        if all_nan_columns:
            return False, f"Columns with all NaN values: {all_nan_columns}"
        
        return True, "DataFrame validation passed"
    
    @staticmethod
    def validate_numerical_range(value: float, min_val: float, max_val: float, 
                                field_name: str = "Value") -> Tuple[bool, str]:
        """Validate numerical value is within specified range"""
        try:
            value = float(value)
            if min_val <= value <= max_val:
                return True, f"{field_name} is valid"
            else:
                return False, f"{field_name} must be between {min_val} and {max_val}"
        except (ValueError, TypeError):
            return False, f"{field_name} must be a valid number"
    
    @staticmethod
    def validate_categorical(value: str, valid_categories: List[str], 
                           field_name: str = "Value") -> Tuple[bool, str]:
        """Validate categorical value is in allowed list"""
        if value in valid_categories:
            return True, f"{field_name} is valid"
        else:
            return False, f"{field_name} must be one of: {valid_categories}"

class DateTimeUtils:
    """Utility functions for date and time operations"""
    
    @staticmethod
    def generate_random_timestamp(start_year: int = 2018, end_year: int = 2018) -> str:
        """Generate random timestamp within specified year range"""
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31, 23, 59, 59)
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = np.random.randint(0, days_between)
        
        random_date = start_date + timedelta(days=random_days)
        random_hour = np.random.randint(0, 24)
        random_minute = np.random.randint(0, 60)
        random_second = np.random.randint(0, 60)
        
        final_datetime = random_date.replace(
            hour=random_hour,
            minute=random_minute,
            second=random_second
        )
        
        return final_datetime.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object"""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None
    
    @staticmethod
    def get_time_features(timestamp: datetime) -> Dict[str, int]:
        """Extract time-based features from datetime"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'quarter': (timestamp.month - 1) // 3 + 1,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_business_hour': 1 if 9 <= timestamp.hour <= 17 else 0
        }

class StatisticsCalculator:
    """Calculate various statistics and metrics"""
    
    @staticmethod
    def calculate_basic_stats(series: pd.Series) -> Dict[str, float]:
        """Calculate basic statistical measures"""
        return {
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
    
    @staticmethod
    def calculate_click_rate_by_category(df: pd.DataFrame, category_col: str, 
                                       target_col: str = 'Clicked on Ad') -> pd.DataFrame:
        """Calculate click rates by categorical variable"""
        if category_col not in df.columns or target_col not in df.columns:
            return pd.DataFrame()
        
        stats = df.groupby(category_col)[target_col].agg([
            'count', 'sum', 'mean'
        ]).round(4)
        
        stats.columns = ['Total_Ads', 'Total_Clicks', 'Click_Rate']
        stats['Non_Clicks'] = stats['Total_Ads'] - stats['Total_Clicks']
        
        return stats.reset_index()
    
    @staticmethod
    def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for numerical columns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return pd.DataFrame()
        
        return df[numerical_cols].corr(method=method)

class TextProcessor:
    """Process and analyze text data"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return str(text)
        
        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text"""
        if not isinstance(text, str):
            return []
        
        # Simple keyword extraction (can be enhanced with NLP libraries)
        words = text.lower().split()
        keywords = [word for word in words if len(word) >= min_length]
        
        return list(set(keywords))  # Remove duplicates
    
    @staticmethod
    def calculate_text_features(text: str) -> Dict[str, int]:
        """Calculate text-based features"""
        if not isinstance(text, str):
            text = str(text)
        
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'digit_count': sum(1 for c in text if c.isdigit()),
            'special_char_count': sum(1 for c in text if not c.isalnum() and not c.isspace())
        }

class ModelUtils:
    """Utilities for machine learning models"""
    
    @staticmethod
    def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive model evaluation metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
            })
        
        return metrics
    
    @staticmethod
    def create_feature_importance_df(model, feature_names: List[str]) -> pd.DataFrame:
        """Create feature importance DataFrame"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df

class ConfigManager:
    """Manage application configuration"""
    
    @staticmethod
    def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
        """Load configuration from JSON file"""
        return FileManager.load_json(config_path)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str = 'config.json') -> bool:
        """Save configuration to JSON file"""
        return FileManager.save_json(config, config_path)
    
    @staticmethod
    def get_env_variable(var_name: str, default_value: str = None) -> str:
        """Get environment variable with default fallback"""
        return os.getenv(var_name, default_value)
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, str]:
        """Validate configuration has required keys"""
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            return False, f"Missing required configuration keys: {missing_keys}"
        return True, "Configuration validation passed"

class PerformanceMonitor:
    """Monitor application performance"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start_timer(self):
        """Start performance timer"""
        self.start_time = datetime.now()
    
    def end_timer(self) -> float:
        """End performance timer and return elapsed time"""
        self.end_time = datetime.now()
        if self.start_time:
            elapsed = (self.end_time - self.start_time).total_seconds()
            return elapsed
        return 0.0
    
    def log_metric(self, name: str, value: Any):
        """Log a performance metric"""
        self.metrics[name] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all logged metrics"""
        return {
            'metrics': self.metrics,
            'total_execution_time': self.end_timer() if self.start_time else None
        }

# Convenience functions
def setup_logging(level: str = 'INFO', log_file: str = None):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        FileManager.ensure_directory(log_file)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def format_currency(amount: float, currency: str = 'PKR') -> str:
    """Format currency amount"""
    return f"{currency} {amount:,.2f}"

def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format percentage value"""
    return f"{value:.{decimal_places}f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test file manager
    test_data = {'test': 'data', 'timestamp': datetime.now()}
    FileManager.save_json(test_data, 'test_output/test.json')
    loaded_data = FileManager.load_json('test_output/test.json')
    print(f"JSON test: {loaded_data}")
    
    # Test date utils
    timestamp = DateTimeUtils.generate_random_timestamp()
    print(f"Random timestamp: {timestamp}")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    monitor.start_timer()
    import time
    time.sleep(0.1)
    elapsed = monitor.end_timer()
    print(f"Performance test: {elapsed:.3f} seconds")
    
    print("Utility functions test completed!")