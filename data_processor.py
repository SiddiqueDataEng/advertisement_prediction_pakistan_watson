# -*- coding: utf-8 -*-
"""
Data Processing and Feature Engineering Module
Compatible with Python 3.6+ (2018 standards)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AdvertisementDataProcessor:
    """Process and prepare advertisement data for machine learning"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Clicked on Ad'
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load advertisement data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def explore_data(self, df: pd.DataFrame) -> Dict:
        """Perform exploratory data analysis"""
        if df.empty:
            return {}
        
        exploration_results = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_stats': df.describe().to_dict(),
            'categorical_stats': {}
        }
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            exploration_results['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head().to_dict()
            }
        
        # Click rate analysis
        if self.target_column in df.columns:
            exploration_results['click_rate'] = df[self.target_column].mean()
            exploration_results['click_distribution'] = df[self.target_column].value_counts().to_dict()
        
        return exploration_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle outliers in numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != self.target_column:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        
        print(f"Data cleaning completed: {len(df_clean)} rows remaining")
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        if df.empty:
            return df
        
        df_features = df.copy()
        
        # Time-based features from timestamp
        if 'Timestamp' in df_features.columns:
            df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'])
            df_features['Hour'] = df_features['Timestamp'].dt.hour
            df_features['DayOfWeek'] = df_features['Timestamp'].dt.dayofweek
            df_features['Month'] = df_features['Timestamp'].dt.month
            df_features['IsWeekend'] = (df_features['DayOfWeek'] >= 5).astype(int)
            
            # Peak hours (9-11 AM, 7-9 PM)
            df_features['IsPeakHour'] = ((df_features['Hour'].between(9, 11)) | 
                                       (df_features['Hour'].between(19, 21))).astype(int)
        
        # Income categories
        if 'Area Income' in df_features.columns:
            df_features['IncomeCategory'] = pd.cut(
                df_features['Area Income'], 
                bins=[0, 30000, 60000, 90000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # Age groups
        if 'Age' in df_features.columns:
            df_features['AgeGroup'] = pd.cut(
                df_features['Age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '55+']
            )
        
        # Internet usage intensity
        if 'Daily Internet Usage' in df_features.columns:
            df_features['InternetUsageIntensity'] = pd.cut(
                df_features['Daily Internet Usage'],
                bins=[0, 100, 200, 300, float('inf')],
                labels=['Light', 'Moderate', 'Heavy', 'Very Heavy']
            )
        
        # Engagement score (combination of time on site and internet usage)
        if all(col in df_features.columns for col in ['Daily Time Spent on Site', 'Daily Internet Usage']):
            df_features['EngagementScore'] = (
                df_features['Daily Time Spent on Site'] * 0.6 + 
                df_features['Daily Internet Usage'] * 0.4
            ) / 100
        
        print(f"Feature engineering completed: {len(df_features.columns)} features created")
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features for machine learning"""
        if df.empty:
            return df
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        # Exclude timestamp if it exists
        categorical_cols = [col for col in categorical_cols if 'Timestamp' not in col]
        
        for col in categorical_cols:
            if col not in self.label_encoders and fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            elif col in self.label_encoders:
                # Handle unseen categories
                unique_values = set(df_encoded[col].astype(str).unique())
                known_values = set(self.label_encoders[col].classes_)
                
                if not unique_values.issubset(known_values):
                    # Add new categories to the encoder
                    new_categories = list(unique_values - known_values)
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, new_categories
                    )
                
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        print(f"Categorical encoding completed for {len(categorical_cols)} columns")
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for machine learning"""
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        if target_col is None:
            target_col = self.target_column
        
        # Separate features and target
        if target_col in df.columns:
            X = df.drop([target_col, 'Timestamp'], axis=1, errors='ignore')
            y = df[target_col]
        else:
            X = df.drop(['Timestamp'], axis=1, errors='ignore')
            y = pd.Series()
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        print(f"Features prepared: {X.shape[1]} features, {len(y)} samples")
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets"""
        if X.empty or y.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def process_pipeline(self, filepath: str) -> Dict:
        """Complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Load data
        df = self.load_data(filepath)
        if df.empty:
            return {'error': 'Failed to load data'}
        
        # Explore data
        exploration = self.explore_data(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_features = self.feature_engineering(df_clean)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Prepare features
        X, y = self.prepare_features(df_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        results = {
            'exploration': exploration,
            'processed_data': df_encoded,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders
        }
        
        print("Data processing pipeline completed successfully!")
        return results

if __name__ == "__main__":
    processor = AdvertisementDataProcessor()
    results = processor.process_pipeline('data/advertising_pakistan.csv')
    
    if 'error' not in results:
        print("\nProcessing Summary:")
        print(f"Features: {len(results['feature_columns'])}")
        print(f"Training samples: {len(results['X_train'])}")
        print(f"Test samples: {len(results['X_test'])}")
        print(f"Click rate: {results['exploration']['click_rate']:.2%}")