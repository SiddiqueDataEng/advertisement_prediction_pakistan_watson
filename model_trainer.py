# -*- coding: utf-8 -*-
"""
Machine Learning Model Training Module
Compatible with Python 3.6+ (2018 standards)
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvertisementModelTrainer:
    """Train and evaluate machine learning models for advertisement prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.evaluation_results = {}
    
    def initialize_models(self) -> Dict:
        """Initialize different machine learning models"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        self.models = models
        print(f"Initialized {len(models)} models for training")
        return models
    
    def train_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_name: str) -> Any:
        """Train a single model"""
        print(f"Training {model_name}...")
        
        try:
            model.fit(X_train, y_train)
            print(f"{model_name} training completed successfully")
            return model
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            return None
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict:
        """Evaluate a trained model"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            }
            
            # Classification report
            metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            
            print(f"{model_name} evaluation completed")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return {}
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                           cv: int = 5) -> Dict:
        """Perform cross-validation on a model"""
        try:
            cv_scores = {
                'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
                'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
                'recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
                'f1': cross_val_score(model, X, y, cv=cv, scoring='f1')
            }
            
            cv_results = {}
            for metric, scores in cv_scores.items():
                cv_results[f'{metric}_mean'] = scores.mean()
                cv_results[f'{metric}_std'] = scores.std()
            
            return cv_results
            
        except Exception as e:
            print(f"Error in cross-validation: {str(e)}")
            return {}
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame, 
                            y_train: pd.Series) -> Any:
        """Perform hyperparameter tuning for specific models"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 6, 9]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name not in param_grids:
            print(f"No hyperparameter grid defined for {model_name}")
            return self.models[model_name]
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        try:
            grid_search = GridSearchCV(
                self.models[model_name],
                param_grids[model_name],
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            return self.models[model_name]
    
    def extract_feature_importance(self, model, feature_names: List[str], 
                                 model_name: str) -> pd.DataFrame:
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                print(f"Cannot extract feature importance from {model_name}")
                return pd.DataFrame()
            
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
            
        except Exception as e:
            print(f"Error extracting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        tune_hyperparameters: bool = False) -> Dict:
        """Train and evaluate all models"""
        print("Starting model training pipeline...")
        
        # Initialize models
        self.initialize_models()
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Processing {model_name}")
            print(f"{'='*50}")
            
            # Hyperparameter tuning (optional)
            if tune_hyperparameters:
                model = self.hyperparameter_tuning(model_name, X_train, y_train)
            
            # Train model
            trained_model = self.train_model(model, X_train, y_train, model_name)
            
            if trained_model is not None:
                # Evaluate model
                evaluation = self.evaluate_model(trained_model, X_test, y_test, model_name)
                
                # Cross-validation
                cv_results = self.cross_validate_model(trained_model, X_train, y_train)
                
                # Feature importance
                feature_importance = self.extract_feature_importance(
                    trained_model, list(X_train.columns), model_name
                )
                
                results[model_name] = {
                    'model': trained_model,
                    'evaluation': evaluation,
                    'cross_validation': cv_results,
                    'feature_importance': feature_importance
                }
        
        self.evaluation_results = results
        
        # Find best model
        self.find_best_model()
        
        print(f"\nModel training completed. Best model: {self.best_model_name}")
        return results
    
    def find_best_model(self) -> str:
        """Find the best performing model based on F1-score"""
        best_f1 = 0
        best_model_name = None
        
        for model_name, results in self.evaluation_results.items():
            if 'evaluation' in results and 'f1_score' in results['evaluation']:
                f1_score = results['evaluation']['f1_score']
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.evaluation_results[best_model_name]['model']
            self.best_model_name = best_model_name
            self.feature_importance = self.evaluation_results[best_model_name]['feature_importance']
        
        return best_model_name
    
    def save_model(self, filepath: str = 'models/best_advertisement_model.pkl') -> str:
        """Save the best trained model"""
        if self.best_model is None:
            print("No trained model to save")
            return ""
        
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_importance': self.feature_importance,
                'evaluation_results': self.evaluation_results[self.best_model_name]['evaluation']
            }
            
            joblib.dump(model_data, filepath)
            print(f"Best model ({self.best_model_name}) saved to {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return ""
    
    def load_model(self, filepath: str) -> bool:
        """Load a saved model"""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_importance = model_data['feature_importance']
            
            print(f"Model ({self.best_model_name}) loaded successfully from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the best model"""
        if self.best_model is None:
            print("No trained model available for prediction")
            return np.array([]), np.array([])
        
        try:
            predictions = self.best_model.predict(X)
            probabilities = self.best_model.predict_proba(X)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
            
            return predictions, probabilities
            
        except Exception as e:
            print(f"Error making predictions: {str(e)}")
            return np.array([]), np.array([])
    
    def generate_model_report(self) -> str:
        """Generate a comprehensive model performance report"""
        if not self.evaluation_results:
            return "No model results available"
        
        report = "Advertisement Prediction Model Performance Report\n"
        report += "=" * 60 + "\n\n"
        
        # Best model summary
        if self.best_model_name:
            report += f"Best Model: {self.best_model_name}\n"
            best_results = self.evaluation_results[self.best_model_name]['evaluation']
            report += f"Accuracy: {best_results['accuracy']:.4f}\n"
            report += f"Precision: {best_results['precision']:.4f}\n"
            report += f"Recall: {best_results['recall']:.4f}\n"
            report += f"F1-Score: {best_results['f1_score']:.4f}\n"
            if best_results['roc_auc']:
                report += f"ROC-AUC: {best_results['roc_auc']:.4f}\n"
            report += "\n"
        
        # All models comparison
        report += "All Models Comparison:\n"
        report += "-" * 30 + "\n"
        
        for model_name, results in self.evaluation_results.items():
            if 'evaluation' in results:
                eval_results = results['evaluation']
                report += f"{model_name}:\n"
                report += f"  Accuracy: {eval_results['accuracy']:.4f}\n"
                report += f"  F1-Score: {eval_results['f1_score']:.4f}\n"
                if eval_results['roc_auc']:
                    report += f"  ROC-AUC: {eval_results['roc_auc']:.4f}\n"
                report += "\n"
        
        # Feature importance (top 10)
        if self.feature_importance is not None and not self.feature_importance.empty:
            report += "Top 10 Most Important Features:\n"
            report += "-" * 35 + "\n"
            top_features = self.feature_importance.head(10)
            for idx, row in top_features.iterrows():
                report += f"{row['feature']}: {row['importance']:.4f}\n"
        
        return report

if __name__ == "__main__":
    # Example usage
    from data_processor import AdvertisementDataProcessor
    
    # Process data
    processor = AdvertisementDataProcessor()
    results = processor.process_pipeline('data/advertising_pakistan.csv')
    
    if 'error' not in results:
        # Train models
        trainer = AdvertisementModelTrainer()
        model_results = trainer.train_all_models(
            results['X_train'], results['y_train'],
            results['X_test'], results['y_test'],
            tune_hyperparameters=True
        )
        
        # Save best model
        trainer.save_model()
        
        # Generate report
        report = trainer.generate_model_report()
        print("\n" + report)