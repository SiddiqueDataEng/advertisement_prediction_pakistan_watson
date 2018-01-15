# -*- coding: utf-8 -*-
"""
IBM Watson Machine Learning Integration Module
Compatible with Python 3.6+ (2018 standards)
"""

import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from config import Config
import warnings
warnings.filterwarnings('ignore')

class WatsonMLIntegration:
    """Handle IBM Watson Machine Learning API integration"""
    
    def __init__(self, api_key: str = None):
        self.config = Config()
        self.api_key = api_key or self.config.IBM_API_KEY
        self.access_token = None
        self.deployment_url = None
        self.headers = None
        
        if self.api_key and self.api_key != '<Enter your API Key>':
            self.authenticate()
    
    def authenticate(self) -> bool:
        """Authenticate with IBM Watson ML service"""
        try:
            token_url = 'https://iam.cloud.ibm.com/identity/token'
            token_data = {
                'apikey': self.api_key,
                'grant_type': 'urn:ibm:params:oauth:grant-type:apikey'
            }
            
            print("Authenticating with IBM Watson ML...")
            response = requests.post(token_url, data=token_data)
            
            if response.status_code == 200:
                self.access_token = response.json()['access_token']
                self.headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.access_token}'
                }
                
                # Construct deployment URL
                self.deployment_url = (
                    f"{self.config.IBM_ML_ENDPOINT}/"
                    f"{self.config.IBM_DEPLOYMENT_ID}/predictions"
                    f"?version={self.config.IBM_VERSION}"
                )
                
                print("Authentication successful!")
                return True
            else:
                print(f"Authentication failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False
    
    def validate_input_data(self, input_data: Dict) -> Tuple[bool, str]:
        """Validate input data format and values"""
        required_fields = [
            'daily_time_spent_on_site', 'age', 'area_income',
            'daily_internet_usage', 'ad_topic_line', 'city',
            'gender', 'country', 'timestamp'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in input_data:
                return False, f"Missing required field: {field}"
        
        # Validate data types and ranges
        try:
            # Numerical validations
            if not (0 <= float(input_data['daily_time_spent_on_site']) <= 300):
                return False, "Daily time spent on site must be between 0 and 300 minutes"
            
            if not (18 <= int(input_data['age']) <= 100):
                return False, "Age must be between 18 and 100 years"
            
            if not (10000 <= float(input_data['area_income']) <= 500000):
                return False, "Area income must be between 10,000 and 500,000"
            
            if not (0 <= float(input_data['daily_internet_usage']) <= 500):
                return False, "Daily internet usage must be between 0 and 500 minutes"
            
            # String validations
            if not input_data['ad_topic_line'].strip():
                return False, "Ad topic line cannot be empty"
            
            if not input_data['city'].strip():
                return False, "City cannot be empty"
            
            if input_data['gender'].lower() not in ['male', 'female', '0', '1']:
                return False, "Gender must be 'Male', 'Female', '0', or '1'"
            
            if not input_data['country'].strip():
                return False, "Country cannot be empty"
            
            return True, "Validation successful"
            
        except (ValueError, TypeError) as e:
            return False, f"Data type validation error: {str(e)}"
    
    def prepare_watson_payload(self, input_data: Dict) -> Dict:
        """Prepare data payload for Watson ML API"""
        # Convert gender to numeric if needed
        gender_value = input_data['gender']
        if isinstance(gender_value, str):
            gender_value = 1 if gender_value.lower() == 'male' else 0
        
        # Prepare feature values in the correct order
        feature_values = [
            float(input_data['daily_time_spent_on_site']),
            int(input_data['age']),
            float(input_data['area_income']),
            float(input_data['daily_internet_usage']),
            str(input_data['ad_topic_line']),
            str(input_data['city']),
            int(gender_value),
            str(input_data['country']),
            str(input_data['timestamp'])
        ]
        
        payload = {
            "input_data": [{
                "fields": [
                    "daily_time_spent_on_site", "age", "area_income",
                    "daily_internet_usage", "ad_topic_line", "city",
                    "gender", "country", "timestamp"
                ],
                "values": [feature_values]
            }]
        }
        
        return payload
    
    def make_prediction(self, input_data: Dict) -> Dict:
        """Make prediction using Watson ML deployment"""
        if not self.access_token:
            return {
                'success': False,
                'error': 'Not authenticated with Watson ML. Please check your API key.'
            }
        
        # Validate input data
        is_valid, validation_message = self.validate_input_data(input_data)
        if not is_valid:
            return {
                'success': False,
                'error': f'Input validation failed: {validation_message}'
            }
        
        try:
            # Prepare payload
            payload = self.prepare_watson_payload(input_data)
            
            print("Making prediction request to Watson ML...")
            print(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Make API request
            response = requests.post(
                self.deployment_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Raw response: {json.dumps(result, indent=2)}")
                
                # Parse prediction result
                predictions = result.get('predictions', [])
                if predictions and len(predictions) > 0:
                    prediction_values = predictions[0].get('values', [])
                    if prediction_values and len(prediction_values) > 0:
                        # Extract prediction and probability
                        prediction_result = prediction_values[0]
                        
                        if len(prediction_result) >= 2:
                            predicted_class = prediction_result[0]
                            probabilities = prediction_result[1]
                            
                            # Get probability for positive class (clicked = 1)
                            click_probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                            
                            return {
                                'success': True,
                                'prediction': int(predicted_class),
                                'probability': float(click_probability),
                                'confidence': float(click_probability) * 100,
                                'will_click': predicted_class == 1,
                                'raw_response': result
                            }
                
                return {
                    'success': False,
                    'error': 'Invalid response format from Watson ML',
                    'raw_response': result
                }
            
            else:
                error_message = f"API request failed with status {response.status_code}"
                try:
                    error_details = response.json()
                    error_message += f": {error_details}"
                except:
                    error_message += f": {response.text}"
                
                return {
                    'success': False,
                    'error': error_message
                }
        
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout. Please try again.'
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': 'Connection error. Please check your internet connection.'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }
    
    def batch_predict(self, input_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Make batch predictions for multiple records"""
        if input_dataframe.empty:
            return pd.DataFrame()
        
        results = []
        
        for idx, row in input_dataframe.iterrows():
            input_data = {
                'daily_time_spent_on_site': row.get('Daily Time Spent on Site', 0),
                'age': row.get('Age', 25),
                'area_income': row.get('Area Income', 50000),
                'daily_internet_usage': row.get('Daily Internet Usage', 150),
                'ad_topic_line': row.get('Ad Topic Line', 'Default Ad'),
                'city': row.get('City', 'Karachi'),
                'gender': row.get('Male', 0),
                'country': row.get('Country', 'Pakistan'),
                'timestamp': row.get('Timestamp', '2018-01-01 12:00:00')
            }
            
            prediction_result = self.make_prediction(input_data)
            
            result_row = {
                'index': idx,
                'prediction': prediction_result.get('prediction', 0),
                'probability': prediction_result.get('probability', 0.5),
                'success': prediction_result.get('success', False),
                'error': prediction_result.get('error', '')
            }
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def test_connection(self) -> Dict:
        """Test the Watson ML connection with sample data"""
        sample_data = {
            'daily_time_spent_on_site': 68.95,
            'age': 35,
            'area_income': 61833.9,
            'daily_internet_usage': 256.09,
            'ad_topic_line': 'Premium Shan Masala - Limited Time Offer',
            'city': 'Karachi',
            'gender': 'Female',
            'country': 'Pakistan',
            'timestamp': '2018-03-27 00:53:11'
        }
        
        print("Testing Watson ML connection with sample data...")
        result = self.make_prediction(sample_data)
        
        if result['success']:
            print("Connection test successful!")
            print(f"Sample prediction: {'Will click' if result['will_click'] else 'Will not click'}")
            print(f"Confidence: {result['confidence']:.1f}%")
        else:
            print(f"Connection test failed: {result['error']}")
        
        return result
    
    def get_deployment_info(self) -> Dict:
        """Get information about the Watson ML deployment"""
        if not self.access_token:
            return {'error': 'Not authenticated'}
        
        try:
            info_url = f"{self.config.IBM_ML_ENDPOINT}/{self.config.IBM_DEPLOYMENT_ID}"
            response = requests.get(info_url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Failed to get deployment info: {response.status_code}'}
        
        except Exception as e:
            return {'error': f'Error getting deployment info: {str(e)}'}

class MockWatsonML:
    """Mock Watson ML service for testing when API key is not available"""
    
    def __init__(self):
        self.model_loaded = True
        print("Using Mock Watson ML service (API key not configured)")
    
    def make_prediction(self, input_data: Dict) -> Dict:
        """Make mock prediction based on simple rules"""
        try:
            # Simple rule-based prediction for demonstration
            age = int(input_data.get('age', 30))
            income = float(input_data.get('area_income', 50000))
            internet_usage = float(input_data.get('daily_internet_usage', 150))
            time_on_site = float(input_data.get('daily_time_spent_on_site', 60))
            
            # Calculate mock probability
            probability = 0.3  # Base probability
            
            # Age factor
            if 25 <= age <= 35:
                probability += 0.2
            
            # Income factor
            if 40000 <= income <= 80000:
                probability += 0.15
            
            # Internet usage factor
            if internet_usage > 200:
                probability += 0.1
            
            # Time on site factor
            if time_on_site > 60:
                probability += 0.15
            
            probability = max(0, min(1, probability))
            prediction = 1 if probability > 0.5 else 0
            
            return {
                'success': True,
                'prediction': prediction,
                'probability': probability,
                'confidence': probability * 100,
                'will_click': prediction == 1,
                'mock': True
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Mock prediction error: {str(e)}'
            }
    
    def test_connection(self) -> Dict:
        """Test mock connection"""
        sample_data = {
            'daily_time_spent_on_site': 68.95,
            'age': 35,
            'area_income': 61833.9,
            'daily_internet_usage': 256.09,
            'ad_topic_line': 'Premium Shan Masala - Limited Time Offer',
            'city': 'Karachi',
            'gender': 'Female',
            'country': 'Pakistan',
            'timestamp': '2018-03-27 00:53:11'
        }
        
        return self.make_prediction(sample_data)

def get_watson_client(api_key: str = None) -> Any:
    """Factory function to get Watson ML client (real or mock)"""
    config = Config()
    
    if api_key or (config.IBM_API_KEY and config.IBM_API_KEY != '<Enter your API Key>'):
        return WatsonMLIntegration(api_key)
    else:
        return MockWatsonML()

if __name__ == "__main__":
    # Test Watson ML integration
    client = get_watson_client()
    result = client.test_connection()
    
    print(f"Test result: {json.dumps(result, indent=2)}")