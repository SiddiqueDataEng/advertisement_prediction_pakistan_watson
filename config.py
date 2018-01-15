# -*- coding: utf-8 -*-
"""
Configuration file for Advertisement Prediction Application
Compatible with Python 3.6+ (2018 standards)
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration class"""
    
    # IBM Watson ML Configuration
    IBM_API_KEY = os.getenv('IBM_API_KEY', '<Enter your API Key>')
    IBM_ML_ENDPOINT = 'https://eu-de.ml.cloud.ibm.com/ml/v4/deployments'
    IBM_DEPLOYMENT_ID = os.getenv('IBM_DEPLOYMENT_ID', 'c92f5ce2-3533-4dfc-a21f-d9d9bbe9d95a')
    IBM_VERSION = '2021-10-02'
    
    # Application Settings
    APP_TITLE = 'Advertisement Success Prediction - Pakistan Market'
    APP_DESCRIPTION = 'Predict advertisement click-through rates for Pakistani market using IBM Watson ML'
    
    # Data Configuration
    DATA_FILE = 'data/advertising_pakistan.csv'
    MODEL_FEATURES = [
        'daily_time_spent_on_site',
        'age', 
        'area_income',
        'daily_internet_usage',
        'ad_topic_line',
        'city',
        'gender',
        'country',
        'timestamp'
    ]
    
    # Pakistani Cities for data generation
    PAKISTANI_CITIES = [
        'Karachi', 'Lahore', 'Faisalabad', 'Rawalpindi', 'Gujranwala',
        'Peshawar', 'Multan', 'Hyderabad', 'Islamabad', 'Quetta',
        'Bahawalpur', 'Sargodha', 'Sialkot', 'Sukkur', 'Larkana',
        'Sheikhupura', 'Jhang', 'Rahim Yar Khan', 'Gujrat', 'Kasur'
    ]
    
    # Income ranges by city (PKR)
    CITY_INCOME_RANGES = {
        'Karachi': (45000, 120000),
        'Lahore': (40000, 110000),
        'Islamabad': (50000, 150000),
        'Rawalpindi': (35000, 95000),
        'Faisalabad': (30000, 85000),
        'Peshawar': (25000, 75000),
        'Multan': (28000, 80000),
        'Hyderabad': (25000, 70000),
        'Gujranwala': (30000, 85000),
        'Quetta': (22000, 65000)
    }

class ModelConfig:
    """Model-specific configuration"""
    
    PREDICTION_THRESHOLD = 0.5
    FEATURE_COLUMNS = [
        'daily_time_spent_on_site', 'age', 'area_income', 
        'daily_internet_usage', 'ad_topic_line', 'city', 
        'gender', 'country', 'timestamp'
    ]
    TARGET_COLUMN = 'clicked_on_ad'

class UIConfig:
    """UI-specific configuration"""
    
    SIDEBAR_WIDTH = 300
    MAIN_PANEL_WIDTH = 700
    CHART_HEIGHT = 400
    
    # Streamlit theme colors
    PRIMARY_COLOR = '#1f77b4'
    BACKGROUND_COLOR = '#ffffff'
    SECONDARY_BACKGROUND_COLOR = '#f0f2f6'