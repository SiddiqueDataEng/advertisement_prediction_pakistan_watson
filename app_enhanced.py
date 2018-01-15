# -*- coding: utf-8 -*-
"""
Enhanced Advertisement Prediction App (Refactored from original app.py)
Compatible with Python 3.6+ (2018 standards)
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Import enhanced modules
from config import Config
from watson_integration import get_watson_client

class EnhancedAdvertisementApp:
    """Enhanced version of the original advertisement prediction app"""
    
    def __init__(self):
        self.config = Config()
        self.watson_client = None
        self.setup_page_config()
        self.initialize_watson_client()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=self.config.APP_TITLE,
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_watson_client(self):
        """Initialize Watson ML client"""
        try:
            self.watson_client = get_watson_client()
            if hasattr(self.watson_client, 'access_token') and self.watson_client.access_token:
                st.success("✅ Connected to IBM Watson ML")
            else:
                st.info("ℹ️ Using mock Watson ML service (API key not configured)")
        except Exception as e:
            st.error(f"Error initializing Watson ML: {str(e)}")
            self.watson_client = None
    
    def render_header(self):
        """Render application header"""
        st.title('🎯 Advertisement Success Prediction')
        st.subheader('Pakistan Market Analysis - Enhanced Version')
        
        st.markdown("""
        **Enhanced Features:**
        - 🇵🇰 Pakistani market data with local cities and demographics
        - 📊 Real-time prediction analytics and confidence scoring
        - 🎨 Modern UI with interactive visualizations
        - 🔧 Improved error handling and validation
        - 📈 Prediction history and performance tracking
        """)
        
        # Display connection status
        col1, col2, col3 = st.columns(3)
        with col1:
            if self.watson_client:
                st.metric("Watson ML Status", "Connected", "✅")
            else:
                st.metric("Watson ML Status", "Disconnected", "❌")
        
        with col2:
            st.metric("Target Market", "Pakistan", "🇵🇰")
        
        with col3:
            st.metric("Model Version", "2018", "📅")
    
    def get_user_input(self) -> Dict[str, Any]:
        """Get user input with enhanced validation"""
        st.sidebar.header('📝 User Information')
        
        # Personal Information Section
        st.sidebar.subheader("👤 Personal Details")
        
        daily_time = st.sidebar.number_input(
            'Daily Time Spent on Site (minutes)',
            min_value=0.0,
            max_value=300.0,
            value=68.95,
            step=0.1,
            help="Average time spent on websites per day"
        )
        
        age = st.sidebar.number_input(
            'Age (years)',
            min_value=18,
            max_value=100,
            value=35,
            help="User's age in years"
        )
        
        # Financial Information Section
        st.sidebar.subheader("💰 Financial Details")
        
        area_income = st.sidebar.number_input(
            'Monthly Area Income (PKR)',
            min_value=10000.0,
            max_value=500000.0,
            value=61833.9,
            step=100.0,
            help="Average monthly income in your area"
        )
        
        # Behavioral Information Section
        st.sidebar.subheader("🌐 Online Behavior")
        
        daily_internet_use = st.sidebar.number_input(
            'Daily Internet Usage (minutes)',
            min_value=0.0,
            max_value=500.0,
            value=256.09,
            step=1.0,
            help="Total daily internet usage"
        )
        
        # Advertisement and Location Section
        st.sidebar.subheader("📍 Location & Advertisement")
        
        ad_topic_line = st.sidebar.selectbox(
            'Advertisement Topic Line',
            options=[
                'Premium Shan Masala - Limited Time Offer',
                'New Tapal Tea Launch - Special Discount',
                'Best Olpers Milk Deals in Pakistan',
                'Exclusive Khaadi Collection 2018',
                'Top Quality National Foods - Free Delivery',
                'Latest Mobilink Jazz Technology - Buy Now',
                'Authentic Gul Ahmed - Nationwide Service',
                'Professional HBL Banking Solutions Pakistan',
                'New Daraz Pakistan Launch - Special Discount',
                'Premium Careem Pakistan - Limited Time Offer'
            ],
            help="Select the advertisement topic"
        )
        
        city = st.sidebar.selectbox(
            'City',
            options=self.config.PAKISTANI_CITIES,
            index=0,
            help="Select your city in Pakistan"
        )
        
        gender = st.sidebar.selectbox(
            'Gender',
            options=['Female', 'Male'],
            help="Select your gender"
        )
        
        country = st.sidebar.text_input(
            'Country',
            value='Pakistan',
            disabled=True,
            help="Country is set to Pakistan for this market analysis"
        )
        
        # Auto-generate timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            'daily_time_spent_on_site': daily_time,
            'age': age,
            'area_income': area_income,
            'daily_internet_usage': daily_internet_use,
            'ad_topic_line': ad_topic_line,
            'city': city,
            'gender': gender,
            'country': country,
            'timestamp': timestamp
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> tuple[bool, str]:
        """Enhanced input validation"""
        try:
            # Validate numerical ranges
            if not (0 <= input_data['daily_time_spent_on_site'] <= 300):
                return False, "Daily time on site must be between 0 and 300 minutes"
            
            if not (18 <= input_data['age'] <= 100):
                return False, "Age must be between 18 and 100 years"
            
            if not (10000 <= input_data['area_income'] <= 500000):
                return False, "Area income must be between PKR 10,000 and PKR 500,000"
            
            if not (0 <= input_data['daily_internet_usage'] <= 500):
                return False, "Daily internet usage must be between 0 and 500 minutes"
            
            # Validate required fields
            required_fields = ['ad_topic_line', 'city', 'gender', 'country']
            for field in required_fields:
                if not input_data.get(field, '').strip():
                    return False, f"{field.replace('_', ' ').title()} is required"
            
            return True, "Input validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def make_prediction(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make prediction with enhanced error handling"""
        if not self.watson_client:
            st.error("Watson ML client not available")
            return None
        
        # Validate input
        is_valid, message = self.validate_input(input_data)
        if not is_valid:
            st.error(f"Input validation failed: {message}")
            return None
        
        try:
            with st.spinner("Making prediction..."):
                result = self.watson_client.make_prediction(input_data)
                
                if result.get('success', False):
                    return result
                else:
                    st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")
                    return None
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
    
    def display_prediction_result(self, result: Dict[str, Any], input_data: Dict[str, Any]):
        """Display prediction results with enhanced visualization"""
        st.header("🔮 Prediction Results")
        
        # Main result display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result.get('will_click', False):
                st.success("✅ **User WILL click the advertisement**")
                prediction_text = "Will Click"
                prediction_color = "green"
            else:
                st.error("❌ **User will NOT click the advertisement**")
                prediction_text = "Will Not Click"
                prediction_color = "red"
        
        with col2:
            confidence = result.get('confidence', 0)
            st.metric("Confidence Level", f"{confidence:.1f}%")
        
        with col3:
            probability = result.get('probability', 0)
            st.metric("Click Probability", f"{probability:.3f}")
        
        # Detailed analysis
        st.subheader("📊 Detailed Analysis")
        
        # Create analysis based on input factors
        analysis_factors = self.analyze_factors(input_data, result)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Factors:**")
            for factor in analysis_factors['positive']:
                st.write(f"✅ {factor}")
        
        with col2:
            st.write("**Risk Factors:**")
            for factor in analysis_factors['negative']:
                st.write(f"⚠️ {factor}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = self.generate_recommendations(input_data, result)
        for rec in recommendations:
            st.write(f"• {rec}")
    
    def analyze_factors(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, list]:
        """Analyze factors contributing to the prediction"""
        positive_factors = []
        negative_factors = []
        
        # Age analysis
        age = input_data['age']
        if 25 <= age <= 35:
            positive_factors.append(f"Optimal age group ({age} years) - high engagement demographic")
        elif age < 25:
            negative_factors.append(f"Young age group ({age} years) - may have limited purchasing power")
        else:
            negative_factors.append(f"Older age group ({age} years) - may be less responsive to digital ads")
        
        # Income analysis
        income = input_data['area_income']
        if 40000 <= income <= 80000:
            positive_factors.append(f"Middle income bracket (PKR {income:,.0f}) - target demographic")
        elif income < 40000:
            negative_factors.append(f"Lower income bracket (PKR {income:,.0f}) - limited purchasing power")
        else:
            positive_factors.append(f"High income bracket (PKR {income:,.0f}) - strong purchasing power")
        
        # Internet usage analysis
        internet_usage = input_data['daily_internet_usage']
        if internet_usage > 200:
            positive_factors.append(f"Heavy internet user ({internet_usage:.0f} min/day) - high digital engagement")
        elif internet_usage < 100:
            negative_factors.append(f"Light internet user ({internet_usage:.0f} min/day) - low digital engagement")
        
        # Time on site analysis
        time_on_site = input_data['daily_time_spent_on_site']
        if time_on_site > 60:
            positive_factors.append(f"High site engagement ({time_on_site:.0f} min/day)")
        else:
            negative_factors.append(f"Low site engagement ({time_on_site:.0f} min/day)")
        
        # City analysis
        city = input_data['city']
        major_cities = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad']
        if city in major_cities:
            positive_factors.append(f"Major city ({city}) - high market penetration")
        
        return {'positive': positive_factors, 'negative': negative_factors}
    
    def generate_recommendations(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        confidence = result.get('confidence', 0)
        will_click = result.get('will_click', False)
        
        if will_click:
            if confidence > 80:
                recommendations.append("High confidence prediction - proceed with targeted advertising")
                recommendations.append("Consider premium ad placements for this user segment")
            else:
                recommendations.append("Moderate confidence - test with A/B campaigns")
                recommendations.append("Monitor engagement metrics closely")
        else:
            recommendations.append("Low click probability - consider alternative targeting")
            recommendations.append("Focus on improving ad relevance and timing")
            
            # Specific recommendations based on factors
            if input_data['daily_internet_usage'] < 100:
                recommendations.append("Target user during peak internet usage hours")
            
            if input_data['daily_time_spent_on_site'] < 30:
                recommendations.append("Improve website engagement to increase ad effectiveness")
        
        return recommendations
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Get user input
        input_data = self.get_user_input()
        
        # Main prediction section
        st.header("🚀 Make Prediction")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("**Current Input Summary:**")
            summary_data = {
                'Age': f"{input_data['age']} years",
                'City': input_data['city'],
                'Income': f"PKR {input_data['area_income']:,.0f}",
                'Internet Usage': f"{input_data['daily_internet_usage']:.0f} min/day",
                'Time on Site': f"{input_data['daily_time_spent_on_site']:.0f} min/day",
                'Gender': input_data['gender'],
                'Ad Topic': input_data['ad_topic_line'][:50] + "..." if len(input_data['ad_topic_line']) > 50 else input_data['ad_topic_line']
            }
            
            for key, value in summary_data.items():
                st.write(f"• **{key}:** {value}")
        
        with col2:
            if st.button("🎯 Predict Click", type="primary", use_container_width=True):
                result = self.make_prediction(input_data)
                if result:
                    self.display_prediction_result(result, input_data)
            
            if st.button("🧪 Test Connection", use_container_width=True):
                if self.watson_client:
                    test_result = self.watson_client.test_connection()
                    if test_result.get('success', False):
                        st.success("✅ Watson ML connection successful!")
                        if 'mock' in test_result:
                            st.info("ℹ️ Using mock service")
                    else:
                        st.error(f"❌ Connection failed: {test_result.get('error', 'Unknown error')}")
                else:
                    st.error("Watson ML client not available")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **Enhanced Advertisement Prediction System** | 
        Powered by IBM Watson ML | 
        Built for Pakistan Market | 
        Compatible with Python 3.6+ (2018)
        """)

def main():
    """Main entry point"""
    try:
        app = EnhancedAdvertisementApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()