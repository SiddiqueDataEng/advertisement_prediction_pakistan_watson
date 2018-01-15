# -*- coding: utf-8 -*-
"""
Enhanced Streamlit Application for Advertisement Prediction
Compatible with Python 3.6+ (2018 standards)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

# Import custom modules
from config import Config, UIConfig
from watson_integration import get_watson_client
from data_generator import PakistaniAdDataGenerator
from data_processor import AdvertisementDataProcessor

# Page configuration
st.set_page_config(
    page_title="Advertisement Prediction - Pakistan Market",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvertisementPredictionApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = Config()
        self.ui_config = UIConfig()
        self.watson_client = None
        self.initialize_session_state()
        self.load_watson_client()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        if 'data_generated' not in st.session_state:
            st.session_state.data_generated = False
        
        if 'sample_data' not in st.session_state:
            st.session_state.sample_data = None
    
    def load_watson_client(self):
        """Load Watson ML client"""
        try:
            self.watson_client = get_watson_client()
        except Exception as e:
            st.error(f"Error loading Watson ML client: {str(e)}")
            self.watson_client = None
    
    def render_header(self):
        """Render application header"""
        st.title("🎯 Advertisement Success Prediction")
        st.subheader("Pakistan Market Analysis - Powered by IBM Watson ML")
        
        st.markdown("""
        This application predicts whether a user will click on an advertisement based on their 
        demographic and behavioral characteristics. The model is specifically trained on Pakistani 
        market data and uses IBM Watson Machine Learning for predictions.
        """)
        
        # Add metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Target Market", "Pakistan", "🇵🇰")
        
        with col2:
            st.metric("Model Type", "Classification", "🤖")
        
        with col3:
            st.metric("Predictions Made", len(st.session_state.prediction_history), "📈")
        
        with col4:
            if st.session_state.prediction_history:
                avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history])
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%", "🎯")
            else:
                st.metric("Avg Confidence", "N/A", "🎯")
    
    def render_sidebar(self):
        """Render sidebar with input controls"""
        st.sidebar.header("📝 User Information")
        
        # Personal Information
        st.sidebar.subheader("Personal Details")
        
        age = st.sidebar.slider(
            "Age", 
            min_value=18, 
            max_value=65, 
            value=30,
            help="User's age in years"
        )
        
        gender = st.sidebar.selectbox(
            "Gender",
            options=["Female", "Male"],
            help="User's gender"
        )
        
        city = st.sidebar.selectbox(
            "City",
            options=self.config.PAKISTANI_CITIES,
            index=0,
            help="User's city in Pakistan"
        )
        
        # Financial Information
        st.sidebar.subheader("Financial Details")
        
        # Set income range based on city
        if city in self.config.CITY_INCOME_RANGES:
            min_income, max_income = self.config.CITY_INCOME_RANGES[city]
        else:
            min_income, max_income = 25000, 75000
        
        area_income = st.sidebar.slider(
            "Monthly Income (PKR)",
            min_value=min_income,
            max_value=max_income,
            value=int((min_income + max_income) / 2),
            step=1000,
            help=f"Average monthly income in {city}"
        )
        
        # Behavioral Information
        st.sidebar.subheader("Online Behavior")
        
        daily_time_on_site = st.sidebar.slider(
            "Daily Time on Site (minutes)",
            min_value=10.0,
            max_value=120.0,
            value=65.0,
            step=0.5,
            help="Average time spent on websites daily"
        )
        
        daily_internet_usage = st.sidebar.slider(
            "Daily Internet Usage (minutes)",
            min_value=30.0,
            max_value=300.0,
            value=180.0,
            step=5.0,
            help="Total daily internet usage"
        )
        
        # Advertisement Information
        st.sidebar.subheader("Advertisement Details")
        
        ad_topic_options = [
            "Premium Shan Masala - Limited Time Offer",
            "New Tapal Tea Launch - Special Discount",
            "Best Olpers Milk Deals in Pakistan",
            "Exclusive Khaadi Collection 2018",
            "Top Quality National Foods - Free Delivery",
            "Latest Mobilink Jazz Technology - Buy Now",
            "Authentic Gul Ahmed - Nationwide Service",
            "Professional HBL Banking Solutions Pakistan"
        ]
        
        ad_topic_line = st.sidebar.selectbox(
            "Advertisement Topic",
            options=ad_topic_options,
            help="Select advertisement topic or campaign"
        )
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            'daily_time_spent_on_site': daily_time_on_site,
            'age': age,
            'area_income': area_income,
            'daily_internet_usage': daily_internet_usage,
            'ad_topic_line': ad_topic_line,
            'city': city,
            'gender': gender,
            'country': 'Pakistan',
            'timestamp': timestamp
        }
    
    def render_prediction_section(self, input_data: Dict):
        """Render prediction section"""
        st.header("🔮 Prediction Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🚀 Make Prediction", type="primary", use_container_width=True):
                self.make_prediction(input_data)
        
        with col1:
            if st.button("🧪 Test Watson Connection", use_container_width=True):
                self.test_watson_connection()
    
    def make_prediction(self, input_data: Dict):
        """Make prediction using Watson ML"""
        if not self.watson_client:
            st.error("Watson ML client not available. Please check your configuration.")
            return
        
        with st.spinner("Making prediction..."):
            try:
                result = self.watson_client.make_prediction(input_data)
                
                if result['success']:
                    # Store prediction in history
                    prediction_record = {
                        'timestamp': datetime.now(),
                        'input_data': input_data.copy(),
                        'prediction': result['prediction'],
                        'probability': result['probability'],
                        'confidence': result['confidence'],
                        'will_click': result['will_click']
                    }
                    st.session_state.prediction_history.append(prediction_record)
                    
                    # Display results
                    self.display_prediction_results(result, input_data)
                    
                else:
                    st.error(f"Prediction failed: {result['error']}")
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    def display_prediction_results(self, result: Dict, input_data: Dict):
        """Display prediction results with visualizations"""
        # Main result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result['will_click']:
                st.success("✅ **User WILL click the ad**")
            else:
                st.error("❌ **User will NOT click the ad**")
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        
        with col3:
            st.metric("Probability", f"{result['probability']:.3f}")
        
        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['confidence'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Prediction Confidence"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Factors analysis
        st.subheader("📊 Prediction Factors Analysis")
        
        factors = self.analyze_prediction_factors(input_data, result['probability'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Positive Factors:**")
            for factor in factors['positive']:
                st.write(f"✅ {factor}")
        
        with col2:
            st.write("**Negative Factors:**")
            for factor in factors['negative']:
                st.write(f"❌ {factor}")
    
    def analyze_prediction_factors(self, input_data: Dict, probability: float) -> Dict:
        """Analyze factors contributing to the prediction"""
        positive_factors = []
        negative_factors = []
        
        # Age analysis
        age = input_data['age']
        if 25 <= age <= 35:
            positive_factors.append(f"Optimal age group ({age} years)")
        elif age < 25:
            negative_factors.append(f"Young age group ({age} years)")
        else:
            negative_factors.append(f"Older age group ({age} years)")
        
        # Income analysis
        income = input_data['area_income']
        if 40000 <= income <= 80000:
            positive_factors.append(f"Middle income bracket (PKR {income:,})")
        elif income < 40000:
            negative_factors.append(f"Lower income bracket (PKR {income:,})")
        else:
            negative_factors.append(f"High income bracket (PKR {income:,})")
        
        # Internet usage analysis
        internet_usage = input_data['daily_internet_usage']
        if internet_usage > 200:
            positive_factors.append(f"High internet usage ({internet_usage:.0f} min/day)")
        elif internet_usage < 100:
            negative_factors.append(f"Low internet usage ({internet_usage:.0f} min/day)")
        else:
            positive_factors.append(f"Moderate internet usage ({internet_usage:.0f} min/day)")
        
        # Time on site analysis
        time_on_site = input_data['daily_time_spent_on_site']
        if time_on_site > 60:
            positive_factors.append(f"High engagement ({time_on_site:.0f} min/day)")
        else:
            negative_factors.append(f"Low engagement ({time_on_site:.0f} min/day)")
        
        # City analysis
        city = input_data['city']
        if city in ['Karachi', 'Lahore', 'Islamabad']:
            positive_factors.append(f"Major city ({city})")
        else:
            negative_factors.append(f"Smaller city ({city})")
        
        return {'positive': positive_factors, 'negative': negative_factors}
    
    def test_watson_connection(self):
        """Test Watson ML connection"""
        if not self.watson_client:
            st.error("Watson ML client not available.")
            return
        
        with st.spinner("Testing Watson ML connection..."):
            result = self.watson_client.test_connection()
            
            if result['success']:
                st.success("✅ Watson ML connection successful!")
                if 'mock' in result:
                    st.info("ℹ️ Using mock service (API key not configured)")
                st.json(result)
            else:
                st.error(f"❌ Connection failed: {result['error']}")
    
    def render_data_section(self):
        """Render data generation and analysis section"""
        st.header("📊 Dataset Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Generate New Dataset", use_container_width=True):
                self.generate_dataset()
        
        with col2:
            if st.button("📈 Analyze Current Data", use_container_width=True):
                self.analyze_dataset()
        
        with col3:
            if st.button("💾 Download Dataset", use_container_width=True):
                self.download_dataset()
    
    def generate_dataset(self):
        """Generate new Pakistani advertisement dataset"""
        with st.spinner("Generating Pakistani advertisement dataset..."):
            try:
                generator = PakistaniAdDataGenerator(1000)
                filename = generator.save_dataset('data/advertising_pakistan.csv')
                
                st.success(f"✅ Dataset generated successfully!")
                st.info(f"📁 Saved to: {filename}")
                
                # Load and display sample
                df = pd.read_csv(filename)
                st.session_state.sample_data = df
                st.session_state.data_generated = True
                
                st.subheader("📋 Dataset Preview")
                st.dataframe(df.head(10))
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", len(df))
                
                with col2:
                    st.metric("Click Rate", f"{df['Clicked on Ad'].mean():.1%}")
                
                with col3:
                    st.metric("Avg Age", f"{df['Age'].mean():.1f}")
                
                with col4:
                    st.metric("Cities", df['City'].nunique())
            
            except Exception as e:
                st.error(f"Error generating dataset: {str(e)}")
    
    def analyze_dataset(self):
        """Analyze the current dataset"""
        try:
            df = pd.read_csv('data/advertising_pakistan.csv')
            
            st.subheader("📊 Dataset Analysis")
            
            # Basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Click Rate by City**")
                city_stats = df.groupby('City')['Clicked on Ad'].agg(['count', 'mean']).round(3)
                city_stats.columns = ['Total Ads', 'Click Rate']
                st.dataframe(city_stats)
            
            with col2:
                st.write("**Age Group Analysis**")
                df['Age Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
                age_stats = df.groupby('Age Group')['Clicked on Ad'].agg(['count', 'mean']).round(3)
                age_stats.columns = ['Total Users', 'Click Rate']
                st.dataframe(age_stats)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_city = px.bar(
                    city_stats.reset_index(), 
                    x='City', 
                    y='Click Rate',
                    title='Click Rate by City'
                )
                st.plotly_chart(fig_city, use_container_width=True)
            
            with col2:
                fig_age = px.histogram(
                    df, 
                    x='Age', 
                    color='Clicked on Ad',
                    title='Age Distribution by Click Status'
                )
                st.plotly_chart(fig_age, use_container_width=True)
        
        except FileNotFoundError:
            st.warning("Dataset not found. Please generate a new dataset first.")
        except Exception as e:
            st.error(f"Error analyzing dataset: {str(e)}")
    
    def download_dataset(self):
        """Provide dataset download functionality"""
        try:
            df = pd.read_csv('data/advertising_pakistan.csv')
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"advertising_pakistan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        except FileNotFoundError:
            st.warning("Dataset not found. Please generate a new dataset first.")
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")
    
    def render_history_section(self):
        """Render prediction history section"""
        if not st.session_state.prediction_history:
            st.info("No predictions made yet. Make your first prediction above!")
            return
        
        st.header("📈 Prediction History")
        
        # Convert history to DataFrame
        history_data = []
        for record in st.session_state.prediction_history:
            history_data.append({
                'Timestamp': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Age': record['input_data']['age'],
                'City': record['input_data']['city'],
                'Income': record['input_data']['area_income'],
                'Prediction': 'Will Click' if record['will_click'] else 'Will Not Click',
                'Confidence': f"{record['confidence']:.1f}%"
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history)
        
        # History statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            positive_predictions = sum(1 for r in st.session_state.prediction_history if r['will_click'])
            st.metric("Positive Predictions", positive_predictions)
        
        with col2:
            avg_confidence = np.mean([r['confidence'] for r in st.session_state.prediction_history])
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.experimental_rerun()
    
    def run(self):
        """Run the main application"""
        # Render header
        self.render_header()
        
        # Get user input from sidebar
        input_data = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dataset", "📈 History"])
        
        with tab1:
            self.render_prediction_section(input_data)
        
        with tab2:
            self.render_data_section()
        
        with tab3:
            self.render_history_section()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Advertisement Prediction System** | "
            "Powered by IBM Watson ML | "
            "Built with Streamlit | "
            "© 2018 Pakistan Market Analysis"
        )

def main():
    """Main application entry point"""
    app = AdvertisementPredictionApp()
    app.run()

if __name__ == "__main__":
    main()