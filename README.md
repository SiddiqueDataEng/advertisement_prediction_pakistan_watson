# Advertisement Prediction System - Pakistan Market

## Overview

This enhanced advertisement prediction system uses IBM Watson Machine Learning to predict whether users will click on advertisements based on their demographic and behavioral characteristics. The system is specifically designed for the Pakistani market with localized data and features.

## Features

### 🎯 Core Functionality
- **Real-time Prediction**: Predict ad click-through rates using IBM Watson ML
- **Pakistani Market Focus**: Tailored for Pakistani cities, income ranges, and demographics
- **Interactive Web Interface**: User-friendly Streamlit application
- **Comprehensive Analytics**: Detailed analysis of prediction factors

### 📊 Data Management
- **Synthetic Data Generation**: Generate realistic Pakistani advertisement data
- **Data Processing Pipeline**: Clean, validate, and prepare data for ML
- **Feature Engineering**: Create meaningful features from raw data
- **Export Capabilities**: Download datasets in CSV format

### 🤖 Machine Learning
- **Multiple Model Support**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Model Training Pipeline**: Automated training and evaluation
- **Hyperparameter Tuning**: Optimize model performance
- **Feature Importance Analysis**: Understand key prediction factors

### 🔗 IBM Watson Integration
- **Watson ML API**: Seamless integration with IBM Watson Machine Learning
- **Authentication Handling**: Secure API key management
- **Batch Predictions**: Process multiple records efficiently
- **Mock Service**: Fallback service for testing without API keys

## Installation

### Prerequisites
- Python 3.6 or higher (2018 compatible)
- IBM Cloud account (for Watson ML integration)
- Git (for cloning the repository)

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd "Advertisement Prediction -IBM watson-Code Files/Enhanced_Code"
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure IBM Watson ML** (Optional)
   - Create an IBM Cloud account
   - Set up Watson Machine Learning service
   - Get your API key and deployment ID
   - Update `config.py` with your credentials

## Usage

### Running the Application

1. **Start the Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the Application**
   - Open your browser to `http://localhost:8501`
   - Use the sidebar to input user characteristics
   - Click "Make Prediction" to get results

### Generating Data

1. **Generate Pakistani Dataset**
   ```bash
   python data_generator.py
   ```

2. **Process Data**
   ```bash
   python data_processor.py
   ```

3. **Train Models**
   ```bash
   python model_trainer.py
   ```

## Project Structure

```
Enhanced_Code/
├── config.py                 # Configuration settings
├── data_generator.py         # Pakistani data generation
├── data_processor.py         # Data processing pipeline
├── model_trainer.py          # ML model training
├── watson_integration.py     # IBM Watson ML integration
├── streamlit_app.py          # Main web application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/                     # Generated datasets
│   └── advertising_pakistan.csv
└── models/                   # Trained models
    └── best_advertisement_model.pkl
```

## Configuration

### IBM Watson ML Setup

1. **Get API Credentials**
   - Log in to IBM Cloud
   - Create Watson Machine Learning service
   - Get API key from service credentials

2. **Update Configuration**
   ```python
   # In config.py
   IBM_API_KEY = 'your-api-key-here'
   IBM_DEPLOYMENT_ID = 'your-deployment-id'
   ```

3. **Environment Variables** (Recommended)
   ```bash
   export IBM_API_KEY='your-api-key'
   export IBM_DEPLOYMENT_ID='your-deployment-id'
   ```

### Application Settings

Customize the application by modifying `config.py`:

- **Pakistani Cities**: Add/remove cities from the list
- **Income Ranges**: Adjust income ranges by city
- **Model Features**: Configure input features
- **UI Settings**: Customize interface appearance

## Data Schema

### Input Features
- `daily_time_spent_on_site`: Time spent on websites (minutes)
- `age`: User age (18-65 years)
- `area_income`: Monthly income in PKR
- `daily_internet_usage`: Daily internet usage (minutes)
- `ad_topic_line`: Advertisement headline/topic
- `city`: Pakistani city
- `gender`: Male/Female (encoded as 0/1)
- `country`: Always "Pakistan"
- `timestamp`: When the ad was shown

### Target Variable
- `clicked_on_ad`: Binary (0 = No click, 1 = Click)

## Model Performance

The system supports multiple ML algorithms:

- **Random Forest**: Best for feature importance analysis
- **Gradient Boosting**: High accuracy with ensemble methods
- **Logistic Regression**: Fast and interpretable
- **Support Vector Machine**: Good for complex patterns

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-validation scores

## API Integration

### Watson ML Prediction
```python
from watson_integration import get_watson_client

client = get_watson_client(api_key='your-key')
result = client.make_prediction(input_data)

if result['success']:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1f}%")
```

### Batch Processing
```python
# Process multiple records
results = client.batch_predict(dataframe)
```

## Troubleshooting

### Common Issues

1. **Watson ML Authentication Error**
   - Verify API key is correct
   - Check deployment ID
   - Ensure Watson ML service is active

2. **Data Generation Fails**
   - Check write permissions in data/ directory
   - Verify pandas and numpy are installed

3. **Streamlit Won't Start**
   - Ensure all dependencies are installed
   - Check Python version (3.6+)
   - Try running with `python -m streamlit run streamlit_app.py`

### Mock Service

If Watson ML is not available, the system automatically uses a mock service:
- Rule-based predictions
- No API key required
- Good for testing and development

## Development

### Adding New Features

1. **New Cities**: Update `PAKISTANI_CITIES` in `config.py`
2. **New Models**: Add to `initialize_models()` in `model_trainer.py`
3. **New Metrics**: Extend evaluation functions
4. **UI Components**: Add to Streamlit app sections

### Testing

```bash
# Run data generation test
python data_generator.py

# Test Watson integration
python watson_integration.py

# Test data processing
python data_processor.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review IBM Watson ML documentation
- Create an issue in the repository

## Acknowledgments

- IBM Watson Machine Learning for prediction services
- Streamlit for the web interface framework
- Pakistani market data sources and research
- Open source Python ML ecosystem

---

**Built with ❤️ for the Pakistani market | Compatible with Python 3.6+ (2018 standards)**
