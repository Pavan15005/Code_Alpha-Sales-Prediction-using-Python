# Sales Prediction & Advertising Impact Analysis System

A comprehensive Streamlit-based web application for predicting sales and analyzing advertising impact using advanced machine learning models and time series forecasting.

## Features

### 🔍 Data Processing
- **Automated Data Cleaning**: Handles missing values, outliers, and data type conversions
- **Feature Engineering**: Creates lag features, rolling statistics, seasonal indicators, and advertising efficiency metrics
- **Data Validation**: Comprehensive validation with actionable feedback for data quality issues

### 🤖 Machine Learning Models
- **Linear Regression**: Simple baseline model for interpretable results
- **Polynomial Regression**: Captures non-linear relationships in the data
- **Random Forest**: Ensemble method with feature importance analysis
- **Time Series (ARIMA)**: Advanced forecasting for temporal patterns

### 📊 Interactive Visualizations
- **Model Performance Comparison**: Visual comparison of R², MAE, and RMSE metrics
- **Prediction vs Actual**: Line plots showing model accuracy over time
- **Residual Analysis**: Diagnostic plots for model validation
- **Advertising Impact**: Scatter plots showing spend vs sales relationships
- **Feature Importance**: Bar charts for understanding key drivers

### 🎯 Advanced Analytics
- **Scenario Planning**: What-if analysis for different advertising spend levels
- **Future Forecasting**: Predict sales for upcoming periods
- **ROAS Analysis**: Return on advertising spend calculations
- **Business Insights**: Automated generation of actionable recommendations

### 📤 Export Capabilities
- **Performance Reports**: Detailed CSV reports of model metrics
- **Scenario Results**: Export scenario analysis for strategic planning
- **Business Summary**: Comprehensive text reports with insights
- **Processed Data**: Export cleaned and feature-engineered datasets

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Dependencies
```bash
# Core libraries
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Time series analysis
statsmodels>=0.14.0

# Utilities
python-dateutil>=2.8.0
