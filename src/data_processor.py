import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.outliers_count = 0
        
    def process_data(self, df, date_column, sales_column, advertising_columns):
        """
        Comprehensive data processing pipeline
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # 1. Date processing
        data = self._process_dates(data, date_column)
        
        # 2. Clean and validate sales data
        data = self._clean_sales_data(data, sales_column)
        
        # 3. Process advertising columns
        data = self._process_advertising_data(data, advertising_columns)
        
        # 4. Handle missing values
        data = self._handle_missing_values(data)
        
        # 5. Feature engineering
        data = self._create_features(data, sales_column, advertising_columns)
        
        # 6. Outlier detection and handling
        data = self._handle_outliers(data, sales_column)
        
        # 7. Final validation
        data = self._final_validation(data)
        
        return data
    
    def _process_dates(self, data, date_column):
        """Process date column and create time-based features"""
        try:
            # Convert to datetime
            data[date_column] = pd.to_datetime(data[date_column])
            
            # Sort by date
            data = data.sort_values(date_column).reset_index(drop=True)
            
            # Set date as index for time series analysis
            data.set_index(date_column, inplace=True)
            
            # Create time-based features
            data['year'] = data.index.year
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['day_of_week'] = data.index.dayofweek
            data['day_of_year'] = data.index.dayofyear
            
            # Create cyclical features for seasonality
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
            data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error processing date column: {str(e)}")
    
    def _clean_sales_data(self, data, sales_column):
        """Clean and validate sales data"""
        # Rename sales column to 'sales' for consistency
        if sales_column != 'sales':
            data['sales'] = data[sales_column]
            data.drop(columns=[sales_column], inplace=True)
        
        # Convert to numeric, coerce errors to NaN
        data['sales'] = pd.to_numeric(data['sales'], errors='coerce')
        
        # Remove negative sales (likely data errors)
        negative_sales = data['sales'] < 0
        if negative_sales.any():
            data.loc[negative_sales, 'sales'] = np.nan
        
        return data
    
    def _process_advertising_data(self, data, advertising_columns):
        """Process advertising spend data"""
        for col in advertising_columns:
            if col in data.columns:
                # Convert to numeric
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Replace negative values with 0 (no spend)
                data[col] = data[col].clip(lower=0)
                
                # Fill NaN with 0 for advertising spend
                data[col] = data[col].fillna(0)
        
        return data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        # For sales, use forward fill then backward fill
        if 'sales' in data.columns:
            data['sales'] = data['sales'].ffill().bfill()
        
        # For other numeric columns, use mean imputation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'sales' and data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mean())
        
        return data
    
    def _create_features(self, data, sales_column, advertising_columns):
        """Create engineered features"""
        # Lag features for sales (previous periods)
        for lag in [1, 2, 3]:
            data[f'sales_lag_{lag}'] = data['sales'].shift(lag)
        
        # Rolling window features
        for window in [3, 7, 14]:
            data[f'sales_rolling_mean_{window}'] = data['sales'].rolling(window=window).mean()
            data[f'sales_rolling_std_{window}'] = data['sales'].rolling(window=window).std()
        
        # Advertising features
        if advertising_columns:
            # Total advertising spend
            ad_cols_present = [col for col in advertising_columns if col in data.columns]
            if ad_cols_present:
                data['total_ad_spend'] = data[ad_cols_present].sum(axis=1)
                
                # Advertising mix (percentage of total for each channel)
                for col in ad_cols_present:
                    data[f'{col}_mix'] = data[col] / (data['total_ad_spend'] + 1e-8)
                
                # Advertising efficiency (sales per dollar spent)
                data['ad_efficiency'] = data['sales'] / (data['total_ad_spend'] + 1e-8)
                
                # Lag features for advertising
                for col in ad_cols_present:
                    for lag in [1, 2]:
                        data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Trend feature (linear trend over time)
        data['trend'] = range(len(data))
        
        return data
    
    def _handle_outliers(self, data, sales_column):
        """Detect and handle outliers using IQR method"""
        if 'sales' in data.columns:
            Q1 = data['sales'].quantile(0.25)
            Q3 = data['sales'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data['sales'] < lower_bound) | (data['sales'] > upper_bound)
            self.outliers_count = outliers.sum()
            
            # Cap outliers instead of removing them
            data.loc[data['sales'] < lower_bound, 'sales'] = lower_bound
            data.loc[data['sales'] > upper_bound, 'sales'] = upper_bound
        
        return data
    
    def _final_validation(self, data):
        """Final data validation and cleanup"""
        # Remove rows with NaN in sales column
        initial_length = len(data)
        data = data.dropna(subset=['sales'])
        final_length = len(data)
        
        if final_length < initial_length * 0.8:  # Lost more than 20% of data
            raise ValueError("Too much data lost during processing. Please check your data quality.")
        
        # Remove columns with too many NaN values
        threshold = 0.5  # Remove columns with more than 50% NaN
        data = data.dropna(axis=1, thresh=int(threshold * len(data)))
        
        # Fill remaining NaN values with 0
        data = data.fillna(0)
        
        # Ensure all data is numeric (except index)
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
                except:
                    data.drop(columns=[col], inplace=True)
        
        return data
    
    def get_feature_names(self):
        """Return list of created feature names"""
        return [
            'year', 'month', 'quarter', 'day_of_week', 'day_of_year',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
            'sales_rolling_mean_3', 'sales_rolling_mean_7', 'sales_rolling_mean_14',
            'sales_rolling_std_3', 'sales_rolling_std_7', 'sales_rolling_std_14',
            'total_ad_spend', 'ad_efficiency', 'trend'
        ]
