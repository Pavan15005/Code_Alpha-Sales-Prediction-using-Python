import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class SalesPredictionModels:
    def __init__(self):
        self.linear_model = None
        self.polynomial_model = None
        self.poly_features = None
        self.random_forest_model = None
        self.arima_model = None
        self.arima_fitted = None
        
    def train_all_models(self, X, y, test_size=0.2, random_state=42, cv_folds=5, selected_models=None):
        """
        Train all selected models and return performance metrics
        """
        results = {}
        
        # Split data for traditional ML models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        if selected_models is None:
            selected_models = ['Linear Regression', 'Polynomial Regression', 'Random Forest', 'Time Series (ARIMA)']
        
        # Train Linear Regression
        if 'Linear Regression' in selected_models:
            results['Linear Regression'] = self._train_linear_regression(
                X_train, X_test, y_train, y_test, cv_folds
            )
        
        # Train Polynomial Regression
        if 'Polynomial Regression' in selected_models:
            results['Polynomial Regression'] = self._train_polynomial_regression(
                X_train, X_test, y_train, y_test, cv_folds
            )
        
        # Train Random Forest
        if 'Random Forest' in selected_models:
            results['Random Forest'] = self._train_random_forest(
                X_train, X_test, y_train, y_test, cv_folds, X.columns
            )
        
        # Train ARIMA (Time Series)
        if 'Time Series (ARIMA)' in selected_models:
            results['Time Series (ARIMA)'] = self._train_arima(y, test_size)
        
        return results
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test, cv_folds):
        """Train Linear Regression model"""
        try:
            self.linear_model = LinearRegression()
            self.linear_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = self.linear_model.predict(X_train)
            y_pred_test = self.linear_model.predict(X_test)
            
            # Metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.linear_model, X_train, y_train, cv=cv_folds, scoring='r2')
            
            return {
                'model': self.linear_model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'coefficients': self.linear_model.coef_,
                'feature_names': X_train.columns.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _train_polynomial_regression(self, X_train, X_test, y_train, y_test, cv_folds):
        """Train Polynomial Regression model"""
        try:
            # Use degree 2 to avoid overfitting
            self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
            
            # Transform features
            X_train_poly = self.poly_features.fit_transform(X_train)
            X_test_poly = self.poly_features.transform(X_test)
            
            # Train model
            self.polynomial_model = LinearRegression()
            self.polynomial_model.fit(X_train_poly, y_train)
            
            # Predictions
            y_pred_train = self.polynomial_model.predict(X_train_poly)
            y_pred_test = self.polynomial_model.predict(X_test_poly)
            
            # Metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation (on polynomial features)
            cv_scores = cross_val_score(
                LinearRegression(), X_train_poly, y_train, cv=cv_folds, scoring='r2'
            )
            
            return {
                'model': self.polynomial_model,
                'poly_features': self.poly_features,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test, cv_folds, feature_names):
        """Train Random Forest model"""
        try:
            self.random_forest_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.random_forest_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = self.random_forest_model.predict(X_train)
            y_pred_test = self.random_forest_model.predict(X_test)
            
            # Metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.random_forest_model, X_train, y_train, cv=cv_folds, scoring='r2')
            
            # Feature importance
            feature_importance = self.random_forest_model.feature_importances_
            
            return {
                'model': self.random_forest_model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'feature_names': feature_names
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _train_arima(self, y, test_size):
        """Train ARIMA time series model"""
        try:
            # Split time series data
            train_size = int(len(y) * (1 - test_size))
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            # Auto-select ARIMA parameters (simplified approach)
            # In a production environment, you might want to use auto_arima
            best_aic = float('inf')
            best_order = None
            
            # Grid search for optimal parameters (limited to avoid long computation)
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(y_train, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                        except:
                            continue
            
            if best_order is None:
                best_order = (1, 1, 1)  # Default fallback
            
            # Fit final model
            self.arima_model = ARIMA(y_train, order=best_order)
            self.arima_fitted = self.arima_model.fit()
            
            # Make predictions
            forecast_steps = len(y_test)
            forecast = self.arima_fitted.forecast(steps=forecast_steps)
            
            # Calculate metrics
            test_metrics = self._calculate_metrics(y_test, forecast)
            
            return {
                'model': self.arima_fitted,
                'order': best_order,
                'aic': best_aic,
                'test_metrics': test_metrics,
                'forecast': forecast
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).all() else 0
        }
    
    def predict_timeseries(self, steps):
        """Make time series predictions using ARIMA model"""
        if self.arima_fitted is None:
            raise ValueError("ARIMA model not trained yet")
        
        try:
            forecast = self.arima_fitted.forecast(steps=steps)
            return forecast
        except Exception as e:
            raise ValueError(f"Error making time series prediction: {str(e)}")
    
    def predict_regression(self, X, model_type='linear'):
        """Make predictions using regression models"""
        if model_type == 'linear':
            if self.linear_model is None:
                raise ValueError("Linear model not trained yet")
            return self.linear_model.predict(X)
        
        elif model_type == 'polynomial':
            if self.polynomial_model is None or self.poly_features is None:
                raise ValueError("Polynomial model not trained yet")
            X_poly = self.poly_features.transform(X)
            return self.polynomial_model.predict(X_poly)
        
        elif model_type == 'random_forest':
            if self.random_forest_model is None:
                raise ValueError("Random Forest model not trained yet")
            return self.random_forest_model.predict(X)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        summary = {}
        
        if self.linear_model is not None:
            summary['Linear Regression'] = 'Trained'
        
        if self.polynomial_model is not None:
            summary['Polynomial Regression'] = 'Trained'
        
        if self.random_forest_model is not None:
            summary['Random Forest'] = 'Trained'
        
        if self.arima_fitted is not None:
            summary['ARIMA'] = f'Trained (order: {self.arima_fitted.model.order})'
        
        return summary
