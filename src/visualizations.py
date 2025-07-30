import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

class Visualizer:
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_correlation_matrix(self, data):
        """Create correlation matrix heatmap"""
        # Calculate correlation matrix for numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_model_comparison(self, df_performance, metric, title):
        """Create bar chart comparing model performance"""
        fig = go.Figure(data=[
            go.Bar(
                x=df_performance['Model'],
                y=df_performance[metric],
                marker_color=self.color_palette[:len(df_performance)],
                text=df_performance[metric],
                texttemplate='%{text}',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title=metric,
            showlegend=False
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df):
        """Create horizontal bar chart for feature importance"""
        fig = go.Figure(data=[
            go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker_color='lightblue',
                text=np.round(importance_df['Importance'], 3),
                texttemplate='%{text}',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Feature Importance (Random Forest)",
            xaxis_title="Importance",
            yaxis_title="Features",
            height=max(400, len(importance_df) * 25)
        )
        
        return fig
    
    def plot_predictions_vs_actual(self, y_actual, y_pred, dates=None):
        """Create line plot comparing predictions vs actual values"""
        if dates is None:
            dates = range(len(y_actual))
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_actual,
            mode='lines',
            name='Actual Sales',
            line=dict(color='blue', width=2)
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_pred,
            mode='lines',
            name='Predicted Sales',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Actual vs Predicted Sales",
            xaxis_title="Time Period",
            yaxis_title="Sales",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_residuals(self, residuals, y_pred):
        """Create residual plots for model diagnostics"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residuals vs Fitted", "Residual Distribution"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residual histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Distribution',
                nbinsx=20,
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_layout(
            title="Model Diagnostics",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_advertising_impact(self, data, ad_column, sales_column):
        """Create scatter plot showing advertising impact on sales"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[ad_column],
            y=data[sales_column],
            mode='markers',
            name=f'{ad_column} Impact',
            marker=dict(
                color=data[sales_column],
                colorscale='Viridis',
                size=8,
                colorbar=dict(title="Sales")
            ),
            text=[f"Date: {idx}<br>{ad_column}: ${x:,.0f}<br>Sales: ${y:,.0f}" 
                  for idx, x, y in zip(data.index, data[ad_column], data[sales_column])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add trend line
        z = np.polyfit(data[ad_column], data[sales_column], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=data[ad_column],
            y=p(data[ad_column]),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"Impact of {ad_column} on Sales",
            xaxis_title=f"{ad_column} Spend ($)",
            yaxis_title="Sales ($)",
            hovermode='closest'
        )
        
        return fig
    
    def plot_forecast(self, historical_sales, forecast, forecast_dates):
        """Create forecast visualization with historical data"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_sales.index,
            y=historical_sales.values,
            mode='lines',
            name='Historical Sales',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add vertical line separating historical and forecast
        if len(historical_sales) > 0:
            fig.add_vline(
                x=historical_sales.index[-1],
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast Start"
            )
        
        fig.update_layout(
            title="Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified'
        )
        
        return fig
    
    def plot_scenario_comparison(self, df_scenarios, metric, title):
        """Create bar chart comparing scenarios"""
        fig = go.Figure(data=[
            go.Bar(
                x=df_scenarios['Scenario'],
                y=df_scenarios[metric],
                marker_color=self.color_palette[:len(df_scenarios)],
                text=[f"${x:,.0f}" if 'Sales' in metric else f"{x:.2f}" 
                      for x in df_scenarios[metric]],
                texttemplate='%{text}',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Scenario",
            yaxis_title=metric,
            showlegend=False
        )
        
        return fig
    
    def plot_time_series_decomposition(self, data, sales_column):
        """Create time series decomposition plot"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Perform decomposition
            decomposition = seasonal_decompose(data[sales_column], model='additive', period=12)
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.1
            )
            
            # Original
            fig.add_trace(go.Scatter(x=data.index, y=data[sales_column], name='Original'), row=1, col=1)
            
            # Trend
            fig.add_trace(go.Scatter(x=data.index, y=decomposition.trend, name='Trend'), row=2, col=1)
            
            # Seasonal
            fig.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
            
            # Residual
            fig.add_trace(go.Scatter(x=data.index, y=decomposition.resid, name='Residual'), row=4, col=1)
            
            fig.update_layout(
                title="Time Series Decomposition",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure if decomposition fails
            fig = go.Figure()
            fig.add_annotation(
                text=f"Time series decomposition failed: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            return fig
    
    def plot_sales_trends(self, data, sales_column, groupby='month'):
        """Create sales trend visualization"""
        if groupby == 'month':
            grouped_data = data.groupby(data.index.month)[sales_column].mean()
            x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            title = "Average Sales by Month"
        elif groupby == 'quarter':
            grouped_data = data.groupby(data.index.quarter)[sales_column].mean()
            x_labels = ['Q1', 'Q2', 'Q3', 'Q4']
            title = "Average Sales by Quarter"
        else:
            grouped_data = data.groupby(data.index.year)[sales_column].mean()
            x_labels = grouped_data.index
            title = "Average Sales by Year"
        
        fig = go.Figure(data=[
            go.Bar(
                x=x_labels[:len(grouped_data)],
                y=grouped_data.values,
                marker_color='lightblue',
                text=[f"${x:,.0f}" for x in grouped_data.values],
                texttemplate='%{text}',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=groupby.title(),
            yaxis_title="Average Sales ($)",
            showlegend=False
        )
        
        return fig
