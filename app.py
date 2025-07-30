import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_processor import DataProcessor
from src.models import SalesPredictionModels
from src.visualizations import Visualizer
from src.utils import generate_insights, export_results

# Page configuration
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def main():
    st.title("📈 Sales Prediction & Advertising Impact Analysis")
    st.markdown("A comprehensive system for predicting sales and analyzing advertising impact using machine learning models.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a page:",
        ["Data Upload & Processing", "Model Training & Evaluation", "Predictions & Analysis", "Scenario Planning", "Export Results"]
    )
    
    if page == "Data Upload & Processing":
        data_upload_page()
    elif page == "Model Training & Evaluation":
        model_training_page()
    elif page == "Predictions & Analysis":
        predictions_page()
    elif page == "Scenario Planning":
        scenario_planning_page()
    elif page == "Export Results":
        export_page()

def data_upload_page():
    st.header("📊 Data Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your sales data (CSV format)",
        type=['csv'],
        help="Upload a CSV file containing sales data with columns like date, sales, advertising spend, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_data = df
            
            st.success("✅ Data uploaded successfully!")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Date Range", f"{len(df)} periods")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data processing options
            st.subheader("Data Processing Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                date_column = st.selectbox(
                    "Select Date Column",
                    options=df.columns.tolist(),
                    help="Choose the column containing date information"
                )
            
            with col2:
                sales_column = st.selectbox(
                    "Select Sales Column",
                    options=[col for col in df.columns if col != date_column],
                    help="Choose the column containing sales data"
                )
            
            # Advertising columns selection
            st.subheader("Advertising Spend Columns")
            advertising_columns = st.multiselect(
                "Select advertising spend columns",
                options=[col for col in df.columns if col not in [date_column, sales_column]],
                help="Select columns containing advertising spend data for different platforms/channels"
            )
            
            if st.button("Process Data", type="primary"):
                processor = DataProcessor()
                
                with st.spinner("Processing data..."):
                    try:
                        processed_data = processor.process_data(
                            df, date_column, sales_column, advertising_columns
                        )
                        
                        st.session_state.processed_data = processed_data
                        st.session_state.data_config = {
                            'date_column': date_column,
                            'sales_column': sales_column,
                            'advertising_columns': advertising_columns
                        }
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Data processed successfully!")
                        
                        # Display processed data summary
                        st.subheader("Processed Data Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Clean Rows", len(processed_data))
                        with col2:
                            st.metric("Features Created", len(processed_data.columns) - 1)
                        with col3:
                            st.metric("Missing Values", processed_data.isnull().sum().sum())
                        with col4:
                            st.metric("Outliers Detected", processor.outliers_count if hasattr(processor, 'outliers_count') else 0)
                        
                        # Show feature correlation
                        if len(advertising_columns) > 0:
                            st.subheader("Feature Correlation Matrix")
                            visualizer = Visualizer()
                            fig = visualizer.plot_correlation_matrix(processed_data)
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing data: {str(e)}")
                        st.info("Please check your data format and column selections.")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")
    
    else:
        st.info("👆 Please upload a CSV file to get started.")
        
        # Sample data format guidance
        st.subheader("Expected Data Format")
        st.markdown("""
        Your CSV file should contain the following columns:
        - **Date column**: Date/time information (e.g., 'date', 'month', 'period')
        - **Sales column**: Sales figures (e.g., 'sales', 'revenue', 'units_sold')
        - **Advertising columns**: Spend data for different channels (e.g., 'tv_spend', 'digital_spend', 'print_spend')
        
        Example:
        ```
        date,sales,tv_spend,digital_spend,social_spend
        2023-01-01,50000,10000,5000,2000
        2023-01-02,52000,12000,5500,2200
        ...
        ```
        """)

def model_training_page():
    st.header("🤖 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload and process data first in the 'Data Upload & Processing' page.")
        return
    
    data = st.session_state.processed_data
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, help="Percentage of data to use for testing")
        random_state = st.number_input("Random State", 0, 100, 42, help="Seed for reproducible results")
    
    with col2:
        cross_validation = st.checkbox("Enable Cross-Validation", True, help="Use k-fold cross-validation")
        if cross_validation:
            cv_folds = st.selectbox("CV Folds", [3, 5, 10], index=1)
    
    # Feature selection
    st.subheader("Feature Selection")
    available_features = [col for col in data.columns if col != 'sales']
    selected_features = st.multiselect(
        "Select features for modeling",
        options=available_features,
        default=available_features,
        help="Choose which features to include in the models"
    )
    
    if len(selected_features) == 0:
        st.warning("Please select at least one feature for modeling.")
        return
    
    # Model selection
    st.subheader("Model Selection")
    model_options = {
        'Linear Regression': st.checkbox("Linear Regression", True),
        'Polynomial Regression': st.checkbox("Polynomial Regression", True),
        'Random Forest': st.checkbox("Random Forest", True),
        'Time Series (ARIMA)': st.checkbox("Time Series (ARIMA)", True)
    }
    
    selected_models = [model for model, selected in model_options.items() if selected]
    
    if len(selected_models) == 0:
        st.warning("Please select at least one model to train.")
        return
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                models = SalesPredictionModels()
                
                # Prepare data
                X = data[selected_features]
                y = data['sales']
                
                # Train models
                results = models.train_all_models(
                    X, y, 
                    test_size=test_size/100,
                    random_state=random_state,
                    cv_folds=cv_folds if cross_validation else 5,
                    selected_models=selected_models
                )
                
                st.session_state.model_results = results
                st.session_state.trained_models = models
                st.session_state.selected_features = selected_features
                st.session_state.models_trained = True
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                display_model_results(results)
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")

def display_model_results(results):
    st.subheader("📊 Model Performance Comparison")
    
    # Create performance dataframe
    performance_data = []
    for model_name, metrics in results.items():
        if 'test_metrics' in metrics:
            performance_data.append({
                'Model': model_name,
                'R²': round(metrics['test_metrics']['r2'], 4),
                'MAE': round(metrics['test_metrics']['mae'], 2),
                'RMSE': round(metrics['test_metrics']['rmse'], 2),
                'CV Score': round(metrics.get('cv_score', 0), 4) if metrics.get('cv_score') else 'N/A'
            })
    
    if performance_data:
        df_performance = pd.DataFrame(performance_data)
        
        # Highlight best performing model
        def highlight_best(s):
            if s.name == 'R²' or s.name == 'CV Score':
                best_idx = s.idxmax() if s.name != 'CV Score' or all(isinstance(x, (int, float)) for x in s) else None
            else:
                best_idx = s.idxmin()
            
            colors = ['background-color: lightgreen' if i == best_idx else '' for i in range(len(s))]
            return colors
        
        styled_df = df_performance.style.apply(highlight_best, subset=['R²', 'MAE', 'RMSE'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Model performance visualization
        st.subheader("📈 Model Performance Visualization")
        visualizer = Visualizer()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_r2 = visualizer.plot_model_comparison(df_performance, 'R²', 'Model R² Comparison')
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            fig_mae = visualizer.plot_model_comparison(df_performance, 'MAE', 'Model MAE Comparison')
            st.plotly_chart(fig_mae, use_container_width=True)
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
            st.subheader("🎯 Feature Importance (Random Forest)")
            importance_df = pd.DataFrame({
                'Feature': st.session_state.selected_features,
                'Importance': results['Random Forest']['feature_importance']
            }).sort_values('Importance', ascending=False)
            
            fig_importance = visualizer.plot_feature_importance(importance_df)
            st.plotly_chart(fig_importance, use_container_width=True)

def predictions_page():
    st.header("🔮 Predictions & Analysis")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Model Training & Evaluation' page.")
        return
    
    data = st.session_state.processed_data
    models = st.session_state.trained_models
    results = st.session_state.model_results
    
    # Model selection for predictions
    st.subheader("Model Selection")
    available_models = list(results.keys())
    selected_model = st.selectbox(
        "Choose a model for predictions:",
        options=available_models,
        help="Select the model to use for generating predictions"
    )
    
    # Prediction options
    st.subheader("Prediction Options")
    prediction_type = st.radio(
        "Prediction Type:",
        ["Historical Analysis", "Future Forecast"],
        help="Choose whether to analyze historical data or forecast future values"
    )
    
    if prediction_type == "Historical Analysis":
        # Historical predictions and analysis
        st.subheader("📊 Historical Predictions vs Actual")
        
        try:
            # Get predictions for the entire dataset
            X = data[st.session_state.selected_features]
            y_actual = data['sales']
            
            if selected_model == 'Time Series (ARIMA)':
                y_pred = models.predict_timeseries(len(data))
            else:
                # Map model names to methods
                model_map = {
                    'Linear Regression': 'linear_model',
                    'Polynomial Regression': 'polynomial_model', 
                    'Random Forest': 'random_forest_model'
                }
                model_attr = model_map.get(selected_model)
                if model_attr:
                    model_obj = getattr(models, model_attr)
                    if selected_model == 'Polynomial Regression':
                        X_poly = models.poly_features.transform(X)
                        y_pred = model_obj.predict(X_poly)
                    else:
                        y_pred = model_obj.predict(X)
                else:
                    st.error(f"Unknown model: {selected_model}")
                    return
            
            # Create visualization
            visualizer = Visualizer()
            fig_pred = visualizer.plot_predictions_vs_actual(y_actual, y_pred, data.index)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Residual analysis
            st.subheader("📉 Residual Analysis")
            residuals = y_actual - y_pred
            fig_residuals = visualizer.plot_residuals(residuals, y_pred)
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Advertising impact analysis
            if len(st.session_state.data_config['advertising_columns']) > 0:
                st.subheader("📺 Advertising Impact Analysis")
                ad_columns = st.session_state.data_config['advertising_columns']
                
                for ad_col in ad_columns:
                    if ad_col in data.columns:
                        fig_impact = visualizer.plot_advertising_impact(data, ad_col, 'sales')
                        st.plotly_chart(fig_impact, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error generating predictions: {str(e)}")
    
    else:
        # Future forecasting
        st.subheader("🔮 Future Forecast")
        
        col1, col2 = st.columns(2)
        with col1:
            forecast_periods = st.number_input("Forecast Periods", 1, 100, 12, help="Number of periods to forecast")
        with col2:
            confidence_interval = st.slider("Confidence Interval (%)", 80, 99, 95)
        
        # Input for future advertising spend
        future_ad_spend = {}
        if len(st.session_state.data_config['advertising_columns']) > 0:
            st.subheader("Future Advertising Spend")
            
            for ad_col in st.session_state.data_config['advertising_columns']:
                if ad_col in st.session_state.selected_features:
                    current_avg = data[ad_col].mean()
                    future_ad_spend[ad_col] = st.number_input(
                        f"Average {ad_col} per period",
                        value=float(current_avg),
                        min_value=0.0,
                        help=f"Current average: {current_avg:.2f}"
                    )
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                try:
                    if selected_model == 'Time Series (ARIMA)':
                        forecast = models.predict_timeseries(forecast_periods)
                        dates = pd.date_range(
                            start=data.index[-1] + timedelta(days=1),
                            periods=forecast_periods,
                            freq='D'
                        )
                    else:
                        # Create future feature matrix
                        if future_ad_spend:
                            future_features = []
                            for i in range(forecast_periods):
                                features = {}
                                for feature in st.session_state.selected_features:
                                    if feature in future_ad_spend:
                                        features[feature] = future_ad_spend[feature]
                                    else:
                                        # Use average of last few periods
                                        features[feature] = data[feature].tail(5).mean()
                                future_features.append(features)
                            
                            future_df = pd.DataFrame(future_features)
                            # Use the same mapping as before
                            model_map = {
                                'Linear Regression': 'linear_model',
                                'Polynomial Regression': 'polynomial_model', 
                                'Random Forest': 'random_forest_model'
                            }
                            model_attr = model_map.get(selected_model)
                            if model_attr:
                                model_obj = getattr(models, model_attr)
                                if selected_model == 'Polynomial Regression':
                                    future_df_poly = models.poly_features.transform(future_df)
                                    forecast = model_obj.predict(future_df_poly)
                                else:
                                    forecast = model_obj.predict(future_df)
                            else:
                                st.error(f"Unknown model: {selected_model}")
                                return
                            dates = pd.date_range(
                                start=data.index[-1] + timedelta(days=1),
                                periods=forecast_periods,
                                freq='D'
                            )
                        else:
                            st.warning("Please specify future advertising spend values.")
                            return
                    
                    # Display forecast
                    st.subheader("📈 Forecast Results")
                    
                    forecast_df = pd.DataFrame({
                        'Date': dates,
                        'Predicted Sales': forecast
                    })
                    
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Visualize forecast
                    visualizer = Visualizer()
                    fig_forecast = visualizer.plot_forecast(data['sales'], forecast, dates)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predicted Sales", f"${forecast.sum():,.2f}")
                    with col2:
                        st.metric("Average Per Period", f"${forecast.mean():,.2f}")
                    with col3:
                        growth_rate = ((forecast.mean() - data['sales'].tail(10).mean()) / data['sales'].tail(10).mean()) * 100
                        st.metric("Growth Rate", f"{growth_rate:.1f}%")
                
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")

def scenario_planning_page():
    st.header("🎯 Scenario Planning & What-If Analysis")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Model Training & Evaluation' page.")
        return
    
    data = st.session_state.processed_data
    models = st.session_state.trained_models
    ad_columns = st.session_state.data_config['advertising_columns']
    
    if len(ad_columns) == 0:
        st.warning("⚠️ No advertising columns available for scenario analysis.")
        return
    
    st.subheader("📊 Current vs Scenario Comparison")
    
    # Current baseline
    st.subheader("Current Baseline")
    current_spend = {}
    baseline_metrics = {}
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Current Average Spending:**")
        for ad_col in ad_columns:
            if ad_col in data.columns:
                avg_spend = data[ad_col].mean()
                current_spend[ad_col] = avg_spend
                st.metric(f"{ad_col}", f"${avg_spend:,.2f}")
    
    with col2:
        st.write("**Current Performance:**")
        current_sales = data['sales'].mean()
        st.metric("Average Sales", f"${current_sales:,.2f}")
        
        total_ad_spend = sum(current_spend.values())
        if total_ad_spend > 0:
            roas = current_sales / total_ad_spend
            st.metric("Return on Ad Spend (ROAS)", f"{roas:.2f}")
    
    # Scenario creation
    st.subheader("🎬 Create Scenarios")
    
    num_scenarios = st.selectbox("Number of scenarios to compare:", [1, 2, 3, 4], index=1)
    
    scenarios = []
    scenario_names = []
    
    for i in range(num_scenarios):
        st.subheader(f"Scenario {i+1}")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            scenario_name = st.text_input(f"Scenario {i+1} Name", f"Scenario {i+1}")
            scenario_names.append(scenario_name)
        
        with col2:
            change_type = st.selectbox(
                f"Change Type for {scenario_name}:",
                ["Percentage Change", "Absolute Values"],
                key=f"change_type_{i}"
            )
        
        scenario_spend = {}
        for ad_col in ad_columns:
            if ad_col in st.session_state.selected_features:
                if change_type == "Percentage Change":
                    change_pct = st.slider(
                        f"{ad_col} change (%)",
                        -50, 200, 0,
                        key=f"scenario_{i}_{ad_col}",
                        help=f"Current: ${current_spend[ad_col]:,.2f}"
                    )
                    scenario_spend[ad_col] = current_spend[ad_col] * (1 + change_pct/100)
                else:
                    scenario_spend[ad_col] = st.number_input(
                        f"{ad_col} spend",
                        value=current_spend[ad_col],
                        min_value=0.0,
                        key=f"scenario_abs_{i}_{ad_col}"
                    )
        
        scenarios.append(scenario_spend)
        st.divider()
    
    # Run scenario analysis
    if st.button("Run Scenario Analysis", type="primary"):
        with st.spinner("Analyzing scenarios..."):
            try:
                # Select best performing model for scenarios
                results = st.session_state.model_results
                best_model = max(results.keys(), key=lambda k: results[k].get('test_metrics', {}).get('r2', 0))
                
                st.info(f"Using {best_model} for scenario analysis (best R² score)")
                
                scenario_results = []
                
                # Baseline scenario
                baseline_features = {feature: data[feature].mean() for feature in st.session_state.selected_features}
                if best_model == 'Time Series (ARIMA)':
                    baseline_pred = models.predict_timeseries(1)[0]
                else:
                    model_obj = getattr(models, f"{best_model.lower().replace(' ', '_')}_model")
                    baseline_pred = model_obj.predict([list(baseline_features.values())])[0]
                
                scenario_results.append({
                    'Scenario': 'Current Baseline',
                    'Predicted Sales': baseline_pred,
                    'Total Ad Spend': sum(current_spend.values()),
                    'ROAS': baseline_pred / sum(current_spend.values()) if sum(current_spend.values()) > 0 else 0,
                    **{f"{col}_spend": current_spend.get(col, 0) for col in ad_columns}
                })
                
                # Scenario predictions
                for i, scenario in enumerate(scenarios):
                    scenario_features = baseline_features.copy()
                    scenario_features.update(scenario)
                    
                    if best_model == 'Time Series (ARIMA)':
                        # For time series, we can't easily do what-if analysis
                        pred_sales = baseline_pred  # Fallback to baseline
                    else:
                        model_obj = getattr(models, f"{best_model.lower().replace(' ', '_')}_model")
                        pred_sales = model_obj.predict([list(scenario_features.values())])[0]
                    
                    total_spend = sum(scenario.values())
                    roas = pred_sales / total_spend if total_spend > 0 else 0
                    
                    scenario_results.append({
                        'Scenario': scenario_names[i],
                        'Predicted Sales': pred_sales,
                        'Total Ad Spend': total_spend,
                        'ROAS': roas,
                        **{f"{col}_spend": scenario.get(col, 0) for col in ad_columns}
                    })
                
                # Display results
                st.subheader("📊 Scenario Analysis Results")
                
                df_scenarios = pd.DataFrame(scenario_results)
                
                # Format display
                display_df = df_scenarios.copy()
                display_df['Predicted Sales'] = display_df['Predicted Sales'].apply(lambda x: f"${x:,.2f}")
                display_df['Total Ad Spend'] = display_df['Total Ad Spend'].apply(lambda x: f"${x:,.2f}")
                display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.2f}")
                
                for col in ad_columns:
                    if f"{col}_spend" in display_df.columns:
                        display_df[f"{col}_spend"] = display_df[f"{col}_spend"].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualization
                visualizer = Visualizer()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_sales = visualizer.plot_scenario_comparison(df_scenarios, 'Predicted Sales', 'Sales Comparison')
                    st.plotly_chart(fig_sales, use_container_width=True)
                
                with col2:
                    fig_roas = visualizer.plot_scenario_comparison(df_scenarios, 'ROAS', 'ROAS Comparison')
                    st.plotly_chart(fig_roas, use_container_width=True)
                
                # Insights and recommendations
                st.subheader("💡 Insights & Recommendations")
                insights = generate_insights(df_scenarios, scenario_names)
                for insight in insights:
                    st.info(insight)
                
                # Store results for export
                st.session_state.scenario_results = df_scenarios
                
            except Exception as e:
                st.error(f"❌ Error analyzing scenarios: {str(e)}")

def export_page():
    st.header("📤 Export Results")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first before exporting results.")
        return
    
    st.subheader("Available Exports")
    
    # Model performance export
    if st.session_state.model_results:
        st.subheader("📊 Model Performance Report")
        
        if st.button("Generate Performance Report"):
            report = export_results.generate_performance_report(
                st.session_state.model_results,
                st.session_state.data_config
            )
            
            st.download_button(
                label="Download Performance Report (CSV)",
                data=report,
                file_name=f"model_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Predictions export
    if hasattr(st.session_state, 'scenario_results'):
        st.subheader("🎯 Scenario Analysis Export")
        
        if st.button("Export Scenario Results"):
            csv_data = st.session_state.scenario_results.to_csv(index=False)
            
            st.download_button(
                label="Download Scenario Results (CSV)",
                data=csv_data,
                file_name=f"scenario_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Complete data export
    if st.session_state.data_loaded:
        st.subheader("📋 Processed Data Export")
        
        if st.button("Export Processed Data"):
            csv_data = st.session_state.processed_data.to_csv()
            
            st.download_button(
                label="Download Processed Data (CSV)",
                data=csv_data,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Business insights export
    st.subheader("💼 Business Insights Summary")
    
    if st.button("Generate Business Summary"):
        summary = export_results.generate_business_summary(
            st.session_state.model_results if hasattr(st.session_state, 'model_results') else {},
            st.session_state.scenario_results if hasattr(st.session_state, 'scenario_results') else pd.DataFrame(),
            st.session_state.data_config if hasattr(st.session_state, 'data_config') else {}
        )
        
        st.download_button(
            label="Download Business Summary (TXT)",
            data=summary,
            file_name=f"business_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
