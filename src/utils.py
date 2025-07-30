import pandas as pd
import numpy as np
from datetime import datetime
import io

def generate_insights(scenario_df, scenario_names):
    """
    Generate business insights from scenario analysis
    """
    insights = []
    
    # Best performing scenario
    best_sales_idx = scenario_df['Predicted Sales'].idxmax()
    best_sales_scenario = scenario_df.loc[best_sales_idx, 'Scenario']
    best_sales_value = scenario_df.loc[best_sales_idx, 'Predicted Sales']
    
    insights.append(f"🏆 **Best Sales Performance**: {best_sales_scenario} is projected to generate the highest sales of ${best_sales_value:,.2f}")
    
    # Best ROAS
    best_roas_idx = scenario_df['ROAS'].idxmax()
    best_roas_scenario = scenario_df.loc[best_roas_idx, 'Scenario']
    best_roas_value = scenario_df.loc[best_roas_idx, 'ROAS']
    
    insights.append(f"💰 **Best Return on Ad Spend**: {best_roas_scenario} offers the highest ROAS of {best_roas_value:.2f}")
    
    # Most efficient spending
    baseline_idx = scenario_df[scenario_df['Scenario'] == 'Current Baseline'].index[0]
    baseline_roas = scenario_df.loc[baseline_idx, 'ROAS']
    
    improved_scenarios = scenario_df[scenario_df['ROAS'] > baseline_roas]
    if len(improved_scenarios) > 0:
        insights.append(f"📈 **Optimization Opportunity**: {len(improved_scenarios)} scenario(s) show better ROAS than current baseline")
    
    # Spending efficiency analysis
    max_spend_idx = scenario_df['Total Ad Spend'].idxmax()
    max_spend_scenario = scenario_df.loc[max_spend_idx, 'Scenario']
    max_spend_roas = scenario_df.loc[max_spend_idx, 'ROAS']
    
    if max_spend_roas < baseline_roas:
        insights.append(f"⚠️ **Diminishing Returns**: Higher spend in {max_spend_scenario} shows lower efficiency (ROAS: {max_spend_roas:.2f})")
    
    # Growth potential
    baseline_sales = scenario_df.loc[baseline_idx, 'Predicted Sales']
    max_improvement = ((best_sales_value - baseline_sales) / baseline_sales) * 100
    
    if max_improvement > 0:
        insights.append(f"🚀 **Growth Potential**: Optimized advertising could increase sales by up to {max_improvement:.1f}%")
    
    return insights

def generate_marketing_recommendations(scenario_df, ad_columns):
    """
    Generate specific marketing recommendations based on scenario analysis
    """
    recommendations = []
    
    # Find best scenario (excluding baseline)
    non_baseline = scenario_df[scenario_df['Scenario'] != 'Current Baseline']
    if len(non_baseline) > 0:
        best_scenario_idx = non_baseline['ROAS'].idxmax()
        best_scenario = non_baseline.loc[best_scenario_idx]
        
        recommendations.append("## 🎯 Recommended Action Plan:")
        recommendations.append(f"**Implement {best_scenario['Scenario']} strategy** for optimal results")
        
        # Channel-specific recommendations
        for col in ad_columns:
            spend_col = f"{col}_spend"
            if spend_col in scenario_df.columns:
                current_spend = scenario_df[scenario_df['Scenario'] == 'Current Baseline'][spend_col].iloc[0]
                recommended_spend = best_scenario[spend_col]
                change = ((recommended_spend - current_spend) / current_spend) * 100
                
                if abs(change) > 5:  # Only recommend if change is significant
                    direction = "increase" if change > 0 else "decrease"
                    recommendations.append(f"- **{col}**: {direction} spend by {abs(change):.1f}% (${current_spend:,.0f} → ${recommended_spend:,.0f})")
        
        # Expected outcomes
        expected_sales = best_scenario['Predicted Sales']
        expected_roas = best_scenario['ROAS']
        recommendations.append(f"**Expected Outcome**: ${expected_sales:,.2f} in sales with ROAS of {expected_roas:.2f}")
    
    return recommendations

class export_results:
    @staticmethod
    def generate_performance_report(model_results, data_config):
        """Generate CSV report of model performance"""
        report_data = []
        
        for model_name, results in model_results.items():
            if 'test_metrics' in results:
                report_data.append({
                    'Model': model_name,
                    'R_Squared': results['test_metrics']['r2'],
                    'MAE': results['test_metrics']['mae'],
                    'RMSE': results['test_metrics']['rmse'],
                    'MAPE': results['test_metrics'].get('mape', 'N/A'),
                    'CV_Score': results.get('cv_score', 'N/A'),
                    'CV_Std': results.get('cv_std', 'N/A'),
                    'Training_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        df = pd.DataFrame(report_data)
        return df.to_csv(index=False)
    
    @staticmethod
    def generate_business_summary(model_results, scenario_results, data_config):
        """Generate business summary text report"""
        summary = []
        summary.append("# SALES PREDICTION ANALYSIS SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # Data overview
        summary.append("## DATA OVERVIEW")
        summary.append("-" * 20)
        if data_config:
            summary.append(f"Sales Column: {data_config.get('sales_column', 'N/A')}")
            summary.append(f"Date Column: {data_config.get('date_column', 'N/A')}")
            ad_cols = data_config.get('advertising_columns', [])
            summary.append(f"Advertising Channels: {', '.join(ad_cols) if ad_cols else 'None'}")
        summary.append("")
        
        # Model performance
        summary.append("## MODEL PERFORMANCE")
        summary.append("-" * 25)
        if model_results:
            best_model = None
            best_r2 = -1
            
            for model_name, results in model_results.items():
                if 'test_metrics' in results:
                    r2 = results['test_metrics']['r2']
                    mae = results['test_metrics']['mae']
                    rmse = results['test_metrics']['rmse']
                    
                    summary.append(f"{model_name}:")
                    summary.append(f"  - R² Score: {r2:.4f}")
                    summary.append(f"  - MAE: ${mae:,.2f}")
                    summary.append(f"  - RMSE: ${rmse:,.2f}")
                    summary.append("")
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
            
            if best_model:
                summary.append(f"BEST PERFORMING MODEL: {best_model} (R² = {best_r2:.4f})")
        summary.append("")
        
        # Scenario analysis
        if not scenario_results.empty:
            summary.append("## SCENARIO ANALYSIS")
            summary.append("-" * 23)
            
            baseline = scenario_results[scenario_results['Scenario'] == 'Current Baseline']
            if not baseline.empty:
                baseline_sales = baseline['Predicted Sales'].iloc[0]
                baseline_roas = baseline['ROAS'].iloc[0]
                
                summary.append(f"Current Baseline:")
                summary.append(f"  - Predicted Sales: ${baseline_sales:,.2f}")
                summary.append(f"  - ROAS: {baseline_roas:.2f}")
                summary.append("")
                
                # Best scenarios
                other_scenarios = scenario_results[scenario_results['Scenario'] != 'Current Baseline']
                if not other_scenarios.empty:
                    best_sales = other_scenarios.loc[other_scenarios['Predicted Sales'].idxmax()]
                    best_roas = other_scenarios.loc[other_scenarios['ROAS'].idxmax()]
                    
                    summary.append("Top Performing Scenarios:")
                    summary.append(f"  - Highest Sales: {best_sales['Scenario']} (${best_sales['Predicted Sales']:,.2f})")
                    summary.append(f"  - Best ROAS: {best_roas['Scenario']} ({best_roas['ROAS']:.2f})")
        
        summary.append("")
        summary.append("## RECOMMENDATIONS")
        summary.append("-" * 20)
        summary.append("1. Use the best performing model for future predictions")
        summary.append("2. Implement the highest ROAS scenario for budget optimization")
        summary.append("3. Monitor model performance regularly and retrain as needed")
        summary.append("4. Consider seasonal factors in advertising planning")
        summary.append("")
        summary.append("Note: This analysis is based on historical data and model predictions.")
        summary.append("Actual results may vary due to market conditions and external factors.")
        
        return "\n".join(summary)

def validate_data_format(df):
    """
    Validate uploaded data format and provide feedback
    """
    issues = []
    recommendations = []
    
    # Check for minimum data requirements
    if len(df) < 30:
        issues.append("Dataset has less than 30 rows - may not be sufficient for reliable predictions")
        recommendations.append("Consider collecting more data points for better model performance")
    
    # Check for date column
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head())
                date_columns.append(col)
            except:
                continue
    
    if not date_columns:
        issues.append("No clear date column found")
        recommendations.append("Ensure your data has a properly formatted date column")
    
    # Check for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) < 2:
        issues.append("Insufficient numeric columns for analysis")
        recommendations.append("Ensure you have at least one sales column and one advertising spend column")
    
    # Check for missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_percentage[missing_percentage > 50]
    
    if not high_missing.empty:
        issues.append(f"High missing values in columns: {', '.join(high_missing.index)}")
        recommendations.append("Consider removing or imputing columns with excessive missing values")
    
    return issues, recommendations

def create_sample_data_template():
    """
    Create a sample data template for users
    """
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Base sales with trend and seasonality
    trend = np.linspace(45000, 55000, len(dates))
    seasonality = 5000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 2000, len(dates))
    
    # Advertising spend with some correlation to sales
    tv_spend = np.random.uniform(8000, 15000, len(dates))
    digital_spend = np.random.uniform(3000, 8000, len(dates))
    social_spend = np.random.uniform(1000, 4000, len(dates))
    
    # Sales influenced by advertising (simplified relationship)
    ad_effect = (tv_spend * 0.8 + digital_spend * 1.2 + social_spend * 1.5)
    sales = trend + seasonality + ad_effect * 0.1 + noise
    
    sample_data = pd.DataFrame({
        'date': dates,
        'sales': np.round(sales, 2),
        'tv_spend': np.round(tv_spend, 2),
        'digital_spend': np.round(digital_spend, 2),
        'social_spend': np.round(social_spend, 2)
    })
    
    return sample_data
