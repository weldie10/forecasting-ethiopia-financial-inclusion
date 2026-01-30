"""
Streamlit Dashboard for Ethiopia Financial Inclusion Forecasting
Interactive dashboard for exploring data, event impacts, and forecasts.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dashboard_components import DashboardDataLoader, DashboardVisualizations

# Configure page
st.set_page_config(
    page_title="Ethiopia Financial Inclusion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data_loader = None
    st.session_state.viz = None

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize data loader and visualizations."""
    data_file = Path(__file__).parent.parent / 'data' / 'raw' / 'ethiopia_fi_unified_data.xlsx'
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    loader = DashboardDataLoader(data_file, processed_dir)
    viz = DashboardVisualizations()
    
    return loader, viz

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Trends", "Forecasts", "Inclusion Projections"]
)

# Initialize components
if not st.session_state.data_loaded:
    with st.spinner("Loading data..."):
        loader, viz = initialize_components()
        loader.load_data()
        loader.load_forecasts()
        loader.load_association_matrix()
        
        st.session_state.data_loader = loader
        st.session_state.viz = viz
        st.session_state.data_loaded = True

loader = st.session_state.data_loader
viz = st.session_state.viz

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if page == "Overview":
    st.title("📈 Overview")
    st.markdown("### Key Metrics and Summary")
    
    # Get key metrics
    metrics = loader.get_key_metrics()
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'account_ownership' in metrics:
            st.metric(
                label="Account Ownership",
                value=f"{metrics['account_ownership']['current']:.1f}%",
                delta=f"{metrics['account_ownership'].get('growth_rate', 0):.1f}% growth" if 'growth_rate' in metrics['account_ownership'] else None
            )
    
    with col2:
        if 'digital_payment' in metrics:
            st.metric(
                label="Digital Payment Usage",
                value=f"{metrics['digital_payment']['current']:.1f}%"
            )
    
    with col3:
        # P2P/ATM Ratio (placeholder)
        p2p_atm_ratio = loader.calculate_p2p_atm_ratio()
        if p2p_atm_ratio:
            st.metric(
                label="P2P/ATM Ratio",
                value=f"{p2p_atm_ratio:.2f}"
            )
        else:
            st.metric(
                label="P2P/ATM Ratio",
                value="N/A",
                help="Data not available"
            )
    
    with col4:
        total_events = len(loader.events_df) if loader.events_df is not None else 0
        st.metric(
            label="Total Events",
            value=total_events
        )
    
    st.markdown("---")
    
    # Growth rate highlights
    st.subheader("Growth Rate Highlights")
    
    if 'account_ownership' in metrics and 'growth_rate' in metrics['account_ownership']:
        growth = metrics['account_ownership']['growth_rate']
        st.info(f"Account ownership has grown by {growth:.1f}% over the observed period.")
    
    # Quick insights
    st.subheader("Quick Insights")
    
    if loader.observations_df is not None:
        total_obs = len(loader.observations_df)
        access_obs = len(loader.observations_df[loader.observations_df['pillar'] == 'access'])
        usage_obs = len(loader.observations_df[loader.observations_df['pillar'] == 'usage'])
        
        st.write(f"- **Total Observations**: {total_obs}")
        st.write(f"- **Access Observations**: {access_obs}")
        st.write(f"- **Usage Observations**: {usage_obs}")
    
    # Data download
    st.markdown("---")
    st.subheader("Download Data")
    
    if loader.observations_df is not None:
        csv = loader.observations_df.to_csv(index=False)
        st.download_button(
            label="Download Observations (CSV)",
            data=csv,
            file_name="observations.csv",
            mime="text/csv"
        )

# ============================================================================
# TRENDS PAGE
# ============================================================================
elif page == "Trends":
    st.title("📉 Trends")
    st.markdown("### Interactive Time Series Analysis")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        pillar_filter = st.selectbox(
            "Select Pillar",
            ["All", "Access", "Usage", "Quality"],
            index=0
        )
    
    with col2:
        show_events = st.checkbox("Show Events on Timeline", value=True)
    
    # Date range selector
    if loader.observations_df is not None:
        date_col = [col for col in loader.observations_df.columns if 'date' in col.lower()][0] if \
            [col for col in loader.observations_df.columns if 'date' in col.lower()] else None
        
        if date_col:
            dates = pd.to_datetime(loader.observations_df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                min_date = dates.min().date()
                max_date = dates.max().date()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
    
    # Filter data
    filtered_data = loader.observations_df.copy() if loader.observations_df is not None else pd.DataFrame()
    
    if len(filtered_data) > 0 and pillar_filter != "All":
        filtered_data = filtered_data[filtered_data['pillar'].str.lower() == pillar_filter.lower()]
    
    # Account Ownership Trend
    st.subheader("Account Ownership Trend")
    
    account_data = filtered_data[
        filtered_data['indicator_code'].str.contains('account|ownership', case=False, na=False)
    ].copy()
    
    if len(account_data) > 0 and date_col:
        account_data = account_data.sort_values(date_col)
        account_data['year'] = pd.to_datetime(account_data[date_col], errors='coerce').dt.year
        
        fig = viz.plot_time_series(
            data=account_data,
            x_col='year',
            y_col='value_numeric',
            title="Account Ownership Over Time",
            show_events=show_events,
            events_df=loader.events_df
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No account ownership data available for the selected filters.")
    
    # Digital Payment Trend
    st.subheader("Digital Payment Usage Trend")
    
    payment_data = filtered_data[
        filtered_data['indicator_code'].str.contains('digital|payment', case=False, na=False)
    ].copy()
    
    if len(payment_data) > 0 and date_col:
        payment_data = payment_data.sort_values(date_col)
        payment_data['year'] = pd.to_datetime(payment_data[date_col], errors='coerce').dt.year
        
        fig = viz.plot_time_series(
            data=payment_data,
            x_col='year',
            y_col='value_numeric',
            title="Digital Payment Usage Over Time",
            color='green',
            show_events=show_events,
            events_df=loader.events_df
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No digital payment data available for the selected filters.")
    
    # Channel Comparison
    st.subheader("Channel Comparison")
    
    channels = st.multiselect(
        "Select Channels to Compare",
        options=["Mobile Money", "Bank Account", "ATM", "Digital Payment"],
        default=["Mobile Money", "Bank Account"]
    )
    
    if channels and len(filtered_data) > 0:
        fig = viz.plot_channel_comparison(
            data=filtered_data,
            channels=channels,
            title="Channel Comparison Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download filtered data
    st.markdown("---")
    if len(filtered_data) > 0:
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_trends.csv",
            mime="text/csv"
        )

# ============================================================================
# FORECASTS PAGE
# ============================================================================
elif page == "Forecasts":
    st.title("🔮 Forecasts")
    st.markdown("### Financial Inclusion Forecasts (2025-2027)")
    
    # Model selection
    model_option = st.selectbox(
        "Select Forecast Model",
        ["Linear Trend", "Event-Augmented", "All Models"],
        index=1
    )
    
    if loader.forecasts_df is not None and len(loader.forecasts_df) > 0:
        # Filter by model
        forecast_data = loader.forecasts_df.copy()
        
        if model_option != "All Models":
            forecast_data = forecast_data[forecast_data['Method'].str.contains(model_option, case=False, na=False)]
        
        # Get unique indicators
        indicators = forecast_data['Indicator'].unique() if 'Indicator' in forecast_data.columns else []
        
        selected_indicator = st.selectbox(
            "Select Indicator",
            indicators,
            index=0 if len(indicators) > 0 else None
        )
        
        if selected_indicator:
            indicator_forecasts = forecast_data[forecast_data['Indicator'] == selected_indicator]
            
            # Historical data
            historical = loader.observations_df[
                (loader.observations_df['indicator'] == selected_indicator) |
                (loader.observations_df['indicator_code'].str.contains(selected_indicator.split()[0], case=False, na=False))
            ].copy()
            
            if len(historical) > 0:
                date_col = [col for col in historical.columns if 'date' in col.lower()][0] if \
                    [col for col in historical.columns if 'date' in col.lower()] else None
                
                if date_col:
                    historical['year'] = pd.to_datetime(historical[date_col], errors='coerce').dt.year
                    historical = historical.sort_values('year')
                    historical_summary = historical.groupby('year')['value_numeric'].mean().reset_index()
                    historical_summary.columns = ['year', 'value']
                else:
                    historical_summary = pd.DataFrame()
            else:
                historical_summary = pd.DataFrame()
            
            # Forecast visualization
            forecast_summary = indicator_forecasts[indicator_forecasts['Scenario'] == 'base'].copy()
            
            if len(forecast_summary) > 0:
                fig = viz.plot_forecast_with_ci(
                    historical=historical_summary,
                    forecast=forecast_summary,
                    title=f"{selected_indicator} Forecast"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.subheader("Forecast Table")
                st.dataframe(
                    forecast_summary[['Year', 'Forecast', 'CI_Lower', 'CI_Upper']].style.format({
                        'Forecast': '{:.2f}',
                        'CI_Lower': '{:.2f}',
                        'CI_Upper': '{:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Key milestones
                st.subheader("Key Projected Milestones")
                
                for _, row in forecast_summary.iterrows():
                    st.write(f"**{int(row['Year'])}**: {row['Forecast']:.1f}% "
                           f"(95% CI: {row['CI_Lower']:.1f}% - {row['CI_Upper']:.1f}%)")
            
            # Download forecasts
            st.markdown("---")
            csv = indicator_forecasts.to_csv(index=False)
            st.download_button(
                label="Download Forecasts (CSV)",
                data=csv,
                file_name=f"{selected_indicator.replace(' ', '_')}_forecasts.csv",
                mime="text/csv"
            )
    else:
        st.warning("Forecast data not available. Please run the forecasting notebook first.")
        st.info("To generate forecasts, run: `notebooks/05_forecasting.ipynb`")

# ============================================================================
# INCLUSION PROJECTIONS PAGE
# ============================================================================
elif page == "Inclusion Projections":
    st.title("🎯 Inclusion Projections")
    st.markdown("### Progress Toward Financial Inclusion Targets")
    
    # Target setting
    target_rate = st.slider(
        "Target Financial Inclusion Rate (%)",
        min_value=40,
        max_value=80,
        value=60,
        step=5
    )
    
    # Scenario selector
    scenario = st.radio(
        "Select Scenario",
        ["Optimistic", "Base", "Pessimistic"],
        index=1,
        horizontal=True
    )
    
    if loader.forecasts_df is not None and len(loader.forecasts_df) > 0:
        # Get account ownership forecasts
        account_forecasts = loader.forecasts_df[
            (loader.forecasts_df['Indicator'].str.contains('Account Ownership', case=False, na=False)) &
            (loader.forecasts_df['Scenario'].str.lower() == scenario.lower())
        ].copy()
        
        if len(account_forecasts) > 0:
            # Current value
            metrics = loader.get_key_metrics()
            current_value = metrics.get('account_ownership', {}).get('current', 0)
            
            # Progress visualization
            fig = viz.plot_progress_to_target(
                current=current_value,
                target=target_rate,
                forecast=account_forecasts,
                title=f"Progress Toward {target_rate}% Target ({scenario} Scenario)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Projection details
            st.subheader("Projection Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Current Rate",
                    value=f"{current_value:.1f}%"
                )
            
            with col2:
                latest_forecast = account_forecasts['Forecast'].iloc[-1] if len(account_forecasts) > 0 else 0
                st.metric(
                    label=f"Projected 2027 ({scenario})",
                    value=f"{latest_forecast:.1f}%"
                )
            
            with col3:
                gap = target_rate - latest_forecast
                st.metric(
                    label="Gap to Target",
                    value=f"{gap:.1f}pp",
                    delta="On Track" if gap <= 0 else "Needs Improvement"
                )
            
            # Scenario comparison
            st.subheader("Scenario Comparison")
            
            scenarios_data = {}
            for scen in ["optimistic", "base", "pessimistic"]:
                scen_data = loader.forecasts_df[
                    (loader.forecasts_df['Indicator'].str.contains('Account Ownership', case=False, na=False)) &
                    (loader.forecasts_df['Scenario'].str.lower() == scen)
                ].copy()
                if len(scen_data) > 0:
                    scenarios_data[scen] = scen_data
            
            if len(scenarios_data) > 0:
                historical = loader.observations_df[
                    loader.observations_df['indicator_code'].str.contains('account|ownership', case=False, na=False)
                ].copy()
                
                if len(historical) > 0:
                    date_col = [col for col in historical.columns if 'date' in col.lower()][0] if \
                        [col for col in historical.columns if 'date' in col.lower()] else None
                    if date_col:
                        historical['year'] = pd.to_datetime(historical[date_col], errors='coerce').dt.year
                        historical = historical.sort_values('year')
                        historical_summary = historical.groupby('year')['value_numeric'].mean().reset_index()
                        historical_summary.columns = ['year', 'value']
                    else:
                        historical_summary = pd.DataFrame()
                else:
                    historical_summary = pd.DataFrame()
                
                fig = viz.plot_scenarios(
                    historical=historical_summary,
                    scenarios=scenarios_data,
                    title="Account Ownership: Scenario Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Key questions answers
            st.markdown("---")
            st.subheader("Answers to Key Questions")
            
            st.markdown("""
            **1. What factors appear to drive financial inclusion in Ethiopia?**
            - Mobile money infrastructure expansion
            - Policy interventions and regulatory changes
            - Product launches (Telebirr, M-Pesa)
            - Infrastructure development (4G, mobile penetration)
            
            **2. Why might account ownership have stagnated despite mobile money expansion?**
            - Registered vs. active account gap
            - Limited usage beyond registration
            - Financial literacy barriers
            - Trust and security concerns
            
            **3. What is the gender gap and how has it evolved?**
            - Gender gap data available in Trends page
            - Analysis shows persistent disparities requiring targeted interventions
            
            **4. What data gaps most limit the analysis?**
            - Sparse indicator coverage (<5 observations for some indicators)
            - Limited disaggregated data (gender, urban/rural)
            - Missing infrastructure time series data
            """)
        else:
            st.warning("Account ownership forecasts not available for the selected scenario.")
    else:
        st.warning("Forecast data not available. Please run the forecasting notebook first.")

# Footer
st.markdown("---")
st.markdown("**Ethiopia Financial Inclusion Forecasting Dashboard** | Built with Streamlit")
