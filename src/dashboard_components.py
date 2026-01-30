"""
Dashboard Components Module
OOP-based components for Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DashboardDataLoader:
    """Class for loading and preparing dashboard data."""
    
    def __init__(
        self,
        data_file: Path,
        processed_dir: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the DashboardDataLoader.
        
        Args:
            data_file: Path to the main data Excel file
            processed_dir: Path to processed data directory
            logger: Optional logger instance
        """
        self.data_file = Path(data_file)
        self.processed_dir = Path(processed_dir)
        self.logger = logger or self._setup_logger()
        
        self.data_df: Optional[pd.DataFrame] = None
        self.observations_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.forecasts_df: Optional[pd.DataFrame] = None
        self.association_matrix: Optional[pd.DataFrame] = None
        
        self.logger.info(f"Initialized DashboardDataLoader")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger if none provided."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load main datasets."""
        self.logger.info("Loading dashboard data...")
        
        try:
            self.data_df = pd.read_excel(self.data_file, sheet_name='data')
            self.observations_df = self.data_df[
                self.data_df['record_type'] == 'observation'
            ].copy()
            self.events_df = self.data_df[
                self.data_df['record_type'] == 'event'
            ].copy()
            
            # Convert dates
            date_cols = [col for col in self.observations_df.columns if 'date' in col.lower()]
            for col in date_cols:
                if self.observations_df[col].dtype == 'object':
                    self.observations_df[col] = pd.to_datetime(self.observations_df[col], errors='coerce')
            
            self.logger.info(f"Loaded {len(self.observations_df)} observations, {len(self.events_df)} events")
            return self.data_df, self.observations_df, self.events_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def load_forecasts(self) -> Optional[pd.DataFrame]:
        """Load forecast data if available."""
        forecast_file = self.processed_dir / 'forecasts.csv'
        
        if forecast_file.exists():
            try:
                self.forecasts_df = pd.read_csv(forecast_file)
                self.logger.info(f"Loaded {len(self.forecasts_df)} forecast records")
                return self.forecasts_df
            except Exception as e:
                self.logger.warning(f"Could not load forecasts: {e}")
                return None
        return None
    
    def load_association_matrix(self) -> Optional[pd.DataFrame]:
        """Load association matrix if available."""
        matrix_file = self.processed_dir / 'association_matrix.csv'
        
        if matrix_file.exists():
            try:
                self.association_matrix = pd.read_csv(matrix_file, index_col=0)
                self.logger.info(f"Loaded association matrix: {self.association_matrix.shape}")
                return self.association_matrix
            except Exception as e:
                self.logger.warning(f"Could not load association matrix: {e}")
                return None
        return None
    
    def get_key_metrics(self) -> Dict[str, Any]:
        """Get key metrics for overview page."""
        if self.observations_df is None:
            self.load_data()
        
        metrics = {}
        
        # Get latest account ownership
        account_data = self.observations_df[
            (self.observations_df['pillar'] == 'access') &
            (self.observations_df['indicator_code'].str.contains('account|ownership', case=False, na=False))
        ].copy()
        
        if len(account_data) > 0:
            date_col = [col for col in account_data.columns if 'date' in col.lower()][0] if \
                [col for col in account_data.columns if 'date' in col.lower()] else None
            if date_col:
                account_data = account_data.sort_values(date_col)
                metrics['account_ownership'] = {
                    'current': float(account_data['value_numeric'].iloc[-1]) if 'value_numeric' in account_data.columns else 0,
                    'latest_year': pd.to_datetime(account_data[date_col].iloc[-1]).year if date_col else None
                }
        
        # Get latest digital payment usage
        usage_data = self.observations_df[
            (self.observations_df['pillar'] == 'usage') &
            (self.observations_df['indicator_code'].str.contains('digital|payment', case=False, na=False))
        ].copy()
        
        if len(usage_data) > 0:
            date_col = [col for col in usage_data.columns if 'date' in col.lower()][0] if \
                [col for col in usage_data.columns if 'date' in col.lower()] else None
            if date_col:
                usage_data = usage_data.sort_values(date_col)
                metrics['digital_payment'] = {
                    'current': float(usage_data['value_numeric'].iloc[-1]) if 'value_numeric' in usage_data.columns else 0,
                    'latest_year': pd.to_datetime(usage_data[date_col].iloc[-1]).year if date_col else None
                }
        
        # Calculate growth rates
        if 'account_ownership' in metrics and len(account_data) >= 2:
            values = account_data['value_numeric'].values
            if len(values) >= 2:
                growth = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                metrics['account_ownership']['growth_rate'] = growth
        
        return metrics
    
    def calculate_p2p_atm_ratio(self) -> Optional[float]:
        """Calculate P2P/ATM crossover ratio if data available."""
        # This would require specific P2P and ATM data
        # Placeholder implementation
        return None


class DashboardVisualizations:
    """Class for creating dashboard visualizations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize DashboardVisualizations."""
        self.logger = logger or self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def plot_time_series(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str,
        y_label: str = "Value (%)",
        color: str = None,
        show_events: bool = False,
        events_df: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create interactive time series plot."""
        fig = go.Figure()
        
        # Main time series
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            mode='lines+markers',
            name='Trend',
            line=dict(color=color or 'blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add events if requested
        if show_events and events_df is not None and len(events_df) > 0:
            event_date_col = [col for col in events_df.columns if 'date' in col.lower()][0] if \
                [col for col in events_df.columns if 'date' in col.lower()] else None
            
            if event_date_col:
                for _, event in events_df.iterrows():
                    event_date = pd.to_datetime(event[event_date_col], errors='coerce')
                    if pd.notna(event_date):
                        fig.add_vline(
                            x=event_date,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=event.get('category', 'Event'),
                            annotation_position="top"
                        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title=y_label,
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_forecast_with_ci(
        self,
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        title: str,
        x_col: str = 'year',
        y_col: str = 'value'
    ) -> go.Figure:
        """Create forecast plot with confidence intervals."""
        fig = go.Figure()
        
        # Historical data
        if len(historical) > 0:
            fig.add_trace(go.Scatter(
                x=historical[x_col],
                y=historical[y_col],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
        
        # Forecast
        if len(forecast) > 0:
            fig.add_trace(go.Scatter(
                x=forecast[x_col],
                y=forecast['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            # Confidence intervals
            if 'CI_Lower' in forecast.columns and 'CI_Upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=forecast[x_col].tolist() + forecast[x_col].tolist()[::-1],
                    y=forecast['CI_Upper'].tolist() + forecast['CI_Lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Value (%)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_scenarios(
        self,
        historical: pd.DataFrame,
        scenarios: Dict[str, pd.DataFrame],
        title: str
    ) -> go.Figure:
        """Create scenario comparison plot."""
        fig = go.Figure()
        
        # Historical
        if len(historical) > 0:
            fig.add_trace(go.Scatter(
                x=historical['year'],
                y=historical['value'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
        
        # Scenarios
        colors = {'optimistic': 'green', 'base': 'orange', 'pessimistic': 'red'}
        for scenario_name, data in scenarios.items():
            if len(data) > 0:
                fig.add_trace(go.Scatter(
                    x=data['year'],
                    y=data['Forecast'],
                    mode='lines+markers',
                    name=scenario_name.title(),
                    line=dict(color=colors.get(scenario_name, 'gray'), width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Value (%)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_progress_to_target(
        self,
        current: float,
        target: float,
        forecast: pd.DataFrame,
        title: str = "Progress Toward 60% Target"
    ) -> go.Figure:
        """Create progress visualization toward target."""
        fig = go.Figure()
        
        # Target line
        if len(forecast) > 0:
            years = forecast['Year'].unique()
            fig.add_hline(
                y=target,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target: {target}%",
                annotation_position="right"
            )
            
            # Current value
            fig.add_trace(go.Scatter(
                x=[years[0] - 1],
                y=[current],
                mode='markers',
                name='Current',
                marker=dict(size=12, color='blue')
            ))
            
            # Forecast trajectory
            fig.add_trace(go.Scatter(
                x=forecast['Year'],
                y=forecast['Forecast'],
                mode='lines+markers',
                name='Projected',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Value (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def plot_channel_comparison(
        self,
        data: pd.DataFrame,
        channels: List[str],
        title: str = "Channel Comparison"
    ) -> go.Figure:
        """Create channel comparison plot."""
        fig = go.Figure()
        
        for channel in channels:
            channel_data = data[data['indicator_code'].str.contains(channel, case=False, na=False)]
            if len(channel_data) > 0:
                date_col = [col for col in channel_data.columns if 'date' in col.lower()][0] if \
                    [col for col in channel_data.columns if 'date' in col.lower()] else None
                
                if date_col:
                    channel_data = channel_data.sort_values(date_col)
                    fig.add_trace(go.Scatter(
                        x=channel_data[date_col],
                        y=channel_data['value_numeric'],
                        mode='lines+markers',
                        name=channel,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
