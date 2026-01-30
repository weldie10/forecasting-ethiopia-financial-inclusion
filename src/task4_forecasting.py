"""
Task 4: Forecasting Access and Usage Module
OOP-based solution for forecasting financial inclusion indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-learn for regression
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using basic regression")


class ForecastMethod(Enum):
    """Forecasting methods."""
    LINEAR_TREND = "linear_trend"
    LOG_TREND = "log_trend"
    EVENT_AUGMENTED = "event_augmented"
    SCENARIO = "scenario"


class Scenario(Enum):
    """Forecast scenarios."""
    OPTIMISTIC = "optimistic"
    BASE = "base"
    PESSIMISTIC = "pessimistic"


@dataclass
class ForecastResult:
    """Container for forecast results."""
    indicator_code: str
    indicator_name: str
    method: str
    forecast_years: List[int]
    forecast_values: List[float]
    confidence_intervals_lower: List[float] = field(default_factory=list)
    confidence_intervals_upper: List[float] = field(default_factory=list)
    scenario: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    r2_score: Optional[float] = None
    rmse: Optional[float] = None


@dataclass
class ForecastingModel:
    """Container for forecasting model results."""
    forecasts: List[ForecastResult] = field(default_factory=list)
    historical_data: pd.DataFrame = None
    event_impacts: Dict[str, Any] = field(default_factory=dict)
    uncertainties: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)


class FinancialInclusionForecaster:
    """Class for forecasting financial inclusion indicators."""
    
    def __init__(
        self,
        data_file: Path,
        impact_model_file: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        figure_dir: Optional[Path] = None
    ):
        """
        Initialize the FinancialInclusionForecaster.
        
        Args:
            data_file: Path to the data Excel file
            impact_model_file: Optional path to impact model/association matrix
            logger: Optional logger instance
            figure_dir: Directory to save figures
        """
        self.data_file = Path(data_file)
        self.impact_model_file = impact_model_file
        self.logger = logger or self._setup_logger()
        self.figure_dir = Path(figure_dir) if figure_dir else Path('../reports/figures')
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.data_df: Optional[pd.DataFrame] = None
        self.observations_df: Optional[pd.DataFrame] = None
        self.association_matrix: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        
        # Model container
        self.model = ForecastingModel()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.logger.info(f"Initialized FinancialInclusionForecaster with data_file: {data_file}")
    
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
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load datasets.
        
        Returns:
            Tuple of (data_df, observations_df)
        """
        self.logger.info("Loading datasets...")
        
        try:
            self.data_df = pd.read_excel(self.data_file, sheet_name='data')
            self.observations_df = self.data_df[
                self.data_df['record_type'] == 'observation'
            ].copy()
            
            # Load events
            self.events_df = self.data_df[
                self.data_df['record_type'] == 'event'
            ].copy()
            
            # Load association matrix if available
            if self.impact_model_file and Path(self.impact_model_file).exists():
                try:
                    self.association_matrix = pd.read_csv(self.impact_model_file, index_col=0)
                    self.logger.info(f"Loaded association matrix: {self.association_matrix.shape}")
                except:
                    self.logger.warning("Could not load association matrix")
            
            self.logger.info(f"Loaded {len(self.data_df)} records, "
                           f"{len(self.observations_df)} observations")
            
            return self.data_df, self.observations_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def extract_indicator_series(
        self,
        indicator_code: str,
        pillar: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract time series for a specific indicator.
        
        Args:
            indicator_code: Indicator code to extract
            pillar: Optional pillar filter
        
        Returns:
            DataFrame with year and value columns
        """
        if self.observations_df is None:
            self.load_data()
        
        self.logger.info(f"Extracting series for {indicator_code}...")
        
        # Filter by indicator
        indicator_data = self.observations_df[
            self.observations_df['indicator_code'] == indicator_code
        ].copy()
        
        if pillar:
            indicator_data = indicator_data[indicator_data['pillar'] == pillar]
        
        if len(indicator_data) == 0:
            self.logger.warning(f"No data found for {indicator_code}")
            return pd.DataFrame()
        
        # Extract date and value
        date_col = [col for col in indicator_data.columns if 'date' in col.lower()][0] if \
            [col for col in indicator_data.columns if 'date' in col.lower()] else None
        
        if date_col:
            indicator_data['year'] = pd.to_datetime(indicator_data[date_col], errors='coerce').dt.year
            indicator_data = indicator_data.sort_values('year')
        
        # Get indicator name
        indicator_name = indicator_data['indicator'].iloc[0] if 'indicator' in indicator_data.columns else indicator_code
        
        # Create time series
        series = indicator_data[['year', 'value_numeric']].dropna().copy()
        series = series.groupby('year')['value_numeric'].mean().reset_index()
        series.columns = ['year', 'value']
        series['indicator_code'] = indicator_code
        series['indicator_name'] = indicator_name
        
        self.logger.info(f"Extracted {len(series)} data points for {indicator_code}")
        
        return series
    
    def linear_trend_forecast(
        self,
        series: pd.DataFrame,
        forecast_years: List[int],
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Forecast using linear trend regression.
        
        Args:
            series: Time series DataFrame with year and value columns
            forecast_years: List of years to forecast
            confidence_level: Confidence level for intervals
        
        Returns:
            ForecastResult object
        """
        self.logger.info(f"Linear trend forecast for {len(forecast_years)} years...")
        
        if len(series) < 2:
            self.logger.error("Insufficient data for linear regression")
            return None
        
        X = series['year'].values.reshape(-1, 1)
        y = series['value'].values
        
        if SKLEARN_AVAILABLE:
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions
            forecast_X = np.array(forecast_years).reshape(-1, 1)
            forecast_values = model.predict(forecast_X)
            
            # Calculate confidence intervals (simplified)
            residuals = y - model.predict(X)
            rmse = np.sqrt(np.mean(residuals**2))
            std_error = rmse
            
            # Confidence intervals (approximate)
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
            ci_lower = forecast_values - z_score * std_error
            ci_upper = forecast_values + z_score * std_error
            
            r2 = r2_score(y, model.predict(X))
            rmse_val = np.sqrt(mean_squared_error(y, model.predict(X)))
            
            model_params = {
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r2': r2,
                'rmse': rmse_val
            }
        else:
            # Basic linear regression
            n = len(series)
            x_mean = series['year'].mean()
            y_mean = series['value'].mean()
            
            slope = ((series['year'] - x_mean) * (series['value'] - y_mean)).sum() / \
                   ((series['year'] - x_mean)**2).sum()
            intercept = y_mean - slope * x_mean
            
            forecast_values = [intercept + slope * year for year in forecast_years]
            
            # Simple confidence intervals
            residuals = series['value'] - (intercept + slope * series['year'])
            std_error = residuals.std()
            z_score = 1.96
            ci_lower = [v - z_score * std_error for v in forecast_values]
            ci_upper = [v + z_score * std_error for v in forecast_values]
            
            r2 = None
            rmse_val = None
            model_params = {'slope': slope, 'intercept': intercept}
        
        result = ForecastResult(
            indicator_code=series['indicator_code'].iloc[0],
            indicator_name=series['indicator_name'].iloc[0],
            method=ForecastMethod.LINEAR_TREND.value,
            forecast_years=forecast_years,
            forecast_values=forecast_values.tolist() if hasattr(forecast_values, 'tolist') else forecast_values,
            confidence_intervals_lower=ci_lower.tolist() if hasattr(ci_lower, 'tolist') else ci_lower,
            confidence_intervals_upper=ci_upper.tolist() if hasattr(ci_upper, 'tolist') else ci_upper,
            model_params=model_params,
            r2_score=r2,
            rmse=rmse_val
        )
        
        return result
    
    def log_trend_forecast(
        self,
        series: pd.DataFrame,
        forecast_years: List[int],
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Forecast using logarithmic trend regression.
        
        Args:
            series: Time series DataFrame with year and value columns
            forecast_years: List of years to forecast
            confidence_level: Confidence level for intervals
        
        Returns:
            ForecastResult object
        """
        self.logger.info(f"Log trend forecast for {len(forecast_years)} years...")
        
        if len(series) < 2:
            self.logger.error("Insufficient data for log regression")
            return None
        
        # Ensure positive values for log
        series_clean = series[series['value'] > 0].copy()
        if len(series_clean) < 2:
            self.logger.warning("Insufficient positive values, using linear instead")
            return self.linear_trend_forecast(series, forecast_years, confidence_level)
        
        # Log transform
        series_clean['log_value'] = np.log(series_clean['value'])
        
        X = series_clean['year'].values.reshape(-1, 1)
        y = series_clean['log_value'].values
        
        if SKLEARN_AVAILABLE:
            model = LinearRegression()
            model.fit(X, y)
            
            forecast_X = np.array(forecast_years).reshape(-1, 1)
            forecast_log = model.predict(forecast_X)
            forecast_values = np.exp(forecast_log)
            
            # Confidence intervals in log space
            residuals = y - model.predict(X)
            rmse = np.sqrt(np.mean(residuals**2))
            z_score = 1.96 if confidence_level == 0.95 else 2.576
            
            ci_lower_log = forecast_log - z_score * rmse
            ci_upper_log = forecast_log + z_score * rmse
            ci_lower = np.exp(ci_lower_log)
            ci_upper = np.exp(ci_upper_log)
            
            r2 = r2_score(y, model.predict(X))
            rmse_val = rmse
            
            model_params = {
                'slope': model.coef_[0],
                'intercept': model.intercept_,
                'r2': r2,
                'rmse': rmse_val
            }
        else:
            # Basic log regression
            n = len(series_clean)
            x_mean = series_clean['year'].mean()
            y_mean = series_clean['log_value'].mean()
            
            slope = ((series_clean['year'] - x_mean) * (series_clean['log_value'] - y_mean)).sum() / \
                   ((series_clean['year'] - x_mean)**2).sum()
            intercept = y_mean - slope * x_mean
            
            forecast_log = [intercept + slope * year for year in forecast_years]
            forecast_values = [np.exp(v) for v in forecast_log]
            
            residuals = series_clean['log_value'] - (intercept + slope * series_clean['year'])
            std_error = residuals.std()
            z_score = 1.96
            ci_lower = [np.exp(v - z_score * std_error) for v in forecast_log]
            ci_upper = [np.exp(v + z_score * std_error) for v in forecast_log]
            
            r2 = None
            rmse_val = None
            model_params = {'slope': slope, 'intercept': intercept}
        
        result = ForecastResult(
            indicator_code=series_clean['indicator_code'].iloc[0],
            indicator_name=series_clean['indicator_name'].iloc[0],
            method=ForecastMethod.LOG_TREND.value,
            forecast_years=forecast_years,
            forecast_values=forecast_values.tolist() if hasattr(forecast_values, 'tolist') else forecast_values,
            confidence_intervals_lower=ci_lower.tolist() if hasattr(ci_lower, 'tolist') else ci_lower,
            confidence_intervals_upper=ci_upper.tolist() if hasattr(ci_upper, 'tolist') else ci_upper,
            model_params=model_params,
            r2_score=r2,
            rmse=rmse_val
        )
        
        return result
    
    def event_augmented_forecast(
        self,
        series: pd.DataFrame,
        forecast_years: List[int],
        association_matrix: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Forecast using trend + event effects.
        
        Args:
            series: Time series DataFrame
            forecast_years: List of years to forecast
            association_matrix: Event-indicator association matrix
            confidence_level: Confidence level for intervals
        
        Returns:
            ForecastResult object
        """
        self.logger.info(f"Event-augmented forecast for {len(forecast_years)} years...")
        
        # Start with linear trend
        trend_forecast = self.linear_trend_forecast(series, forecast_years, confidence_level)
        
        if trend_forecast is None:
            return None
        
        # Add event effects if association matrix available
        if association_matrix is not None and len(association_matrix) > 0:
            indicator_code = series['indicator_code'].iloc[0]
            
            if indicator_code in association_matrix.columns:
                # Get events affecting this indicator
                event_impacts = association_matrix[indicator_code]
                event_impacts = event_impacts[event_impacts != 0]
                
                if len(event_impacts) > 0:
                    # Sum event impacts (simplified - assumes all events apply)
                    total_event_impact = event_impacts.sum()
                    
                    # Adjust forecast
                    adjusted_values = [v + total_event_impact for v in trend_forecast.forecast_values]
                    trend_forecast.forecast_values = adjusted_values
                    
                    # Adjust confidence intervals
                    trend_forecast.confidence_intervals_lower = [
                        v + total_event_impact * 0.5 for v in trend_forecast.confidence_intervals_lower
                    ]
                    trend_forecast.confidence_intervals_upper = [
                        v + total_event_impact * 1.5 for v in trend_forecast.confidence_intervals_upper
                    ]
                    
                    trend_forecast.method = ForecastMethod.EVENT_AUGMENTED.value
                    trend_forecast.model_params['event_impact'] = total_event_impact
                    trend_forecast.model_params['events_count'] = len(event_impacts)
        
        return trend_forecast
    
    def scenario_forecast(
        self,
        base_forecast: ForecastResult,
        optimistic_multiplier: float = 1.2,
        pessimistic_multiplier: float = 0.8
    ) -> Dict[str, ForecastResult]:
        """
        Generate scenario forecasts (optimistic, base, pessimistic).
        
        Args:
            base_forecast: Base forecast result
            optimistic_multiplier: Multiplier for optimistic scenario
            pessimistic_multiplier: Multiplier for pessimistic scenario
        
        Returns:
            Dictionary with scenario forecasts
        """
        self.logger.info("Generating scenario forecasts...")
        
        scenarios = {}
        
        # Base scenario
        scenarios['base'] = base_forecast
        
        # Optimistic scenario
        optimistic = ForecastResult(
            indicator_code=base_forecast.indicator_code,
            indicator_name=base_forecast.indicator_name,
            method=ForecastMethod.SCENARIO.value,
            forecast_years=base_forecast.forecast_years,
            forecast_values=[v * optimistic_multiplier for v in base_forecast.forecast_values],
            confidence_intervals_lower=[v * optimistic_multiplier for v in base_forecast.confidence_intervals_lower],
            confidence_intervals_upper=[v * optimistic_multiplier for v in base_forecast.confidence_intervals_upper],
            scenario=Scenario.OPTIMISTIC.value,
            model_params=base_forecast.model_params.copy()
        )
        scenarios['optimistic'] = optimistic
        
        # Pessimistic scenario
        pessimistic = ForecastResult(
            indicator_code=base_forecast.indicator_code,
            indicator_name=base_forecast.indicator_name,
            method=ForecastMethod.SCENARIO.value,
            forecast_years=base_forecast.forecast_years,
            forecast_values=[v * pessimistic_multiplier for v in base_forecast.forecast_values],
            confidence_intervals_lower=[v * pessimistic_multiplier for v in base_forecast.confidence_intervals_lower],
            confidence_intervals_upper=[v * pessimistic_multiplier for v in base_forecast.confidence_intervals_upper],
            scenario=Scenario.PESSIMISTIC.value,
            model_params=base_forecast.model_params.copy()
        )
        scenarios['pessimistic'] = pessimistic
        
        return scenarios
    
    def visualize_forecast(
        self,
        forecast: ForecastResult,
        historical_data: Optional[pd.DataFrame] = None,
        save: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize forecast with historical data and confidence intervals.
        
        Args:
            forecast: ForecastResult object
            historical_data: Optional historical data DataFrame
            save: Whether to save the figure
            title: Optional custom title
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        if historical_data is not None and len(historical_data) > 0:
            ax.plot(historical_data['year'], historical_data['value'],
                   marker='o', linewidth=2, markersize=8, label='Historical', color='blue')
        
        # Plot forecast
        ax.plot(forecast.forecast_years, forecast.forecast_values,
               marker='s', linewidth=2, markersize=8, label='Forecast', color='green')
        
        # Plot confidence intervals
        if forecast.confidence_intervals_lower and forecast.confidence_intervals_upper:
            ax.fill_between(
                forecast.forecast_years,
                forecast.confidence_intervals_lower,
                forecast.confidence_intervals_upper,
                alpha=0.3, color='green', label=f'95% Confidence Interval'
            )
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Value (%)', fontsize=12)
        ax.set_title(title or f'{forecast.indicator_name} Forecast', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            filename = f"{forecast.indicator_code.lower()}_forecast.png"
            fig.savefig(self.figure_dir / filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved forecast figure to {self.figure_dir / filename}")
        
        return fig
    
    def visualize_scenarios(
        self,
        scenarios: Dict[str, ForecastResult],
        historical_data: Optional[pd.DataFrame] = None,
        save: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize scenario forecasts.
        
        Args:
            scenarios: Dictionary of scenario forecasts
            historical_data: Optional historical data
            save: Whether to save the figure
            title: Optional custom title
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        if historical_data is not None and len(historical_data) > 0:
            ax.plot(historical_data['year'], historical_data['value'],
                   marker='o', linewidth=2, markersize=8, label='Historical', color='blue')
        
        # Plot scenarios
        colors = {'optimistic': 'green', 'base': 'orange', 'pessimistic': 'red'}
        for scenario_name, forecast in scenarios.items():
            ax.plot(forecast.forecast_years, forecast.forecast_values,
                   marker='s', linewidth=2, markersize=6, 
                   label=scenario_name.title(), color=colors.get(scenario_name, 'gray'))
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Value (%)', fontsize=12)
        ax.set_title(title or 'Scenario Forecasts', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            indicator_code = scenarios['base'].indicator_code
            filename = f"{indicator_code.lower()}_scenarios.png"
            fig.savefig(self.figure_dir / filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved scenario figure to {self.figure_dir / filename}")
        
        return fig
    
    def create_forecast_table(
        self,
        forecasts: List[ForecastResult],
        save: bool = True
    ) -> pd.DataFrame:
        """
        Create forecast table with confidence intervals.
        
        Args:
            forecasts: List of ForecastResult objects
            save: Whether to save to CSV
        
        Returns:
            DataFrame with forecast table
        """
        table_data = []
        
        for forecast in forecasts:
            for i, year in enumerate(forecast.forecast_years):
                row = {
                    'Indicator': forecast.indicator_name,
                    'Indicator_Code': forecast.indicator_code,
                    'Year': year,
                    'Forecast': forecast.forecast_values[i],
                    'CI_Lower': forecast.confidence_intervals_lower[i] if forecast.confidence_intervals_lower else None,
                    'CI_Upper': forecast.confidence_intervals_upper[i] if forecast.confidence_intervals_upper else None,
                    'Method': forecast.method,
                    'Scenario': forecast.scenario or 'base'
                }
                table_data.append(row)
        
        table_df = pd.DataFrame(table_data)
        
        if save:
            output_file = Path('../data/processed/forecasts.csv')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            table_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved forecast table to {output_file}")
        
        return table_df
