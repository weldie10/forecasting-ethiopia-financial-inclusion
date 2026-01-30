"""
Task 3: Event Impact Modeling Module
OOP-based solution for modeling how events affect financial inclusion indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ImpactDirection(Enum):
    """Direction of impact."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EffectType(Enum):
    """Type of effect over time."""
    IMMEDIATE = "immediate"  # Effect happens immediately
    GRADUAL = "gradual"  # Effect builds gradually
    DELAYED = "delayed"  # Effect happens after lag period


@dataclass
class ImpactEstimate:
    """Data class for impact estimates."""
    event_id: str
    indicator_code: str
    direction: str
    magnitude: float
    lag_months: int
    effect_type: str
    confidence: str
    source: str
    notes: str = ""


@dataclass
class EventImpactModel:
    """Container for event impact model results."""
    association_matrix: pd.DataFrame = None
    impact_estimates: List[ImpactEstimate] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    methodology: str = ""
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


class EventImpactModeler:
    """Class for modeling event impacts on financial inclusion indicators."""
    
    def __init__(
        self,
        data_file: Path,
        logger: Optional[logging.Logger] = None,
        figure_dir: Optional[Path] = None
    ):
        """
        Initialize the EventImpactModeler.
        
        Args:
            data_file: Path to the data Excel file
            logger: Optional logger instance
            figure_dir: Directory to save figures
        """
        self.data_file = Path(data_file)
        self.logger = logger or self._setup_logger()
        self.figure_dir = Path(figure_dir) if figure_dir else Path('../reports/figures')
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.data_df: Optional[pd.DataFrame] = None
        self.impact_links_df: Optional[pd.DataFrame] = None
        self.events_df: Optional[pd.DataFrame] = None
        self.joined_data: Optional[pd.DataFrame] = None
        
        # Model container
        self.model = EventImpactModel()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.logger.info(f"Initialized EventImpactModeler with data_file: {data_file}")
    
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
        """
        Load datasets.
        
        Returns:
            Tuple of (data_df, impact_links_df, events_df)
        """
        self.logger.info("Loading datasets...")
        
        try:
            self.data_df = pd.read_excel(self.data_file, sheet_name='data')
            self.impact_links_df = pd.read_excel(self.data_file, sheet_name='impact_links')
            
            # Extract events
            self.events_df = self.data_df[self.data_df['record_type'] == 'event'].copy()
            
            self.logger.info(f"Loaded {len(self.data_df)} records, "
                           f"{len(self.impact_links_df)} impact links, "
                           f"{len(self.events_df)} events")
            
            return self.data_df, self.impact_links_df, self.events_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def join_impact_with_events(self) -> pd.DataFrame:
        """
        Join impact_links with events using parent_id.
        
        Returns:
            Joined DataFrame with event details
        """
        if self.impact_links_df is None or self.events_df is None:
            self.load_data()
        
        self.logger.info("Joining impact links with events...")
        
        # Join on parent_id (impact_links) = id (events)
        if 'parent_id' in self.impact_links_df.columns and 'id' in self.events_df.columns:
            self.joined_data = self.impact_links_df.merge(
                self.events_df,
                left_on='parent_id',
                right_on='id',
                how='left',
                suffixes=('_link', '_event')
            )
            
            self.logger.info(f"Joined {len(self.joined_data)} impact links with events")
        else:
            self.logger.warning("Cannot join: missing parent_id or id columns")
            self.joined_data = self.impact_links_df.copy()
        
        return self.joined_data
    
    def summarize_impacts(self) -> Dict[str, Any]:
        """
        Create summary showing which events affect which indicators and by how much.
        
        Returns:
            Dictionary with impact summary
        """
        if self.joined_data is None:
            self.join_impact_with_events()
        
        self.logger.info("Creating impact summary...")
        
        summary = {
            'total_links': len(self.joined_data),
            'unique_events': self.joined_data['parent_id'].nunique() if 'parent_id' in self.joined_data.columns else 0,
            'unique_indicators': self.joined_data['related_indicator'].nunique() if 'related_indicator' in self.joined_data.columns else 0,
            'by_direction': {},
            'by_pillar': {},
            'events_affecting_indicators': {}
        }
        
        if 'impact_direction' in self.joined_data.columns:
            summary['by_direction'] = self.joined_data['impact_direction'].value_counts().to_dict()
        
        if 'pillar' in self.joined_data.columns:
            summary['by_pillar'] = self.joined_data['pillar'].value_counts().to_dict()
        
        # Group by event and indicator
        if 'parent_id' in self.joined_data.columns and 'related_indicator' in self.joined_data.columns:
            grouped = self.joined_data.groupby(['parent_id', 'related_indicator']).agg({
                'impact_magnitude': 'mean',
                'impact_direction': 'first',
                'lag_months': 'mean'
            }).reset_index()
            
            summary['events_affecting_indicators'] = grouped.to_dict('records')
        
        self.logger.info(f"Impact summary: {summary['unique_events']} events affecting "
                        f"{summary['unique_indicators']} indicators")
        
        return summary
    
    def build_association_matrix(self, key_indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Build event-indicator association matrix.
        
        Args:
            key_indicators: List of key indicator codes to include. If None, uses all.
        
        Returns:
            Association matrix DataFrame (events x indicators)
        """
        if self.joined_data is None:
            self.join_impact_with_events()
        
        self.logger.info("Building event-indicator association matrix...")
        
        if 'parent_id' not in self.joined_data.columns or 'related_indicator' not in self.joined_data.columns:
            self.logger.error("Missing required columns for association matrix")
            return pd.DataFrame()
        
        # Get unique events and indicators
        events = self.joined_data['parent_id'].unique()
        
        if key_indicators is None:
            indicators = self.joined_data['related_indicator'].unique()
        else:
            indicators = [ind for ind in key_indicators if ind in self.joined_data['related_indicator'].values]
        
        # Create matrix
        matrix = pd.DataFrame(index=events, columns=indicators, dtype=float)
        
        # Fill matrix with impact magnitudes
        for _, row in self.joined_data.iterrows():
            event_id = row['parent_id']
            indicator = row['related_indicator']
            
            if event_id in matrix.index and indicator in matrix.columns:
                magnitude = row.get('impact_magnitude', 0)
                direction = row.get('impact_direction', 'neutral')
                
                # Apply direction
                if direction == 'negative':
                    magnitude = -abs(magnitude) if magnitude else -1
                elif direction == 'positive':
                    magnitude = abs(magnitude) if magnitude else 1
                else:
                    magnitude = 0
                
                # If multiple links for same event-indicator pair, take mean
                if pd.notna(matrix.loc[event_id, indicator]):
                    matrix.loc[event_id, indicator] = (matrix.loc[event_id, indicator] + magnitude) / 2
                else:
                    matrix.loc[event_id, indicator] = magnitude
        
        # Fill NaN with 0 (no impact)
        matrix = matrix.fillna(0)
        
        self.model.association_matrix = matrix
        self.logger.info(f"Built association matrix: {len(matrix)} events x {len(matrix.columns)} indicators")
        
        return matrix
    
    def model_effect_over_time(
        self,
        event_date: datetime,
        magnitude: float,
        lag_months: int = 0,
        effect_type: str = "immediate",
        duration_months: int = 12
    ) -> pd.Series:
        """
        Model how an event's effect evolves over time.
        
        Args:
            event_date: Date of the event
            magnitude: Magnitude of the effect
            lag_months: Months before effect starts
            effect_type: Type of effect (immediate, gradual, delayed)
            duration_months: Duration of effect in months
        
        Returns:
            Series with effect values over time
        """
        # Generate time series
        start_date = event_date + timedelta(days=lag_months * 30)
        dates = pd.date_range(start=start_date, periods=duration_months, freq='M')
        
        if effect_type == "immediate":
            # Immediate effect, constant over time
            effects = pd.Series([magnitude] * duration_months, index=dates)
        
        elif effect_type == "gradual":
            # Gradual build-up to full magnitude
            effects = pd.Series(
                [magnitude * (i / duration_months) for i in range(1, duration_months + 1)],
                index=dates
            )
        
        elif effect_type == "delayed":
            # No effect until lag period, then immediate
            effects = pd.Series([0] * lag_months + [magnitude] * (duration_months - lag_months), index=dates)
        
        else:
            # Default: immediate
            effects = pd.Series([magnitude] * duration_months, index=dates)
        
        return effects
    
    def combine_event_effects(
        self,
        event_effects: List[pd.Series],
        method: str = "additive"
    ) -> pd.Series:
        """
        Combine effects from multiple events.
        
        Args:
            event_effects: List of effect Series
            method: Combination method (additive, multiplicative, max)
        
        Returns:
            Combined effect Series
        """
        if not event_effects:
            return pd.Series()
        
        # Align all series to same date range
        all_dates = set()
        for series in event_effects:
            all_dates.update(series.index)
        
        all_dates = sorted(all_dates)
        combined = pd.Series(index=all_dates, dtype=float).fillna(0)
        
        for series in event_effects:
            for date in series.index:
                if date in combined.index:
                    if method == "additive":
                        combined[date] += series[date]
                    elif method == "multiplicative":
                        combined[date] = combined[date] * (1 + series[date]) if combined[date] != 0 else series[date]
                    elif method == "max":
                        combined[date] = max(combined[date], series[date])
        
        return combined
    
    def validate_against_historical(
        self,
        event_name: str,
        indicator_code: str,
        observed_before: float,
        observed_after: float,
        event_date: datetime
    ) -> Dict[str, Any]:
        """
        Validate model predictions against historical data.
        
        Args:
            event_name: Name of the event
            indicator_code: Indicator code
            observed_before: Observed value before event
            observed_after: Observed value after event
            event_date: Date of event
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating {event_name} impact on {indicator_code}...")
        
        # Get predicted impact from model
        predicted_impact = None
        
        if self.model.association_matrix is not None:
            # Find event in matrix
            event_id = None
            if 'id' in self.events_df.columns:
                event_row = self.events_df[
                    self.events_df['category'].str.contains(event_name, case=False, na=False) |
                    self.events_df['id'].astype(str).str.contains(event_name, case=False, na=False)
                ]
                if len(event_row) > 0:
                    event_id = event_row.iloc[0]['id']
            
            if event_id and event_id in self.model.association_matrix.index:
                if indicator_code in self.model.association_matrix.columns:
                    predicted_impact = self.model.association_matrix.loc[event_id, indicator_code]
        
        # Calculate observed change
        observed_change = observed_after - observed_before
        observed_change_pct = ((observed_after - observed_before) / observed_before * 100) if observed_before != 0 else 0
        
        validation = {
            'event_name': event_name,
            'indicator_code': indicator_code,
            'event_date': event_date,
            'observed_before': observed_before,
            'observed_after': observed_after,
            'observed_change': observed_change,
            'observed_change_pct': observed_change_pct,
            'predicted_impact': predicted_impact,
            'difference': observed_change - predicted_impact if predicted_impact is not None else None,
            'alignment': 'good' if predicted_impact is not None and abs(observed_change - predicted_impact) < 2 else 'needs_review'
        }
        
        self.logger.info(f"Validation: Observed change {observed_change:.2f}, "
                        f"Predicted {predicted_impact:.2f if predicted_impact else 'N/A'}")
        
        return validation
    
    def visualize_association_matrix(self, save: bool = True) -> plt.Figure:
        """
        Visualize event-indicator association matrix as heatmap.
        
        Args:
            save: Whether to save the figure
        
        Returns:
            Matplotlib figure
        """
        if self.model.association_matrix is None:
            self.logger.warning("Association matrix not built. Building now...")
            self.build_association_matrix()
        
        matrix = self.model.association_matrix
        
        if len(matrix) == 0:
            self.logger.warning("Empty association matrix")
            return None
        
        fig, ax = plt.subplots(figsize=(max(12, len(matrix.columns) * 0.8), 
                                       max(8, len(matrix) * 0.5)))
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=False,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Event-Indicator Association Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Indicators', fontsize=12)
        ax.set_ylabel('Events', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.figure_dir / 'association_matrix.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved association matrix to {self.figure_dir / 'association_matrix.png'}")
        
        return fig
    
    def document_methodology(self) -> str:
        """
        Document the impact modeling methodology.
        
        Returns:
            Methodology documentation string
        """
        methodology = """
# Event Impact Modeling Methodology

## Approach

1. **Data Integration**: Join impact_links with events using parent_id to get event details
2. **Association Matrix**: Build matrix showing which events affect which indicators
3. **Effect Modeling**: Model how effects evolve over time (immediate, gradual, delayed)
4. **Combination**: Combine effects from multiple events using additive method
5. **Validation**: Test predictions against historical observed data

## Functional Forms

### Immediate Effect
Effect happens immediately at event date (or after lag period) and remains constant.

### Gradual Effect
Effect builds gradually over time, reaching full magnitude at end of duration period.

### Delayed Effect
No effect until lag period completes, then immediate full effect.

## Combination Methods

- **Additive**: Sum of all event effects (default)
- **Multiplicative**: Compound effects
- **Max**: Take maximum effect

## Assumptions

1. Effects are linear within the modeled period
2. Events are independent (no interaction effects modeled)
3. Lag periods are fixed as specified in impact_links
4. Magnitudes are point estimates (uncertainty not modeled)

## Limitations

1. Limited historical data for validation
2. No interaction effects between events
3. Uncertainty in impact estimates not quantified
4. Comparable country evidence may not directly apply to Ethiopia
5. Time-varying effects not fully captured
"""
        
        self.model.methodology = methodology
        return methodology
    
    def add_assumptions(self, assumptions: List[str]) -> None:
        """Add assumptions to the model."""
        self.model.assumptions.extend(assumptions)
        self.logger.info(f"Added {len(assumptions)} assumptions")
    
    def add_limitations(self, limitations: List[str]) -> None:
        """Add limitations to the model."""
        self.model.limitations.extend(limitations)
        self.logger.info(f"Added {len(limitations)} limitations")
    
    def refine_estimates(
        self,
        event_id: str,
        indicator_code: str,
        new_magnitude: float,
        reason: str
    ) -> None:
        """
        Refine impact estimates based on validation or new evidence.
        
        Args:
            event_id: Event ID
            indicator_code: Indicator code
            new_magnitude: New magnitude estimate
            reason: Reason for refinement
        """
        self.logger.info(f"Refining estimate: {event_id} -> {indicator_code} = {new_magnitude}")
        
        if self.model.association_matrix is not None:
            if event_id in self.model.association_matrix.index and indicator_code in self.model.association_matrix.columns:
                old_magnitude = self.model.association_matrix.loc[event_id, indicator_code]
                self.model.association_matrix.loc[event_id, indicator_code] = new_magnitude
                
                self.logger.info(f"Updated from {old_magnitude:.2f} to {new_magnitude:.2f}. Reason: {reason}")
            else:
                self.logger.warning(f"Event {event_id} or indicator {indicator_code} not in matrix")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the impact model."""
        summary = {
            'association_matrix_shape': self.model.association_matrix.shape if self.model.association_matrix is not None else (0, 0),
            'total_impact_estimates': len(self.model.impact_estimates),
            'validation_results_count': len(self.model.validation_results),
            'assumptions_count': len(self.model.assumptions),
            'limitations_count': len(self.model.limitations)
        }
        
        return summary
