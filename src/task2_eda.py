"""
Task 2: Exploratory Data Analysis Module
Comprehensive OOP-based solution for analyzing financial inclusion data in Ethiopia.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Try to import Task1 classes if available
try:
    from task1_data_exploration import Task1DataProcessor, DataExplorer
except ImportError:
    Task1DataProcessor = None
    DataExplorer = None


@dataclass
class EDAResults:
    """Container for EDA results."""
    dataset_overview: Dict[str, Any] = None
    access_analysis: Dict[str, Any] = None
    usage_analysis: Dict[str, Any] = None
    infrastructure_analysis: Dict[str, Any] = None
    event_timeline: pd.DataFrame = None
    correlations: pd.DataFrame = None
    insights: List[str] = None
    data_quality: Dict[str, Any] = None


class ExploratoryDataAnalyzer:
    """Class for comprehensive exploratory data analysis."""
    
    def __init__(
        self,
        data_file: Path,
        reference_codes_file: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        figure_dir: Optional[Path] = None
    ):
        """
        Initialize the ExploratoryDataAnalyzer.
        
        Args:
            data_file: Path to the main data Excel file
            reference_codes_file: Optional path to reference codes file
            logger: Optional logger instance
            figure_dir: Directory to save figures
        """
        self.data_file = Path(data_file)
        self.reference_codes_file = reference_codes_file
        self.logger = logger or self._setup_logger()
        self.figure_dir = Path(figure_dir) if figure_dir else Path('../reports/figures')
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.data_df: Optional[pd.DataFrame] = None
        self.impact_links_df: Optional[pd.DataFrame] = None
        self.reference_codes_df: Optional[pd.DataFrame] = None
        
        # Results container
        self.results = EDAResults()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.logger.info(f"Initialized ExploratoryDataAnalyzer with data_file: {data_file}")
    
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
            Tuple of (data_df, impact_links_df)
        """
        self.logger.info("Loading datasets...")
        
        try:
            self.data_df = pd.read_excel(self.data_file, sheet_name='data')
            self.impact_links_df = pd.read_excel(self.data_file, sheet_name='impact_links')
            
            if self.reference_codes_file:
                self.reference_codes_df = pd.read_excel(self.reference_codes_file)
            
            self.logger.info(f"Loaded {len(self.data_df)} records and {len(self.impact_links_df)} impact links")
            
            # Convert date columns
            self._convert_dates()
            
            return self.data_df, self.impact_links_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def _convert_dates(self) -> None:
        """Convert date columns to datetime."""
        if self.data_df is None:
            return
        
        date_columns = [col for col in self.data_df.columns if 'date' in col.lower()]
        for col in date_columns:
            if self.data_df[col].dtype == 'object':
                self.data_df[col] = pd.to_datetime(self.data_df[col], errors='coerce')
                self.logger.debug(f"Converted {col} to datetime")
    
    def dataset_overview(self) -> Dict[str, Any]:
        """
        Summarize dataset by record_type, pillar, and source_type.
        
        Returns:
            Dictionary with overview statistics
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Generating dataset overview...")
        
        overview = {}
        
        # Summary by record_type
        if 'record_type' in self.data_df.columns:
            overview['by_record_type'] = self.data_df['record_type'].value_counts().to_dict()
        
        # Summary by pillar
        if 'pillar' in self.data_df.columns:
            overview['by_pillar'] = self.data_df['pillar'].value_counts().to_dict()
            overview['null_pillars'] = self.data_df['pillar'].isnull().sum()
        
        # Summary by source_type
        if 'source_type' in self.data_df.columns:
            overview['by_source_type'] = self.data_df['source_type'].value_counts().to_dict()
        
        # Confidence distribution
        if 'confidence' in self.data_df.columns:
            overview['confidence_distribution'] = self.data_df['confidence'].value_counts().to_dict()
        
        # Temporal coverage
        overview['temporal_coverage'] = self._get_temporal_coverage()
        
        # Data gaps
        overview['data_gaps'] = self._identify_data_gaps()
        
        self.results.dataset_overview = overview
        self.logger.info("Dataset overview complete")
        
        return overview
    
    def _get_temporal_coverage(self) -> Dict[str, Any]:
        """Get temporal coverage by indicator."""
        if self.data_df is None:
            return {}
        
        temporal_info = {}
        
        # Find date column
        date_col = None
        for col in self.data_df.columns:
            if 'date' in col.lower() and 'observation' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            date_cols = [col for col in self.data_df.columns if 'date' in col.lower()]
            date_col = date_cols[0] if date_cols else None
        
        if date_col and 'indicator_code' in self.data_df.columns:
            # Filter observations only
            observations = self.data_df[self.data_df['record_type'] == 'observation'].copy()
            
            if len(observations) > 0:
                # Group by indicator and get year range
                observations['year'] = pd.to_datetime(observations[date_col], errors='coerce').dt.year
                
                coverage = observations.groupby('indicator_code')['year'].agg([
                    'min', 'max', 'count', lambda x: sorted(x.unique().tolist())
                ]).reset_index()
                coverage.columns = ['indicator_code', 'min_year', 'max_year', 'count', 'years']
                
                temporal_info['by_indicator'] = coverage.to_dict('records')
                temporal_info['overall_min'] = observations['year'].min()
                temporal_info['overall_max'] = observations['year'].max()
        
        return temporal_info
    
    def _identify_data_gaps(self) -> Dict[str, Any]:
        """Identify indicators with sparse coverage."""
        if self.data_df is None:
            return {}
        
        gaps = {}
        
        if 'indicator_code' in self.data_df.columns:
            observations = self.data_df[self.data_df['record_type'] == 'observation'].copy()
            
            if len(observations) > 0:
                indicator_counts = observations['indicator_code'].value_counts()
                
                # Define sparse as less than 5 observations
                sparse_threshold = 5
                sparse_indicators = indicator_counts[indicator_counts < sparse_threshold]
                
                gaps['sparse_indicators'] = sparse_indicators.to_dict()
                gaps['sparse_count'] = len(sparse_indicators)
                gaps['total_indicators'] = len(indicator_counts)
                gaps['sparse_percentage'] = (len(sparse_indicators) / len(indicator_counts)) * 100
        
        return gaps
    
    def visualize_temporal_coverage(self, save: bool = True) -> plt.Figure:
        """
        Create temporal coverage visualization.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Creating temporal coverage visualization...")
        
        temporal_info = self._get_temporal_coverage()
        
        if 'by_indicator' not in temporal_info:
            self.logger.warning("No temporal coverage data available")
            return None
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(temporal_info['by_indicator']) * 0.3)))
        
        coverage_data = temporal_info['by_indicator']
        indicators = [item['indicator_code'] for item in coverage_data]
        min_years = [item['min_year'] for item in coverage_data]
        max_years = [item['max_year'] for item in coverage_data]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(indicators))
        ax.barh(y_pos, [max_y - min_y + 1 for min_y, max_y in zip(min_years, max_years)],
                left=min_years, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(indicators)
        ax.set_xlabel('Year')
        ax.set_title('Temporal Coverage by Indicator', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.figure_dir / 'temporal_coverage.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved temporal coverage figure to {self.figure_dir / 'temporal_coverage.png'}")
        
        return fig
    
    def analyze_access(self) -> Dict[str, Any]:
        """
        Analyze Access pillar data.
        
        Returns:
            Dictionary with access analysis results
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Analyzing Access pillar...")
        
        # Filter access observations
        access_data = self.data_df[
            (self.data_df['record_type'] == 'observation') &
            (self.data_df['pillar'] == 'access')
        ].copy()
        
        if len(access_data) == 0:
            self.logger.warning("No access data found")
            return {}
        
        analysis = {}
        
        # Account ownership trajectory
        account_ownership = self._extract_account_ownership(access_data)
        if account_ownership is not None and len(account_ownership) > 0:
            analysis['account_ownership'] = account_ownership
            analysis['growth_rates'] = self._calculate_growth_rates(account_ownership)
        
        # Gender gap analysis
        gender_analysis = self._analyze_gender_gap(access_data)
        if gender_analysis:
            analysis['gender_gap'] = gender_analysis
        
        # Urban vs Rural analysis
        urban_rural = self._analyze_urban_rural(access_data)
        if urban_rural:
            analysis['urban_rural'] = urban_rural
        
        # 2021-2024 slowdown analysis
        slowdown = self._analyze_slowdown(account_ownership)
        if slowdown:
            analysis['slowdown_analysis'] = slowdown
        
        self.results.access_analysis = analysis
        self.logger.info("Access analysis complete")
        
        return analysis
    
    def _extract_account_ownership(self, access_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract account ownership data."""
        # Look for account ownership indicator
        account_indicators = access_data[
            access_data['indicator_code'].str.contains('account|ownership', case=False, na=False)
        ].copy()
        
        if len(account_indicators) == 0:
            # Try to find any access indicator with value_numeric
            account_indicators = access_data[access_data['value_numeric'].notna()].copy()
        
        if len(account_indicators) == 0:
            return None
        
        # Get date column
        date_col = [col for col in account_indicators.columns if 'date' in col.lower()][0] if \
            [col for col in account_indicators.columns if 'date' in col.lower()] else None
        
        if date_col:
            account_indicators['year'] = pd.to_datetime(account_indicators[date_col], errors='coerce').dt.year
            account_indicators = account_indicators.sort_values('year')
        
        return account_indicators[['year', 'value_numeric', 'indicator_code', 'indicator']].copy() if \
            'year' in account_indicators.columns else account_indicators
    
    def _calculate_growth_rates(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate growth rates between survey years."""
        if 'year' not in data.columns or 'value_numeric' not in data.columns:
            return {}
        
        data = data.sort_values('year').dropna(subset=['year', 'value_numeric'])
        
        if len(data) < 2:
            return {}
        
        growth_rates = {}
        prev_year = None
        prev_value = None
        
        for _, row in data.iterrows():
            if prev_year is not None:
                period = f"{prev_year}-{row['year']}"
                growth = ((row['value_numeric'] - prev_value) / prev_value) * 100 if prev_value != 0 else 0
                growth_rates[period] = growth
            
            prev_year = row['year']
            prev_value = row['value_numeric']
        
        return growth_rates
    
    def _analyze_gender_gap(self, access_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze gender gap in account ownership."""
        gender_data = access_data[
            access_data['indicator'].str.contains('gender|male|female', case=False, na=False) |
            access_data['indicator_code'].str.contains('gender|male|female', case=False, na=False)
        ].copy()
        
        if len(gender_data) == 0:
            return None
        
        # Try to separate male and female
        male_data = gender_data[gender_data['indicator'].str.contains('male', case=False, na=False)]
        female_data = gender_data[gender_data['indicator'].str.contains('female', case=False, na=False)]
        
        if len(male_data) > 0 and len(female_data) > 0:
            return {
                'male_data': male_data[['year', 'value_numeric']].to_dict('records') if 'year' in male_data.columns else [],
                'female_data': female_data[['year', 'value_numeric']].to_dict('records') if 'year' in female_data.columns else [],
                'gap': self._calculate_gender_gap(male_data, female_data)
            }
        
        return None
    
    def _calculate_gender_gap(self, male_data: pd.DataFrame, female_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate gender gap."""
        gap = {}
        
        if 'year' in male_data.columns and 'year' in female_data.columns:
            for year in sorted(set(male_data['year'].dropna()) & set(female_data['year'].dropna())):
                male_val = male_data[male_data['year'] == year]['value_numeric'].values
                female_val = female_data[female_data['year'] == year]['value_numeric'].values
                
                if len(male_val) > 0 and len(female_val) > 0:
                    gap[str(int(year))] = float(male_val[0] - female_val[0])
        
        return gap
    
    def _analyze_urban_rural(self, access_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze urban vs rural account ownership."""
        urban_rural_data = access_data[
            access_data['indicator'].str.contains('urban|rural', case=False, na=False) |
            access_data['indicator_code'].str.contains('urban|rural', case=False, na=False)
        ].copy()
        
        if len(urban_rural_data) == 0:
            return None
        
        urban_data = urban_rural_data[urban_rural_data['indicator'].str.contains('urban', case=False, na=False)]
        rural_data = urban_rural_data[urban_rural_data['indicator'].str.contains('rural', case=False, na=False)]
        
        if len(urban_data) > 0 and len(rural_data) > 0:
            return {
                'urban_data': urban_data[['year', 'value_numeric']].to_dict('records') if 'year' in urban_data.columns else [],
                'rural_data': rural_data[['year', 'value_numeric']].to_dict('records') if 'year' in rural_data.columns else []
            }
        
        return None
    
    def _analyze_slowdown(self, account_ownership: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze 2021-2024 slowdown in account ownership growth."""
        if account_ownership is None or 'year' not in account_ownership.columns:
            return None
        
        # Filter 2021-2024 period
        recent_data = account_ownership[
            (account_ownership['year'] >= 2021) & (account_ownership['year'] <= 2024)
        ].copy()
        
        if len(recent_data) < 2:
            return None
        
        # Compare with previous period
        previous_data = account_ownership[
            (account_ownership['year'] >= 2017) & (account_ownership['year'] < 2021)
        ].copy()
        
        slowdown = {
            'recent_period': {
                'years': sorted(recent_data['year'].unique().tolist()),
                'values': recent_data['value_numeric'].tolist(),
                'total_growth': float(recent_data['value_numeric'].iloc[-1] - recent_data['value_numeric'].iloc[0]) if len(recent_data) > 0 else 0
            }
        }
        
        if len(previous_data) >= 2:
            slowdown['previous_period'] = {
                'years': sorted(previous_data['year'].unique().tolist()),
                'values': previous_data['value_numeric'].tolist(),
                'total_growth': float(previous_data['value_numeric'].iloc[-1] - previous_data['value_numeric'].iloc[0])
            }
            slowdown['comparison'] = {
                'recent_growth': slowdown['recent_period']['total_growth'],
                'previous_growth': slowdown['previous_period']['total_growth'],
                'growth_difference': slowdown['recent_period']['total_growth'] - slowdown['previous_period']['total_growth']
            }
        
        return slowdown
    
    def plot_account_ownership_trajectory(self, save: bool = True) -> plt.Figure:
        """
        Plot account ownership trajectory (2011-2024).
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.results.access_analysis is None:
            self.analyze_access()
        
        account_data = self.results.access_analysis.get('account_ownership')
        
        if account_data is None or len(account_data) == 0:
            self.logger.warning("No account ownership data available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'year' in account_data.columns:
            account_data = account_data.sort_values('year')
            ax.plot(account_data['year'], account_data['value_numeric'], 
                   marker='o', linewidth=2, markersize=8, label='Account Ownership (%)')
            
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Account Ownership (%)', fontsize=12)
            ax.set_title('Ethiopia Account Ownership Trajectory (2011-2024)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add growth rate annotations
            growth_rates = self.results.access_analysis.get('growth_rates', {})
            for period, rate in list(growth_rates.items())[-3:]:  # Last 3 periods
                years = period.split('-')
                if len(years) == 2:
                    try:
                        year = int(years[1])
                        value = account_data[account_data['year'] == year]['value_numeric'].values
                        if len(value) > 0:
                            ax.annotate(f'{rate:.1f}%', 
                                      xy=(year, value[0]),
                                      xytext=(10, 10), textcoords='offset points',
                                      fontsize=9, alpha=0.7)
                    except:
                        pass
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.figure_dir / 'account_ownership_trajectory.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved account ownership trajectory to {self.figure_dir / 'account_ownership_trajectory.png'}")
        
        return fig
    
    def analyze_usage(self) -> Dict[str, Any]:
        """
        Analyze Usage (Digital Payments) pillar data.
        
        Returns:
            Dictionary with usage analysis results
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Analyzing Usage pillar...")
        
        # Filter usage observations
        usage_data = self.data_df[
            (self.data_df['record_type'] == 'observation') &
            (self.data_df['pillar'] == 'usage')
        ].copy()
        
        if len(usage_data) == 0:
            self.logger.warning("No usage data found")
            return {}
        
        analysis = {}
        
        # Mobile money account penetration
        mobile_money = self._extract_mobile_money_data(usage_data)
        if mobile_money is not None and len(mobile_money) > 0:
            analysis['mobile_money'] = mobile_money
        
        # Digital payment adoption
        digital_payments = self._extract_digital_payment_data(usage_data)
        if digital_payments is not None and len(digital_payments) > 0:
            analysis['digital_payments'] = digital_payments
        
        # Registered vs active gap
        registered_active = self._analyze_registered_active_gap(usage_data)
        if registered_active:
            analysis['registered_active_gap'] = registered_active
        
        # Payment use cases
        use_cases = self._analyze_payment_use_cases(usage_data)
        if use_cases:
            analysis['payment_use_cases'] = use_cases
        
        self.results.usage_analysis = analysis
        self.logger.info("Usage analysis complete")
        
        return analysis
    
    def _extract_mobile_money_data(self, usage_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract mobile money account penetration data."""
        mobile_data = usage_data[
            usage_data['indicator'].str.contains('mobile|money|m-pesa|telebirr', case=False, na=False) |
            usage_data['indicator_code'].str.contains('mobile|money|m-pesa|telebirr', case=False, na=False)
        ].copy()
        
        if len(mobile_data) == 0:
            # If no specific mobile money data, use any usage data
            mobile_data = usage_data[usage_data['value_numeric'].notna()].copy()
        
        if len(mobile_data) == 0:
            return None
        
        # Add year if date column exists
        date_col = [col for col in mobile_data.columns if 'date' in col.lower()][0] if \
            [col for col in mobile_data.columns if 'date' in col.lower()] else None
        
        if date_col:
            mobile_data['year'] = pd.to_datetime(mobile_data[date_col], errors='coerce').dt.year
        
        return mobile_data
    
    def _extract_digital_payment_data(self, usage_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract digital payment adoption data."""
        digital_data = usage_data[
            usage_data['indicator'].str.contains('digital|payment|transaction', case=False, na=False) |
            usage_data['indicator_code'].str.contains('digital|payment|transaction', case=False, na=False)
        ].copy()
        
        if len(digital_data) == 0:
            return None
        
        date_col = [col for col in digital_data.columns if 'date' in col.lower()][0] if \
            [col for col in digital_data.columns if 'date' in col.lower()] else None
        
        if date_col:
            digital_data['year'] = pd.to_datetime(digital_data[date_col], errors='coerce').dt.year
        
        return digital_data
    
    def _analyze_registered_active_gap(self, usage_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze registered vs active account gap."""
        registered = usage_data[
            usage_data['indicator'].str.contains('registered|account', case=False, na=False)
        ].copy()
        
        active = usage_data[
            usage_data['indicator'].str.contains('active|usage|transaction', case=False, na=False)
        ].copy()
        
        if len(registered) > 0 and len(active) > 0:
            return {
                'registered_data': registered[['year', 'value_numeric']].to_dict('records') if 'year' in registered.columns else [],
                'active_data': active[['year', 'value_numeric']].to_dict('records') if 'year' in active.columns else []
            }
        
        return None
    
    def _analyze_payment_use_cases(self, usage_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Analyze payment use cases (P2P, merchant, bill pay, wages)."""
        use_cases = {}
        
        for case in ['P2P', 'merchant', 'bill', 'wage', 'salary']:
            case_data = usage_data[
                usage_data['indicator'].str.contains(case, case=False, na=False) |
                usage_data['indicator_code'].str.contains(case, case=False, na=False)
            ].copy()
            
            if len(case_data) > 0:
                use_cases[case] = case_data[['year', 'value_numeric']].to_dict('records') if 'year' in case_data.columns else []
        
        return use_cases if len(use_cases) > 0 else None
    
    def analyze_infrastructure(self) -> Dict[str, Any]:
        """
        Analyze infrastructure and enablers data.
        
        Returns:
            Dictionary with infrastructure analysis results
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Analyzing infrastructure data...")
        
        # Look for infrastructure-related observations
        infra_data = self.data_df[
            (self.data_df['record_type'] == 'observation') &
            (
                self.data_df['indicator'].str.contains('4G|mobile|penetration|ATM|infrastructure', case=False, na=False) |
                self.data_df['indicator_code'].str.contains('4G|mobile|penetration|ATM|infrastructure', case=False, na=False)
            )
        ].copy()
        
        if len(infra_data) == 0:
            self.logger.warning("No infrastructure data found")
            return {}
        
        analysis = {}
        
        # Categorize infrastructure indicators
        analysis['4G_coverage'] = self._extract_indicator(infra_data, '4G|coverage')
        analysis['mobile_penetration'] = self._extract_indicator(infra_data, 'mobile|penetration')
        analysis['ATM_density'] = self._extract_indicator(infra_data, 'ATM|density')
        
        # Relationships with inclusion outcomes
        analysis['relationships'] = self._analyze_infrastructure_relationships(infra_data)
        
        # Leading indicators
        analysis['leading_indicators'] = self._identify_leading_indicators(infra_data)
        
        self.results.infrastructure_analysis = analysis
        self.logger.info("Infrastructure analysis complete")
        
        return analysis
    
    def _extract_indicator(self, data: pd.DataFrame, pattern: str) -> Optional[pd.DataFrame]:
        """Extract specific indicator by pattern."""
        indicator_data = data[
            data['indicator'].str.contains(pattern, case=False, na=False, regex=True) |
            data['indicator_code'].str.contains(pattern, case=False, na=False, regex=True)
        ].copy()
        
        if len(indicator_data) == 0:
            return None
        
        date_col = [col for col in indicator_data.columns if 'date' in col.lower()][0] if \
            [col for col in indicator_data.columns if 'date' in col.lower()] else None
        
        if date_col:
            indicator_data['year'] = pd.to_datetime(indicator_data[date_col], errors='coerce').dt.year
        
        return indicator_data
    
    def _analyze_infrastructure_relationships(self, infra_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between infrastructure and inclusion outcomes."""
        # This would require correlation analysis with access/usage data
        # For now, return structure
        return {
            'note': 'Requires correlation analysis with access/usage indicators'
        }
    
    def _identify_leading_indicators(self, infra_data: pd.DataFrame) -> List[str]:
        """Identify potential leading indicators."""
        leading = []
        
        # Infrastructure typically leads inclusion outcomes
        if len(infra_data) > 0:
            leading = infra_data['indicator_code'].unique().tolist()
        
        return leading
    
    def create_event_timeline(self) -> pd.DataFrame:
        """
        Create event timeline visualization data.
        
        Returns:
            DataFrame with events and dates
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Creating event timeline...")
        
        events = self.data_df[self.data_df['record_type'] == 'event'].copy()
        
        if len(events) == 0:
            self.logger.warning("No events found")
            return pd.DataFrame()
        
        # Get event date column
        date_col = [col for col in events.columns if 'date' in col.lower()][0] if \
            [col for col in events.columns if 'date' in col.lower()] else None
        
        if date_col:
            events['event_date'] = pd.to_datetime(events[date_col], errors='coerce')
            events = events.sort_values('event_date')
        
        timeline = events[['event_date', 'category', 'id']].copy() if 'event_date' in events.columns else events
        
        self.results.event_timeline = timeline
        self.logger.info(f"Created timeline with {len(timeline)} events")
        
        return timeline
    
    def visualize_event_timeline(self, save: bool = True) -> plt.Figure:
        """
        Create timeline visualization showing all cataloged events.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.results.event_timeline is None:
            self.create_event_timeline()
        
        timeline = self.results.event_timeline
        
        if len(timeline) == 0:
            self.logger.warning("No events to visualize")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if 'event_date' in timeline.columns:
            # Create timeline plot
            categories = timeline['category'].unique() if 'category' in timeline.columns else ['event']
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            for i, (date, row) in enumerate(timeline.iterrows()):
                category = row['category'] if 'category' in row else 'event'
                color = colors[list(categories).index(category)] if category in categories else 'gray'
                
                ax.scatter(row['event_date'], i, s=100, c=[color], alpha=0.7, label=category if i == 0 or category not in [l.get_label() for l in ax.get_legend_handles_labels()[0]] else "")
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Event Index', fontsize=12)
            ax.set_title('Event Timeline', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if len(categories) > 1:
                ax.legend()
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.figure_dir / 'event_timeline.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved event timeline to {self.figure_dir / 'event_timeline.png'}")
        
        return fig
    
    def overlay_events_on_trends(self, indicator_data: pd.DataFrame, indicator_name: str, save: bool = True) -> plt.Figure:
        """
        Overlay events on indicator trend charts.
        
        Args:
            indicator_data: DataFrame with indicator data
            indicator_name: Name of the indicator
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.results.event_timeline is None:
            self.create_event_timeline()
        
        timeline = self.results.event_timeline
        
        if len(timeline) == 0 or len(indicator_data) == 0:
            self.logger.warning("No data available for overlay")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot indicator trend
        if 'year' in indicator_data.columns:
            indicator_data = indicator_data.sort_values('year')
            ax.plot(indicator_data['year'], indicator_data['value_numeric'],
                   marker='o', linewidth=2, markersize=8, label=indicator_name)
        
        # Overlay events
        if 'event_date' in timeline.columns:
            for _, event in timeline.iterrows():
                event_year = event['event_date'].year if pd.notna(event['event_date']) else None
                if event_year:
                    ax.axvline(x=event_year, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    if 'category' in event:
                        ax.text(event_year, ax.get_ylim()[1] * 0.95, event['category'],
                               rotation=90, fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{indicator_name} with Event Overlay', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            filename = f'{indicator_name.lower().replace(" ", "_")}_with_events.png'
            fig.savefig(self.figure_dir / filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved overlay figure to {self.figure_dir / filename}")
        
        return fig
    
    def correlation_analysis(self) -> pd.DataFrame:
        """
        Examine correlations between different indicators.
        
        Returns:
            Correlation matrix DataFrame
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Performing correlation analysis...")
        
        # Get numeric observations
        observations = self.data_df[
            (self.data_df['record_type'] == 'observation') &
            (self.data_df['value_numeric'].notna())
        ].copy()
        
        if len(observations) == 0:
            self.logger.warning("No numeric observations for correlation")
            return pd.DataFrame()
        
        # Pivot to get indicators as columns and years as index
        date_col = [col for col in observations.columns if 'date' in col.lower()][0] if \
            [col for col in observations.columns if 'date' in col.lower()] else None
        
        if date_col:
            observations['year'] = pd.to_datetime(observations[date_col], errors='coerce').dt.year
        
        if 'year' in observations.columns and 'indicator_code' in observations.columns:
            # Create pivot table
            pivot = observations.pivot_table(
                index='year',
                columns='indicator_code',
                values='value_numeric',
                aggfunc='mean'
            )
            
            # Calculate correlations
            correlations = pivot.corr()
            
            self.results.correlations = correlations
            self.logger.info(f"Calculated correlations for {len(correlations)} indicators")
            
            return correlations
        
        return pd.DataFrame()
    
    def visualize_correlations(self, save: bool = True) -> plt.Figure:
        """
        Visualize correlation matrix.
        
        Args:
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        if self.results.correlations is None:
            self.correlation_analysis()
        
        correlations = self.results.correlations
        
        if len(correlations) == 0:
            self.logger.warning("No correlations to visualize")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlations, dtype=bool))
        sns.heatmap(correlations, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Indicator Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.figure_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved correlation matrix to {self.figure_dir / 'correlation_matrix.png'}")
        
        return fig
    
    def generate_insights(self) -> List[str]:
        """
        Generate key insights from the analysis.
        
        Returns:
            List of insight strings
        """
        self.logger.info("Generating insights...")
        
        insights = []
        
        # Access insights
        if self.results.access_analysis:
            slowdown = self.results.access_analysis.get('slowdown_analysis')
            if slowdown:
                recent_growth = slowdown.get('recent_period', {}).get('total_growth', 0)
                insights.append(
                    f"Account ownership growth slowed significantly in 2021-2024 period, "
                    f"with only {recent_growth:.1f}pp growth despite massive mobile money expansion."
                )
        
        # Usage insights
        if self.results.usage_analysis:
            mobile_money = self.results.usage_analysis.get('mobile_money')
            if mobile_money is not None and len(mobile_money) > 0:
                insights.append(
                    f"Mobile money account penetration shows {len(mobile_money)} data points, "
                    f"indicating growing digital payment infrastructure."
                )
        
        # Infrastructure insights
        if self.results.infrastructure_analysis:
            leading = self.results.infrastructure_analysis.get('leading_indicators', [])
            if len(leading) > 0:
                insights.append(
                    f"Identified {len(leading)} potential leading indicators from infrastructure data."
                )
        
        # Correlation insights
        if self.results.correlations is not None and len(self.results.correlations) > 0:
            # Find strongest correlations
            corr_matrix = self.results.correlations
            corr_matrix = corr_matrix.replace(1.0, np.nan)  # Remove self-correlations
            
            if not corr_matrix.empty:
                max_corr = corr_matrix.max().max()
                insights.append(
                    f"Strongest correlation between indicators: {max_corr:.2f}, "
                    f"suggesting potential relationships for impact modeling."
                )
        
        # Data quality insights
        if self.results.dataset_overview:
            gaps = self.results.dataset_overview.get('data_gaps', {})
            sparse_count = gaps.get('sparse_count', 0)
            if sparse_count > 0:
                insights.append(
                    f"Data quality assessment: {sparse_count} indicators have sparse coverage "
                    f"(<5 observations), limiting analysis depth."
                )
        
        self.results.insights = insights
        self.logger.info(f"Generated {len(insights)} insights")
        
        return insights
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """
        Assess overall data quality.
        
        Returns:
            Dictionary with data quality assessment
        """
        if self.data_df is None:
            self.load_data()
        
        self.logger.info("Assessing data quality...")
        
        quality = {
            'total_records': len(self.data_df),
            'missing_values': {},
            'confidence_distribution': {},
            'temporal_coverage': {},
            'data_gaps': {},
            'limitations': []
        }
        
        # Missing values
        if self.data_df is not None:
            missing = self.data_df.isnull().sum()
            quality['missing_values'] = missing[missing > 0].to_dict()
        
        # Confidence distribution
        if 'confidence' in self.data_df.columns:
            quality['confidence_distribution'] = self.data_df['confidence'].value_counts().to_dict()
        
        # Temporal coverage
        temporal = self._get_temporal_coverage()
        quality['temporal_coverage'] = {
            'overall_range': f"{temporal.get('overall_min', 'N/A')}-{temporal.get('overall_max', 'N/A')}",
            'indicators_with_data': len(temporal.get('by_indicator', []))
        }
        
        # Data gaps
        gaps = self._identify_data_gaps()
        quality['data_gaps'] = gaps
        
        # Limitations
        if gaps.get('sparse_count', 0) > 0:
            quality['limitations'].append(
                f"{gaps['sparse_count']} indicators have sparse coverage, limiting trend analysis"
            )
        
        if len(quality['missing_values']) > 0:
            quality['limitations'].append(
                f"Missing values found in {len(quality['missing_values'])} columns"
            )
        
        self.results.data_quality = quality
        self.logger.info("Data quality assessment complete")
        
        return quality
    
    def run_full_eda(self) -> EDAResults:
        """
        Run complete EDA pipeline.
        
        Returns:
            EDAResults object with all analysis results
        """
        self.logger.info("Starting full EDA pipeline...")
        
        # Load data
        self.load_data()
        
        # Dataset overview
        self.dataset_overview()
        self.visualize_temporal_coverage()
        
        # Access analysis
        self.analyze_access()
        self.plot_account_ownership_trajectory()
        
        # Usage analysis
        self.analyze_usage()
        
        # Infrastructure analysis
        self.analyze_infrastructure()
        
        # Event timeline
        self.create_event_timeline()
        self.visualize_event_timeline()
        
        # Correlation analysis
        self.correlation_analysis()
        self.visualize_correlations()
        
        # Generate insights
        self.generate_insights()
        
        # Data quality assessment
        self.assess_data_quality()
        
        self.logger.info("Full EDA pipeline complete")
        
        return self.results
