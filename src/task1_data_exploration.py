"""
Task 1: Data Exploration and Enrichment Module
OOP-based module for exploring and enriching the Ethiopia Financial Inclusion dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')


class ConfidenceLevel(Enum):
    """Confidence levels for data quality."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ObservationRecord:
    """Data class for observation records."""
    pillar: str
    indicator: str
    indicator_code: str
    value_numeric: Optional[float] = None
    value_text: Optional[str] = None
    value_type: str = "percentage"
    unit: str = "%"
    observation_date: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    fiscal_year: Optional[int] = None
    gender: str = "all"
    location: str = "national"
    region: Optional[str] = None
    source_name: str = ""
    source_type: str = "survey"
    source_url: str = ""
    confidence: str = ConfidenceLevel.MEDIUM.value
    collected_by: str = ""
    original_text: Optional[str] = None
    notes: Optional[str] = None
    category: Optional[str] = None
    indicator_direction: str = "higher_better"
    related_indicator: Optional[str] = None
    relationship_type: Optional[str] = None


@dataclass
class EventRecord:
    """Data class for event records."""
    event_name: str
    event_date: str
    category: str
    source_name: str = ""
    source_type: str = "news"
    source_url: str = ""
    confidence: str = ConfidenceLevel.MEDIUM.value
    collected_by: str = ""
    original_text: Optional[str] = None
    notes: Optional[str] = None
    pillar: Optional[str] = None
    indicator: Optional[str] = None
    indicator_code: Optional[str] = None


@dataclass
class ImpactLinkRecord:
    """Data class for impact link records."""
    parent_id: str  # Event record_id
    pillar: str
    indicator: str
    indicator_code: str
    impact_direction: str  # "increase", "decrease", "neutral"
    impact_magnitude: str  # "high", "medium", "low"
    impact_estimate: Optional[float] = None
    lag_months: int = 0
    evidence_basis: str = "literature"  # "literature", "empirical", "expert"
    comparable_country: Optional[str] = None
    confidence: str = ConfidenceLevel.MEDIUM.value
    collected_by: str = ""
    notes: Optional[str] = None
    relationship_type: str = "direct"
    observation_date: Optional[str] = None


class DataEnricher:
    """Class for enriching the dataset with new observations, events, and impact links."""
    
    def __init__(self, data_df: pd.DataFrame, impact_df: pd.DataFrame, reference_codes_df: pd.DataFrame):
        """
        Initialize the DataEnricher.
        
        Args:
            data_df: Main data dataframe
            impact_df: Impact links dataframe
            reference_codes_df: Reference codes dataframe for validation
        """
        self.data_df = data_df.copy()
        self.impact_df = impact_df.copy()
        self.reference_codes_df = reference_codes_df
        self.enrichment_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
    def _generate_record_id(self, prefix: str) -> str:
        """Generate a unique record ID."""
        existing_ids = set(self.data_df['record_id'].tolist())
        if prefix == 'IMP':
            existing_ids.update(self.impact_df['record_id'].tolist())
        
        counter = 1
        while True:
            new_id = f"{prefix}_{counter:04d}"
            if new_id not in existing_ids:
                return new_id
            counter += 1
    
    def _validate_pillar(self, pillar: str) -> bool:
        """Validate pillar value against reference codes."""
        valid_pillars = self.reference_codes_df[
            (self.reference_codes_df['field'] == 'pillar') & 
            (self.reference_codes_df['code'].str.upper() == pillar.upper())
        ]
        return not valid_pillars.empty
    
    def _validate_category(self, category: str) -> bool:
        """Validate category value against reference codes."""
        valid_categories = self.reference_codes_df[
            (self.reference_codes_df['field'] == 'category') & 
            (self.reference_codes_df['code'].str.lower() == category.lower())
        ]
        return not valid_categories.empty
    
    def add_observation(self, observation: ObservationRecord) -> str:
        """
        Add a new observation record to the dataset.
        
        Args:
            observation: ObservationRecord instance
            
        Returns:
            record_id of the added observation
        """
        record_id = self._generate_record_id('OBS')
        
        # Validate pillar
        if not self._validate_pillar(observation.pillar):
            self.logger.warning(f"Invalid pillar: {observation.pillar}")
        
        # Convert dates
        obs_date = pd.to_datetime(observation.observation_date) if observation.observation_date else None
        period_start = pd.to_datetime(observation.period_start) if observation.period_start else None
        period_end = pd.to_datetime(observation.period_end) if observation.period_end else None
        
        new_row = {
            'record_id': record_id,
            'record_type': 'observation',
            'category': observation.category,
            'pillar': observation.pillar.upper(),
            'indicator': observation.indicator,
            'indicator_code': observation.indicator_code,
            'indicator_direction': observation.indicator_direction,
            'value_numeric': observation.value_numeric,
            'value_text': observation.value_text,
            'value_type': observation.value_type,
            'unit': observation.unit,
            'observation_date': obs_date,
            'period_start': period_start,
            'period_end': period_end,
            'fiscal_year': observation.fiscal_year,
            'gender': observation.gender,
            'location': observation.location,
            'region': observation.region,
            'source_name': observation.source_name,
            'source_type': observation.source_type,
            'source_url': observation.source_url,
            'confidence': observation.confidence,
            'related_indicator': observation.related_indicator,
            'relationship_type': observation.relationship_type,
            'impact_direction': None,
            'impact_magnitude': None,
            'impact_estimate': None,
            'lag_months': None,
            'evidence_basis': None,
            'comparable_country': None,
            'collected_by': observation.collected_by,
            'collection_date': datetime.now(),
            'original_text': observation.original_text,
            'notes': observation.notes
        }
        
        # Add to dataframe
        new_df = pd.DataFrame([new_row])
        self.data_df = pd.concat([self.data_df, new_df], ignore_index=True)
        
        # Log enrichment
        self.enrichment_log.append({
            'record_id': record_id,
            'record_type': 'observation',
            'indicator': observation.indicator,
            'indicator_code': observation.indicator_code,
            'value': observation.value_numeric,
            'date': observation.observation_date,
            'source': observation.source_name,
            'source_url': observation.source_url,
            'confidence': observation.confidence,
            'collector': observation.collected_by,
            'rationale': observation.notes or observation.original_text or f"Added observation for {observation.indicator}"
        })
        
        self.logger.info(f"Added observation: {record_id} - {observation.indicator}")
        return record_id
    
    def add_event(self, event: EventRecord) -> str:
        """
        Add a new event record to the dataset.
        
        Args:
            event: EventRecord instance
            
        Returns:
            record_id of the added event
        """
        record_id = self._generate_record_id('EVT')
        
        # Validate category
        if not self._validate_category(event.category):
            self.logger.warning(f"Invalid category: {event.category}")
        
        # Convert date
        event_date = pd.to_datetime(event.event_date) if event.event_date else None
        
        new_row = {
            'record_id': record_id,
            'record_type': 'event',
            'category': event.category,
            'pillar': event.pillar,
            'indicator': event.event_name,
            'indicator_code': event.indicator_code,
            'indicator_direction': None,
            'value_numeric': None,
            'value_text': None,
            'value_type': None,
            'unit': None,
            'observation_date': event_date,
            'period_start': None,
            'period_end': None,
            'fiscal_year': None,
            'gender': None,
            'location': None,
            'region': None,
            'source_name': event.source_name,
            'source_type': event.source_type,
            'source_url': event.source_url,
            'confidence': event.confidence,
            'related_indicator': None,
            'relationship_type': None,
            'impact_direction': None,
            'impact_magnitude': None,
            'impact_estimate': None,
            'lag_months': None,
            'evidence_basis': None,
            'comparable_country': None,
            'collected_by': event.collected_by,
            'collection_date': datetime.now(),
            'original_text': event.original_text,
            'notes': event.notes
        }
        
        # Add to dataframe
        new_df = pd.DataFrame([new_row])
        self.data_df = pd.concat([self.data_df, new_df], ignore_index=True)
        
        # Log enrichment
        self.enrichment_log.append({
            'record_id': record_id,
            'record_type': 'event',
            'indicator': event.event_name,
            'indicator_code': event.indicator_code,
            'value': None,
            'date': event.event_date,
            'source': event.source_name,
            'source_url': event.source_url,
            'confidence': event.confidence,
            'collector': event.collected_by,
            'rationale': event.notes or event.original_text or f"Added event: {event.event_name}"
        })
        
        self.logger.info(f"Added event: {record_id} - {event.event_name}")
        return record_id
    
    def add_impact_link(self, impact_link: ImpactLinkRecord) -> str:
        """
        Add a new impact link record to the impact links dataset.
        
        Args:
            impact_link: ImpactLinkRecord instance
            
        Returns:
            record_id of the added impact link
        """
        record_id = self._generate_record_id('IMP')
        
        # Validate pillar
        if not self._validate_pillar(impact_link.pillar):
            self.logger.warning(f"Invalid pillar: {impact_link.pillar}")
        
        # Convert date
        obs_date = pd.to_datetime(impact_link.observation_date) if impact_link.observation_date else None
        
        new_row = {
            'record_id': record_id,
            'parent_id': impact_link.parent_id,
            'record_type': 'impact_link',
            'category': None,
            'pillar': impact_link.pillar.upper(),
            'indicator': impact_link.indicator,
            'indicator_code': impact_link.indicator_code,
            'indicator_direction': None,
            'value_numeric': impact_link.impact_estimate,
            'value_text': None,
            'value_type': 'percentage' if impact_link.impact_estimate else None,
            'unit': '%' if impact_link.impact_estimate else None,
            'observation_date': obs_date,
            'period_start': None,
            'period_end': None,
            'fiscal_year': None,
            'gender': 'all',
            'location': 'national',
            'region': None,
            'source_name': None,
            'source_type': None,
            'source_url': None,
            'confidence': impact_link.confidence,
            'related_indicator': impact_link.indicator_code,
            'relationship_type': impact_link.relationship_type,
            'impact_direction': impact_link.impact_direction,
            'impact_magnitude': impact_link.impact_magnitude,
            'impact_estimate': impact_link.impact_estimate,
            'lag_months': impact_link.lag_months,
            'evidence_basis': impact_link.evidence_basis,
            'comparable_country': impact_link.comparable_country,
            'collected_by': impact_link.collected_by,
            'collection_date': datetime.now(),
            'original_text': None,
            'notes': impact_link.notes
        }
        
        # Add to dataframe
        new_df = pd.DataFrame([new_row])
        self.impact_df = pd.concat([self.impact_df, new_df], ignore_index=True)
        
        # Log enrichment
        self.enrichment_log.append({
            'record_id': record_id,
            'record_type': 'impact_link',
            'indicator': impact_link.indicator,
            'indicator_code': impact_link.indicator_code,
            'value': impact_link.impact_estimate,
            'date': impact_link.observation_date,
            'source': impact_link.evidence_basis,
            'source_url': None,
            'confidence': impact_link.confidence,
            'collector': impact_link.collected_by,
            'rationale': impact_link.notes or f"Impact link: {impact_link.impact_direction} {impact_link.impact_magnitude} impact on {impact_link.indicator_code}"
        })
        
        self.logger.info(f"Added impact link: {record_id} - {impact_link.indicator}")
        return record_id
    
    def get_enrichment_log(self) -> pd.DataFrame:
        """Get the enrichment log as a dataframe."""
        return pd.DataFrame(self.enrichment_log)
    
    def save_enriched_data(self, output_file: Path):
        """Save the enriched dataset to Excel file."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            self.data_df.to_excel(writer, sheet_name='ethiopia_fi_unified_data', index=False)
            self.impact_df.to_excel(writer, sheet_name='Impact_sheet', index=False)
        self.logger.info(f"Saved enriched data to {output_file}")


class Task1DataProcessor:
    """Main processor class for Task 1 data exploration and enrichment."""
    
    def __init__(
        self,
        data_file: Path,
        reference_codes_file: Path,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Task1DataProcessor.
        
        Args:
            data_file: Path to the main data Excel file
            reference_codes_file: Path to reference codes Excel file
            logger: Optional logger instance
        """
        self.data_file = Path(data_file)
        self.reference_codes_file = Path(reference_codes_file)
        self.logger = logger or self._setup_logger()
        
        # Load data
        self.data_df: Optional[pd.DataFrame] = None
        self.impact_df: Optional[pd.DataFrame] = None
        self.reference_codes_df: Optional[pd.DataFrame] = None
        self.enricher: Optional[DataEnricher] = None
        
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
    
    def load_data(self):
        """Load data from Excel files."""
        self.logger.info("Loading data files...")
        
        # Load main data
        xl_file = pd.ExcelFile(self.data_file)
        self.data_df = pd.read_excel(self.data_file, sheet_name=xl_file.sheet_names[0])
        
        # Load impact links
        if len(xl_file.sheet_names) > 1:
            self.impact_df = pd.read_excel(self.data_file, sheet_name=xl_file.sheet_names[1])
        else:
            self.impact_df = pd.DataFrame()
        
        # Load reference codes
        self.reference_codes_df = pd.read_excel(self.reference_codes_file)
        
        # Initialize enricher
        self.enricher = DataEnricher(self.data_df, self.impact_df, self.reference_codes_df)
        
        self.logger.info(f"Loaded {len(self.data_df)} records from main data")
        self.logger.info(f"Loaded {len(self.impact_df)} impact links")
    
    def run_full_exploration(self) -> Dict[str, Any]:
        """Run full data exploration and return summary statistics."""
        if self.data_df is None:
            self.load_data()
        
        results = {
            'total_records': len(self.data_df),
            'observations': len(self.data_df[self.data_df['record_type'] == 'observation']),
            'events': len(self.data_df[self.data_df['record_type'] == 'event']),
            'targets': len(self.data_df[self.data_df['record_type'] == 'target']),
            'impact_links': len(self.impact_df),
            'unique_indicators': self.data_df['indicator_code'].nunique()
        }
        
        return results
