"""
Task 1: Data Exploration and Enrichment Module
Comprehensive OOP-based solution for exploring and enriching the Ethiopia Financial Inclusion dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for data quality."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecordType(Enum):
    """Types of records in the dataset."""
    OBSERVATION = "observation"
    EVENT = "event"
    TARGET = "target"


class ImpactDirection(Enum):
    """Direction of impact for impact links."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class ObservationRecord:
    """Data class for observation records."""
    pillar: str
    indicator: str
    indicator_code: str
    value_numeric: float
    observation_date: str
    source_name: str
    source_url: str
    confidence: str
    collected_by: str
    original_text: str
    notes: str
    record_type: str = "observation"
    collection_date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    additional_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    """Data class for event records."""
    category: str
    event_date: str
    source_name: str
    source_url: str
    confidence: str
    collected_by: str
    original_text: str
    notes: str
    record_type: str = "event"
    pillar: Optional[str] = None  # Events should have empty pillar
    collection_date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    additional_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactLinkRecord:
    """Data class for impact link records."""
    parent_id: str
    pillar: str
    related_indicator: str
    impact_direction: str
    evidence_basis: str
    source_name: str
    source_url: str
    confidence: str
    collected_by: str
    notes: str
    impact_magnitude: Optional[float] = None
    lag_months: Optional[int] = None
    collection_date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    additional_fields: Dict[str, Any] = field(default_factory=dict)


class DataExplorer:
    """Class for exploring and analyzing the dataset."""
    
    def __init__(self, data_file: Path, reference_codes_file: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataExplorer.
        
        Args:
            data_file: Path to the main data Excel file
            reference_codes_file: Path to the reference codes Excel file
            logger: Optional logger instance
        """
        self.data_file = Path(data_file)
        self.reference_codes_file = Path(reference_codes_file)
        self.logger = logger or self._setup_logger()
        
        self.data_df: Optional[pd.DataFrame] = None
        self.impact_links_df: Optional[pd.DataFrame] = None
        self.reference_codes_df: Optional[pd.DataFrame] = None
        
        self.logger.info(f"Initialized DataExplorer with data_file: {data_file}")
    
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
        Load all datasets.
        
        Returns:
            Tuple of (data_df, impact_links_df, reference_codes_df)
        """
        self.logger.info("Loading datasets...")
        
        try:
            # Load main data sheet
            self.data_df = pd.read_excel(self.data_file, sheet_name='data')
            self.logger.info(f"Loaded data sheet: {self.data_df.shape}")
            
            # Load impact links sheet
            self.impact_links_df = pd.read_excel(self.data_file, sheet_name='impact_links')
            self.logger.info(f"Loaded impact links: {self.impact_links_df.shape}")
            
            # Load reference codes
            self.reference_codes_df = pd.read_excel(self.reference_codes_file)
            self.logger.info(f"Loaded reference codes: {self.reference_codes_df.shape}")
            
            return self.data_df, self.impact_links_df, self.reference_codes_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information about the datasets."""
        self.logger.info("Analyzing schema...")
        
        if self.data_df is None or self.impact_links_df is None:
            self.load_data()
        
        schema_info = {
            'data_columns': list(self.data_df.columns),
            'data_shape': self.data_df.shape,
            'data_dtypes': self.data_df.dtypes.to_dict(),
            'impact_links_columns': list(self.impact_links_df.columns),
            'impact_links_shape': self.impact_links_df.shape,
            'impact_links_dtypes': self.impact_links_df.dtypes.to_dict(),
        }
        
        self.logger.info(f"Schema analysis complete. Data columns: {len(schema_info['data_columns'])}")
        return schema_info
    
    def get_missing_values(self) -> Dict[str, pd.Series]:
        """Get missing values summary for all datasets."""
        self.logger.info("Analyzing missing values...")
        
        if self.data_df is None or self.impact_links_df is None:
            self.load_data()
        
        missing = {
            'data': self.data_df.isnull().sum(),
            'impact_links': self.impact_links_df.isnull().sum(),
        }
        
        self.logger.info(f"Missing values found in {missing['data'][missing['data'] > 0].shape[0]} data columns")
        return missing
    
    def count_by_record_type(self) -> pd.Series:
        """Count records by record_type."""
        if self.data_df is None:
            self.load_data()
        
        if 'record_type' not in self.data_df.columns:
            self.logger.warning("'record_type' column not found")
            return pd.Series()
        
        counts = self.data_df['record_type'].value_counts()
        self.logger.info(f"Record type distribution: {counts.to_dict()}")
        return counts
    
    def count_by_pillar(self) -> pd.Series:
        """Count records by pillar."""
        if self.data_df is None:
            self.load_data()
        
        if 'pillar' not in self.data_df.columns:
            self.logger.warning("'pillar' column not found")
            return pd.Series()
        
        counts = self.data_df['pillar'].value_counts()
        null_count = self.data_df['pillar'].isnull().sum()
        self.logger.info(f"Pillar distribution: {counts.to_dict()}, Null values: {null_count}")
        return counts
    
    def count_by_source_type(self) -> pd.Series:
        """Count records by source_type."""
        if self.data_df is None:
            self.load_data()
        
        if 'source_type' not in self.data_df.columns:
            self.logger.warning("'source_type' column not found")
            return pd.Series()
        
        counts = self.data_df['source_type'].value_counts()
        self.logger.info(f"Source type distribution: {counts.to_dict()}")
        return counts
    
    def count_by_confidence(self) -> pd.Series:
        """Count records by confidence level."""
        if self.data_df is None:
            self.load_data()
        
        if 'confidence' not in self.data_df.columns:
            self.logger.warning("'confidence' column not found")
            return pd.Series()
        
        counts = self.data_df['confidence'].value_counts()
        self.logger.info(f"Confidence distribution: {counts.to_dict()}")
        return counts
    
    def get_temporal_range(self) -> Dict[str, Dict[str, Any]]:
        """Get temporal range for all date columns."""
        if self.data_df is None:
            self.load_data()
        
        date_columns = [col for col in self.data_df.columns if 'date' in col.lower()]
        self.logger.info(f"Found date columns: {date_columns}")
        
        temporal_info = {}
        for col in date_columns:
            # Convert to datetime if needed
            if self.data_df[col].dtype == 'object':
                self.data_df[col] = pd.to_datetime(self.data_df[col], errors='coerce')
            
            temporal_info[col] = {
                'min': self.data_df[col].min(),
                'max': self.data_df[col].max(),
                'non_null_count': self.data_df[col].notna().sum(),
                'null_count': self.data_df[col].isnull().sum(),
            }
            
            self.logger.info(f"{col}: {temporal_info[col]['min']} to {temporal_info[col]['max']}")
        
        return temporal_info
    
    def get_unique_indicators(self) -> pd.Series:
        """Get unique indicator codes and their counts."""
        if self.data_df is None:
            self.load_data()
        
        if 'indicator_code' not in self.data_df.columns:
            self.logger.warning("'indicator_code' column not found")
            return pd.Series()
        
        counts = self.data_df['indicator_code'].value_counts()
        self.logger.info(f"Found {len(counts)} unique indicators")
        return counts
    
    def get_indicator_coverage_by_pillar(self) -> pd.DataFrame:
        """Get indicator coverage grouped by pillar."""
        if self.data_df is None:
            self.load_data()
        
        if 'pillar' not in self.data_df.columns or 'indicator_code' not in self.data_df.columns:
            self.logger.warning("Required columns not found")
            return pd.DataFrame()
        
        coverage = self.data_df.groupby('pillar')['indicator_code'].nunique().reset_index()
        coverage.columns = ['pillar', 'unique_indicators']
        self.logger.info(f"Indicator coverage by pillar calculated")
        return coverage
    
    def get_events_catalog(self) -> pd.DataFrame:
        """Get all events from the dataset."""
        if self.data_df is None:
            self.load_data()
        
        if 'record_type' not in self.data_df.columns:
            self.logger.warning("'record_type' column not found")
            return pd.DataFrame()
        
        events = self.data_df[self.data_df['record_type'] == 'event'].copy()
        self.logger.info(f"Found {len(events)} events")
        return events
    
    def get_events_by_category(self) -> pd.Series:
        """Get events grouped by category."""
        events = self.get_events_catalog()
        
        if 'category' not in events.columns:
            self.logger.warning("'category' column not found in events")
            return pd.Series()
        
        counts = events['category'].value_counts()
        self.logger.info(f"Events by category: {counts.to_dict()}")
        return counts
    
    def get_impact_links_summary(self) -> Dict[str, Any]:
        """Get summary of impact links."""
        if self.impact_links_df is None:
            self.load_data()
        
        summary = {
            'total_links': len(self.impact_links_df),
        }
        
        if 'pillar' in self.impact_links_df.columns:
            summary['by_pillar'] = self.impact_links_df['pillar'].value_counts().to_dict()
        
        if 'impact_direction' in self.impact_links_df.columns:
            summary['by_direction'] = self.impact_links_df['impact_direction'].value_counts().to_dict()
        
        if 'parent_id' in self.impact_links_df.columns:
            summary['unique_parent_events'] = self.impact_links_df['parent_id'].nunique()
            summary['top_linked_events'] = self.impact_links_df['parent_id'].value_counts().head(10).to_dict()
        
        self.logger.info(f"Impact links summary: {summary}")
        return summary
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate data quality and return issues found."""
        if self.data_df is None or self.impact_links_df is None or self.reference_codes_df is None:
            self.load_data()
        
        issues = {
            'duplicates_data': self.data_df.duplicated().sum(),
            'duplicates_impact_links': self.impact_links_df.duplicated().sum(),
            'invalid_pillars': 0,
        }
        
        # Validate pillars against reference codes
        if 'pillar' in self.data_df.columns and 'field' in self.reference_codes_df.columns:
            valid_pillars = self.reference_codes_df[
                self.reference_codes_df['field'] == 'pillar'
            ]['value'].values if 'pillar' in self.reference_codes_df['field'].values else []
            
            if len(valid_pillars) > 0:
                invalid = self.data_df[
                    ~self.data_df['pillar'].isin(valid_pillars) & 
                    self.data_df['pillar'].notna()
                ]
                issues['invalid_pillars'] = len(invalid)
                if len(invalid) > 0:
                    issues['invalid_pillar_values'] = invalid['pillar'].value_counts().to_dict()
        
        self.logger.info(f"Data quality validation complete. Issues found: {issues}")
        return issues
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics."""
        if self.data_df is None or self.impact_links_df is None:
            self.load_data()
        
        summary = {
            'total_records': len(self.data_df),
            'total_impact_links': len(self.impact_links_df),
        }
        
        if 'record_type' in self.data_df.columns:
            summary['by_record_type'] = self.data_df['record_type'].value_counts().to_dict()
            summary['observations'] = len(self.data_df[self.data_df['record_type'] == 'observation'])
            summary['events'] = len(self.data_df[self.data_df['record_type'] == 'event'])
            summary['targets'] = len(self.data_df[self.data_df['record_type'] == 'target'])
        
        if 'indicator_code' in self.data_df.columns:
            summary['unique_indicators'] = self.data_df['indicator_code'].nunique()
        
        if 'record_type' in self.data_df.columns and 'id' in self.data_df.columns:
            events = self.data_df[self.data_df['record_type'] == 'event']
            summary['unique_events'] = events['id'].nunique() if len(events) > 0 else 0
        
        self.logger.info(f"Summary statistics calculated: {summary}")
        return summary


class DataEnricher:
    """Class for enriching the dataset with new records."""
    
    def __init__(self, data_file: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataEnricher.
        
        Args:
            data_file: Path to the main data Excel file
            logger: Optional logger instance
        """
        self.data_file = Path(data_file)
        self.logger = logger or self._setup_logger()
        
        self.data_df: Optional[pd.DataFrame] = None
        self.impact_links_df: Optional[pd.DataFrame] = None
        
        self.new_observations: List[Dict] = []
        self.new_events: List[Dict] = []
        self.new_impact_links: List[Dict] = []
        
        self.logger.info(f"Initialized DataEnricher with data_file: {data_file}")
    
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
    
    def load_existing_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load existing data from the Excel file."""
        self.logger.info("Loading existing data...")
        
        try:
            self.data_df = pd.read_excel(self.data_file, sheet_name='data')
            self.impact_links_df = pd.read_excel(self.data_file, sheet_name='impact_links')
            
            self.logger.info(f"Loaded {len(self.data_df)} records and {len(self.impact_links_df)} impact links")
            return self.data_df, self.impact_links_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def add_observation(self, observation: ObservationRecord) -> None:
        """
        Add a new observation record.
        
        Args:
            observation: ObservationRecord instance
        """
        record = {
            'record_type': observation.record_type,
            'pillar': observation.pillar,
            'indicator': observation.indicator,
            'indicator_code': observation.indicator_code,
            'value_numeric': observation.value_numeric,
            'observation_date': observation.observation_date,
            'source_name': observation.source_name,
            'source_url': observation.source_url,
            'confidence': observation.confidence,
            'collected_by': observation.collected_by,
            'collection_date': observation.collection_date,
            'original_text': observation.original_text,
            'notes': observation.notes,
            **observation.additional_fields
        }
        
        self.new_observations.append(record)
        self.logger.info(f"Added observation: {observation.indicator_code} - {observation.indicator}")
    
    def add_event(self, event: EventRecord) -> None:
        """
        Add a new event record.
        
        Args:
            event: EventRecord instance
        """
        record = {
            'record_type': event.record_type,
            'pillar': event.pillar,  # Should be None for events
            'category': event.category,
            'event_date': event.event_date,
            'source_name': event.source_name,
            'source_url': event.source_url,
            'confidence': event.confidence,
            'collected_by': event.collected_by,
            'collection_date': event.collection_date,
            'original_text': event.original_text,
            'notes': event.notes,
            **event.additional_fields
        }
        
        self.new_events.append(record)
        self.logger.info(f"Added event: {event.category} on {event.event_date}")
    
    def add_impact_link(self, impact_link: ImpactLinkRecord) -> None:
        """
        Add a new impact link record.
        
        Args:
            impact_link: ImpactLinkRecord instance
        """
        record = {
            'parent_id': impact_link.parent_id,
            'pillar': impact_link.pillar,
            'related_indicator': impact_link.related_indicator,
            'impact_direction': impact_link.impact_direction,
            'impact_magnitude': impact_link.impact_magnitude,
            'lag_months': impact_link.lag_months,
            'evidence_basis': impact_link.evidence_basis,
            'source_name': impact_link.source_name,
            'source_url': impact_link.source_url,
            'confidence': impact_link.confidence,
            'collected_by': impact_link.collected_by,
            'collection_date': impact_link.collection_date,
            'notes': impact_link.notes,
            **impact_link.additional_fields
        }
        
        self.new_impact_links.append(record)
        self.logger.info(f"Added impact link: {impact_link.parent_id} -> {impact_link.related_indicator}")
    
    def _generate_ids(self, count: int, start_id: int) -> List[int]:
        """Generate sequential IDs for new records."""
        return list(range(start_id + 1, start_id + 1 + count))
    
    def merge_enrichments(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge all enrichments with existing data.
        
        Returns:
            Tuple of (enriched_data_df, enriched_impact_links_df)
        """
        if self.data_df is None or self.impact_links_df is None:
            self.load_existing_data()
        
        self.logger.info(f"Merging enrichments: {len(self.new_observations)} observations, "
                        f"{len(self.new_events)} events, {len(self.new_impact_links)} impact links")
        
        # Merge observations
        if self.new_observations:
            new_obs_df = pd.DataFrame(self.new_observations)
            if 'id' not in new_obs_df.columns:
                max_id = self.data_df['id'].max() if 'id' in self.data_df.columns else 0
                new_obs_df['id'] = self._generate_ids(len(new_obs_df), max_id)
            self.data_df = pd.concat([self.data_df, new_obs_df], ignore_index=True)
            self.logger.info(f"Added {len(new_obs_df)} observations")
        
        # Merge events
        if self.new_events:
            new_events_df = pd.DataFrame(self.new_events)
            if 'id' not in new_events_df.columns:
                max_id = self.data_df['id'].max() if 'id' in self.data_df.columns else 0
                new_events_df['id'] = self._generate_ids(len(new_events_df), max_id)
            self.data_df = pd.concat([self.data_df, new_events_df], ignore_index=True)
            self.logger.info(f"Added {len(new_events_df)} events")
        
        # Merge impact links
        if self.new_impact_links:
            new_links_df = pd.DataFrame(self.new_impact_links)
            self.impact_links_df = pd.concat([self.impact_links_df, new_links_df], ignore_index=True)
            self.logger.info(f"Added {len(new_links_df)} impact links")
        
        return self.data_df, self.impact_links_df
    
    def save_enriched_data(self, output_file: Path) -> None:
        """
        Save enriched dataset to Excel file.
        
        Args:
            output_file: Path to save the enriched dataset
        """
        if self.data_df is None or self.impact_links_df is None:
            self.merge_enrichments()
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving enriched dataset to {output_file}")
        
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.data_df.to_excel(writer, sheet_name='data', index=False)
                self.impact_links_df.to_excel(writer, sheet_name='impact_links', index=False)
            
            self.logger.info(f"Successfully saved enriched dataset: {len(self.data_df)} records, "
                           f"{len(self.impact_links_df)} impact links")
            
        except Exception as e:
            self.logger.error(f"Error saving enriched dataset: {e}")
            raise
    
    def get_enrichment_summary(self) -> Dict[str, Any]:
        """Get summary of enrichments added."""
        summary = {
            'new_observations': len(self.new_observations),
            'new_events': len(self.new_events),
            'new_impact_links': len(self.new_impact_links),
            'total_new_records': len(self.new_observations) + len(self.new_events) + len(self.new_impact_links)
        }
        
        self.logger.info(f"Enrichment summary: {summary}")
        return summary


class Task1DataProcessor:
    """Main class that combines exploration and enrichment functionality."""
    
    def __init__(self, data_file: Path, reference_codes_file: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the Task1DataProcessor.
        
        Args:
            data_file: Path to the main data Excel file
            reference_codes_file: Path to the reference codes Excel file
            logger: Optional logger instance
        """
        self.data_file = Path(data_file)
        self.reference_codes_file = Path(reference_codes_file)
        self.logger = logger or self._setup_logger()
        
        self.explorer = DataExplorer(self.data_file, self.reference_codes_file, self.logger)
        self.enricher = DataEnricher(self.data_file, self.logger)
        
        self.logger.info("Initialized Task1DataProcessor")
    
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
    
    def run_full_exploration(self) -> Dict[str, Any]:
        """Run complete data exploration and return all results."""
        self.logger.info("Starting full data exploration...")
        
        results = {
            'schema_info': self.explorer.get_schema_info(),
            'missing_values': self.explorer.get_missing_values(),
            'record_type_counts': self.explorer.count_by_record_type().to_dict(),
            'pillar_counts': self.explorer.count_by_pillar().to_dict(),
            'source_type_counts': self.explorer.count_by_source_type().to_dict(),
            'confidence_counts': self.explorer.count_by_confidence().to_dict(),
            'temporal_range': self.explorer.get_temporal_range(),
            'unique_indicators': self.explorer.get_unique_indicators().to_dict(),
            'indicator_coverage': self.explorer.get_indicator_coverage_by_pillar().to_dict('records'),
            'events_catalog': len(self.explorer.get_events_catalog()),
            'events_by_category': self.explorer.get_events_by_category().to_dict(),
            'impact_links_summary': self.explorer.get_impact_links_summary(),
            'data_quality': self.explorer.validate_data_quality(),
            'summary_statistics': self.explorer.get_summary_statistics(),
        }
        
        self.logger.info("Full exploration complete")
        return results
    
    def enrich_and_save(self, output_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge enrichments and save to file.
        
        Args:
            output_file: Path to save enriched dataset
            
        Returns:
            Tuple of (enriched_data_df, enriched_impact_links_df)
        """
        self.logger.info("Starting enrichment and save process...")
        
        enriched_data, enriched_links = self.enricher.merge_enrichments()
        self.enricher.save_enriched_data(output_file)
        
        summary = self.enricher.get_enrichment_summary()
        self.logger.info(f"Enrichment complete: {summary}")
        
        return enriched_data, enriched_links
