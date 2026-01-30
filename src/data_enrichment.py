"""
Data enrichment module for Task 1.
Functions to add new observations, events, and impact_links to the dataset.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def create_new_observation(
    pillar: str,
    indicator: str,
    indicator_code: str,
    value_numeric: float,
    observation_date: str,
    source_name: str,
    source_url: str,
    confidence: str,
    collected_by: str,
    original_text: str,
    notes: str,
    **kwargs
) -> Dict:
    """
    Create a new observation record following the schema.
    
    Args:
        pillar: The pillar (e.g., 'access', 'usage', 'quality')
        indicator: Indicator name
        indicator_code: Unique indicator code
        value_numeric: Numeric value
        observation_date: Date of observation (YYYY-MM-DD)
        source_name: Name of the source
        source_url: URL where data was found
        confidence: Confidence level (high/medium/low)
        collected_by: Name of person who collected the data
        original_text: Exact quote or figure from source
        notes: Why this data is useful
        **kwargs: Additional fields to include
    
    Returns:
        Dictionary representing the observation record
    """
    record = {
        'record_type': 'observation',
        'pillar': pillar,
        'indicator': indicator,
        'indicator_code': indicator_code,
        'value_numeric': value_numeric,
        'observation_date': observation_date,
        'source_name': source_name,
        'source_url': source_url,
        'confidence': confidence,
        'collected_by': collected_by,
        'collection_date': datetime.now().strftime('%Y-%m-%d'),
        'original_text': original_text,
        'notes': notes,
        **kwargs
    }
    return record


def create_new_event(
    category: str,
    event_date: str,
    source_name: str,
    source_url: str,
    confidence: str,
    collected_by: str,
    original_text: str,
    notes: str,
    **kwargs
) -> Dict:
    """
    Create a new event record following the schema.
    Note: pillar should be left empty for events.
    
    Args:
        category: Event category (e.g., 'policy', 'product_launch', 'infrastructure')
        event_date: Date of event (YYYY-MM-DD)
        source_name: Name of the source
        source_url: URL where data was found
        confidence: Confidence level (high/medium/low)
        collected_by: Name of person who collected the data
        original_text: Exact quote or figure from source
        notes: Why this data is useful
        **kwargs: Additional fields to include
    
    Returns:
        Dictionary representing the event record
    """
    record = {
        'record_type': 'event',
        'pillar': None,  # Events should have empty pillar
        'category': category,
        'event_date': event_date,
        'source_name': source_name,
        'source_url': source_url,
        'confidence': confidence,
        'collected_by': collected_by,
        'collection_date': datetime.now().strftime('%Y-%m-%d'),
        'original_text': original_text,
        'notes': notes,
        **kwargs
    }
    return record


def create_new_impact_link(
    parent_id: str,
    pillar: str,
    related_indicator: str,
    impact_direction: str,
    impact_magnitude: Optional[float],
    lag_months: Optional[int],
    evidence_basis: str,
    source_name: str,
    source_url: str,
    confidence: str,
    collected_by: str,
    notes: str,
    **kwargs
) -> Dict:
    """
    Create a new impact_link record following the schema.
    
    Args:
        parent_id: ID of the parent event this links to
        pillar: The pillar affected
        related_indicator: Indicator code that is affected
        impact_direction: Direction of impact (positive/negative/neutral)
        impact_magnitude: Magnitude of impact (optional)
        lag_months: Months after event that impact occurs (optional)
        evidence_basis: Basis for the relationship
        source_name: Name of the source
        source_url: URL where data was found
        confidence: Confidence level (high/medium/low)
        collected_by: Name of person who collected the data
        notes: Why this relationship is useful
        **kwargs: Additional fields to include
    
    Returns:
        Dictionary representing the impact_link record
    """
    record = {
        'parent_id': parent_id,
        'pillar': pillar,
        'related_indicator': related_indicator,
        'impact_direction': impact_direction,
        'impact_magnitude': impact_magnitude,
        'lag_months': lag_months,
        'evidence_basis': evidence_basis,
        'source_name': source_name,
        'source_url': source_url,
        'confidence': confidence,
        'collected_by': collected_by,
        'collection_date': datetime.now().strftime('%Y-%m-%d'),
        'notes': notes,
        **kwargs
    }
    return record


def add_enrichments_to_dataset(
    data_file: Path,
    new_observations: List[Dict],
    new_events: List[Dict],
    impact_links_file: Path,
    new_impact_links: List[Dict],
    output_file: Optional[Path] = None
) -> tuple:
    """
    Add new enrichments to the existing dataset.
    
    Args:
        data_file: Path to the original data Excel file
        new_observations: List of new observation dictionaries
        new_events: List of new event dictionaries
        impact_links_file: Path to the original impact_links (same file, different sheet)
        new_impact_links: List of new impact_link dictionaries
        output_file: Optional path to save enriched dataset
    
    Returns:
        Tuple of (enriched_data_df, enriched_impact_links_df)
    """
    # Load existing data
    data_df = pd.read_excel(data_file, sheet_name='data')
    impact_links_df = pd.read_excel(impact_links_file, sheet_name='impact_links')
    
    # Convert new records to DataFrames
    if new_observations:
        new_obs_df = pd.DataFrame(new_observations)
        # Generate IDs for new observations if needed
        if 'id' not in new_obs_df.columns:
            max_id = data_df['id'].max() if 'id' in data_df.columns else 0
            new_obs_df['id'] = range(max_id + 1, max_id + 1 + len(new_obs_df))
        data_df = pd.concat([data_df, new_obs_df], ignore_index=True)
    
    if new_events:
        new_events_df = pd.DataFrame(new_events)
        # Generate IDs for new events if needed
        if 'id' not in new_events_df.columns:
            max_id = data_df['id'].max() if 'id' in data_df.columns else 0
            new_events_df['id'] = range(max_id + 1, max_id + 1 + len(new_events_df))
        data_df = pd.concat([data_df, new_events_df], ignore_index=True)
    
    if new_impact_links:
        new_links_df = pd.DataFrame(new_impact_links)
        impact_links_df = pd.concat([impact_links_df, new_links_df], ignore_index=True)
    
    # Save if output file specified
    if output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            data_df.to_excel(writer, sheet_name='data', index=False)
            impact_links_df.to_excel(writer, sheet_name='impact_links', index=False)
    
    return data_df, impact_links_df
