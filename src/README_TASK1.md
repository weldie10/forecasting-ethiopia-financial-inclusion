# Task 1: Data Exploration and Enrichment - Usage Guide

This module provides a comprehensive OOP-based solution for exploring and enriching the Ethiopia Financial Inclusion dataset.

## Architecture

The module consists of three main classes:

1. **`DataExplorer`**: Handles data exploration and analysis
2. **`DataEnricher`**: Handles adding new records (observations, events, impact links)
3. **`Task1DataProcessor`**: Main class that combines both functionalities

## Quick Start

### In Jupyter Notebook

```python
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path('../src').resolve()))

from task1_data_exploration import (
    Task1DataProcessor,
    ObservationRecord,
    EventRecord,
    ImpactLinkRecord,
    ConfidenceLevel,
    ImpactDirection
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize processor
processor = Task1DataProcessor(
    data_file=Path('../data/raw/ethiopia_fi_unified_data.xlsx'),
    reference_codes_file=Path('../data/raw/reference_codes.xlsx')
)

# Run full exploration
results = processor.run_full_exploration()
```

### Data Exploration

```python
# Get schema information
schema_info = processor.explorer.get_schema_info()

# Count records by type
record_counts = processor.explorer.count_by_record_type()

# Get temporal range
temporal_info = processor.explorer.get_temporal_range()

# Get unique indicators
indicators = processor.explorer.get_unique_indicators()

# Get events catalog
events = processor.explorer.get_events_catalog()

# Validate data quality
quality_issues = processor.explorer.validate_data_quality()

# Get comprehensive summary
summary = processor.explorer.get_summary_statistics()
```

### Data Enrichment

```python
# Add a new observation
observation = ObservationRecord(
    pillar='access',
    indicator='Account ownership',
    indicator_code='ACC_001',
    value_numeric=45.2,
    observation_date='2023-01-01',
    source_name='World Bank Findex',
    source_url='https://example.com',
    confidence=ConfidenceLevel.HIGH.value,
    collected_by='Your Name',
    original_text='45.2% of adults in Ethiopia have an account',
    notes='Important baseline for access pillar'
)
processor.enricher.add_observation(observation)

# Add a new event
event = EventRecord(
    category='policy',
    event_date='2023-06-15',
    source_name='National Bank of Ethiopia',
    source_url='https://example.com',
    confidence=ConfidenceLevel.HIGH.value,
    collected_by='Your Name',
    original_text='New digital payment policy announced',
    notes='Could significantly impact access and usage'
)
processor.enricher.add_event(event)

# Add an impact link
impact_link = ImpactLinkRecord(
    parent_id='EVENT_123',  # ID of the event
    pillar='access',
    related_indicator='ACC_001',
    impact_direction=ImpactDirection.POSITIVE.value,
    impact_magnitude=5.0,
    lag_months=6,
    evidence_basis='Historical analysis',
    source_name='Research Paper',
    source_url='https://example.com',
    confidence=ConfidenceLevel.MEDIUM.value,
    collected_by='Your Name',
    notes='Policy expected to increase account ownership'
)
processor.enricher.add_impact_link(impact_link)

# Save enriched dataset
enriched_data, enriched_links = processor.enrich_and_save(
    output_file=Path('../data/processed/ethiopia_fi_unified_data_enriched.xlsx')
)
```

## Features

- **OOP Design**: Clean, reusable class-based architecture
- **Logging**: Comprehensive logging for all operations
- **Type Safety**: Uses dataclasses and enums for type safety
- **Error Handling**: Proper error handling and validation
- **Flexibility**: Easy to extend and customize

## Data Classes

- `ObservationRecord`: For observation records
- `EventRecord`: For event records (pillar should be None)
- `ImpactLinkRecord`: For impact link records

## Enums

- `ConfidenceLevel`: HIGH, MEDIUM, LOW
- `RecordType`: OBSERVATION, EVENT, TARGET
- `ImpactDirection`: POSITIVE, NEGATIVE, NEUTRAL

## Logging

All operations are logged with appropriate levels:
- INFO: Normal operations
- WARNING: Potential issues
- ERROR: Errors that need attention

Logs include timestamps, module names, and detailed messages.
