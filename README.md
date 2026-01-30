# Ethiopia Financial Inclusion Forecasting

Project for forecasting financial inclusion in Ethiopia using comprehensive data exploration, analysis, and modeling.

## Project Structure

```
├── .github/workflows/
│   └── unittests.yml              # CI/CD workflow for unit tests
├── data/
│   ├── raw/                       # Starter dataset
│   │   ├── ethiopia_fi_unified_data.xlsx
│   │   └── reference_codes.xlsx
│   └── processed/                 # Analysis-ready data
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Task 1: Data exploration
│   ├── 02_data_enrichment.ipynb   # Task 1: Data enrichment
│   └── 03_eda.ipynb               # Task 2: Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── task1_data_exploration.py  # Task 1: OOP data exploration & enrichment
│   └── task2_eda.py                # Task 2: OOP exploratory data analysis
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── tests/
│   └── __init__.py
├── models/                         # Trained models
├── reports/
│   └── figures/                    # Generated visualizations
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Task 1: Data Exploration and Enrichment

The project includes an OOP-based module for exploring and enriching the dataset.

#### Quick Start

```python
import sys
from pathlib import Path
import logging

sys.path.append(str(Path('src').resolve()))
from task1_data_exploration import (
    Task1DataProcessor,
    ObservationRecord,
    EventRecord,
    ImpactLinkRecord,
    ConfidenceLevel
)

# Initialize processor
processor = Task1DataProcessor(
    data_file=Path('data/raw/ethiopia_fi_unified_data.xlsx'),
    reference_codes_file=Path('data/raw/reference_codes.xlsx')
)

# Run full exploration
results = processor.run_full_exploration()

# Add new observation
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
    original_text='45.2% of adults have accounts',
    notes='Important baseline'
)
processor.enricher.add_observation(observation)

# Save enriched dataset
processor.enrich_and_save(Path('data/processed/enriched.xlsx'))
```

#### Features

- **DataExplorer**: Comprehensive data exploration and analysis
- **DataEnricher**: Add new observations, events, and impact links
- **Type Safety**: Dataclasses and enums for type safety
- **Logging**: Comprehensive logging throughout
- **Jupyter Notebooks**: Interactive exploration and enrichment workflows

#### Notebooks

- `notebooks/01_data_exploration.ipynb`: Complete data exploration workflow
- `notebooks/02_data_enrichment.ipynb`: Data enrichment with examples

### Task 2: Exploratory Data Analysis

Comprehensive EDA module for analyzing financial inclusion patterns in Ethiopia.

#### Quick Start

```python
import sys
from pathlib import Path
import logging

sys.path.append(str(Path('src').resolve()))
from task2_eda import ExploratoryDataAnalyzer

# Initialize analyzer
eda = ExploratoryDataAnalyzer(
    data_file=Path('data/raw/ethiopia_fi_unified_data.xlsx'),
    reference_codes_file=Path('data/raw/reference_codes.xlsx'),
    figure_dir=Path('reports/figures')
)

# Run full EDA pipeline
results = eda.run_full_eda()
```

#### Features

- **Dataset Overview**: Summarize by record_type, pillar, source_type, temporal coverage
- **Access Analysis**: Account ownership trajectory, growth rates, gender gap, urban/rural
- **Usage Analysis**: Mobile money trends, digital payments, registered vs active gap
- **Infrastructure Analysis**: 4G coverage, mobile penetration, ATM density
- **Event Timeline**: Visualize events and overlay on indicator trends
- **Correlation Analysis**: Examine relationships between indicators
- **Insight Generation**: Automatic insight generation and data quality assessment

#### Analysis Capabilities

1. **Dataset Overview**
   - Record type, pillar, and source type distributions
   - Temporal coverage visualization
   - Data gaps identification
   - Confidence level assessment

2. **Access Analysis**
   - Account ownership trajectory (2011-2024)
   - Growth rate calculations between survey years
   - Gender gap analysis (male vs female)
   - Urban vs rural comparison
   - 2021-2024 slowdown investigation

3. **Usage Analysis**
   - Mobile money account penetration (2014-2024)
   - Digital payment adoption patterns
   - Registered vs active account gap
   - Payment use cases (P2P, merchant, bill pay, wages)

4. **Infrastructure and Enablers**
   - 4G coverage analysis
   - Mobile penetration trends
   - ATM density analysis
   - Relationships with inclusion outcomes
   - Leading indicators identification

5. **Event Timeline**
   - Timeline visualization of all cataloged events
   - Overlay events on indicator trend charts
   - Identify relationships (e.g., Telebirr launch, M-Pesa entry, Safaricom entry)

6. **Correlation Analysis**
   - Correlation matrix between indicators
   - Identify factors associated with Access and Usage
   - Analyze existing impact_link records

#### Notebook

- `notebooks/03_eda.ipynb`: Complete EDA workflow with all visualizations

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

### Running Tests

```bash
pytest tests/
```

## Key Insights

### Factors Driving Financial Inclusion

- Mobile money infrastructure expansion
- Policy interventions and regulatory changes
- Product launches and market entries
- Infrastructure development (4G, mobile penetration)

### Account Ownership Stagnation (2021-2024)

Despite 65M+ mobile money accounts being opened, account ownership grew only +3pp. Potential factors:

- Registered vs. active account gap
- Limited usage beyond registration
- Financial literacy barriers
- Trust and security concerns
- Infrastructure gaps in rural areas

### Data Gaps

- Sparse indicator coverage (<5 observations for some indicators)
- Limited disaggregated data (gender, urban/rural)
- Missing infrastructure time series data

## Architecture

### OOP Design

Both Task 1 and Task 2 modules follow object-oriented design principles:

- **Separation of Concerns**: Dedicated classes for different functionalities
- **Reusability**: Modular design for easy extension
- **Type Safety**: Dataclasses and enums
- **Logging**: Comprehensive logging throughout
- **Error Handling**: Graceful handling of missing data

### Task 1 Classes

- `Task1DataProcessor`: Main processor combining exploration and enrichment
- `DataExplorer`: Data exploration and analysis
- `DataEnricher`: Data enrichment operations
- `ObservationRecord`, `EventRecord`, `ImpactLinkRecord`: Type-safe data classes

### Task 2 Classes

- `ExploratoryDataAnalyzer`: Comprehensive EDA operations
- `EDAResults`: Container for all analysis results

## Output

### Visualizations

All visualizations are automatically saved to `reports/figures/` with high resolution (300 DPI):

- Temporal coverage charts
- Account ownership trajectory plots
- Gender gap visualizations
- Urban/rural comparisons
- Mobile money trends
- Event timeline visualizations
- Correlation heatmaps
- Event overlay charts

### Data Files

- Enriched datasets saved to `data/processed/`
- All figures saved to `reports/figures/`

## Logging

All modules include comprehensive logging:

- **INFO**: Normal operations
- **WARNING**: Potential issues or missing data
- **ERROR**: Errors that need attention
- **DEBUG**: Detailed debugging information

Logs include timestamps, module names, and detailed messages.

## Development

### Branch Structure

- `main`: Main development branch
- `task-1`: Task 1 implementation branch
- `task-2`: Task 2 implementation branch

### Commit Messages

Follow conventional commit format:
- `feat(task1): Description` for new features
- `fix(task2): Description` for bug fixes
- `docs: Description` for documentation

## Requirements

See `requirements.txt` for full list. Key dependencies:

- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- plotly >= 5.14.0
- scikit-learn >= 1.2.0
- streamlit >= 1.28.0
- pytest >= 7.2.0
- jupyter >= 1.0.0

## License

[Add your license here]
