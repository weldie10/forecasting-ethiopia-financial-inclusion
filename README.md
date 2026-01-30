# Ethiopia Financial Inclusion Forecasting

Project for forecasting financial inclusion in Ethiopia using comprehensive data exploration, analysis, modeling, and interactive visualization.

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
│   ├── 03_eda.ipynb               # Task 2: Exploratory Data Analysis
│   ├── 04_impact_modeling.ipynb   # Task 3: Event Impact Modeling
│   └── 05_forecasting.ipynb       # Task 4: Forecasting Access and Usage
├── src/
│   ├── __init__.py
│   ├── task1_data_exploration.py  # Task 1: OOP data exploration & enrichment
│   ├── task2_eda.py                # Task 2: OOP exploratory data analysis
│   ├── task3_impact_modeling.py    # Task 3: OOP event impact modeling
│   ├── task4_forecasting.py         # Task 4: OOP forecasting module
│   └── dashboard_components.py      # Task 5: Dashboard components
├── dashboard/
│   └── app.py                      # Streamlit dashboard application
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

### Task 5: Dashboard Development (Main Interface)

Interactive Streamlit dashboard for exploring data, understanding event impacts, and viewing forecasts.

#### Quick Start

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

#### Dashboard Pages

**1. Overview Page:**
- Key metrics summary cards (current values, trends)
- P2P/ATM Crossover Ratio indicator
- Growth rate highlights
- Quick insights and data download

**2. Trends Page:**
- Interactive time series plots with Plotly
- Date range selector
- Pillar filter (Access, Usage, Quality)
- Channel comparison view
- Event timeline overlay
- Filtered data download

**3. Forecasts Page:**
- Forecast visualizations with confidence intervals
- Model selection (Linear Trend, Event-Augmented, All Models)
- Key projected milestones
- Forecast table with confidence intervals
- Forecast data download

**4. Inclusion Projections Page:**
- Financial inclusion rate projections
- Progress toward target visualization (default 60%)
- Scenario selector (Optimistic, Base, Pessimistic)
- Gap analysis to target
- Scenario comparison charts
- Answers to consortium's key questions

#### Technical Features

- **OOP Architecture**: `DashboardDataLoader` and `DashboardVisualizations` classes
- **Interactive Visualizations**: Plotly charts with hover, zoom, and pan
- **Data Download**: CSV export functionality on all pages
- **Caching**: Efficient data loading with Streamlit caching
- **Responsive Design**: Wide layout optimized for data visualization
- **Logging**: Comprehensive logging throughout

#### Dashboard Components

- `src/dashboard_components.py`: OOP-based dashboard components
- `dashboard/app.py`: Main Streamlit application

### Task 1: Data Exploration and Enrichment

OOP-based module for exploring and enriching the dataset.

```python
from task1_data_exploration import Task1DataProcessor, ObservationRecord, ConfidenceLevel

processor = Task1DataProcessor(
    data_file=Path('data/raw/ethiopia_fi_unified_data.xlsx'),
    reference_codes_file=Path('data/raw/reference_codes.xlsx')
)

# Run exploration
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
```

**Notebooks**: `01_data_exploration.ipynb`, `02_data_enrichment.ipynb`

### Task 2: Exploratory Data Analysis

Comprehensive EDA module for analyzing financial inclusion patterns.

```python
from task2_eda import ExploratoryDataAnalyzer

eda = ExploratoryDataAnalyzer(
    data_file=Path('data/raw/ethiopia_fi_unified_data.xlsx'),
    figure_dir=Path('reports/figures')
)

# Run full EDA
results = eda.run_full_eda()
```

**Notebook**: `03_eda.ipynb`

### Task 3: Event Impact Modeling

Model how events affect financial inclusion indicators.

```python
from task3_impact_modeling import EventImpactModeler

modeler = EventImpactModeler(
    data_file=Path('data/raw/ethiopia_fi_unified_data.xlsx')
)

# Build association matrix
matrix = modeler.build_association_matrix()

# Validate against historical data
validation = modeler.validate_against_historical(
    event_name='Telebirr',
    indicator_code='ACC_MM_ACCOUNT',
    observed_before=4.7,
    observed_after=9.45,
    event_date=datetime(2021, 5, 1)
)
```

**Notebook**: `04_impact_modeling.ipynb`

### Task 4: Forecasting Access and Usage

Forecast Account Ownership and Digital Payment Usage for 2025-2027.

```python
from task4_forecasting import FinancialInclusionForecaster

forecaster = FinancialInclusionForecaster(
    data_file=Path('data/raw/ethiopia_fi_unified_data.xlsx'),
    impact_model_file=Path('data/processed/association_matrix.csv')
)

# Generate forecast
series = forecaster.extract_indicator_series('ACC_OWNERSHIP', pillar='access')
forecast = forecaster.linear_trend_forecast(series, [2025, 2026, 2027])

# Generate scenarios
scenarios = forecaster.scenario_forecast(forecast)
```

**Notebook**: `05_forecasting.ipynb`

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

## Running Tests

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

Despite 65M+ mobile money accounts, account ownership grew only +3pp. Potential factors:
- Registered vs. active account gap
- Limited usage beyond registration
- Financial literacy barriers
- Trust and security concerns

## Architecture

All modules follow OOP design principles with comprehensive logging:
- **Task 1**: `Task1DataProcessor`, `DataExplorer`, `DataEnricher`
- **Task 2**: `ExploratoryDataAnalyzer`
- **Task 3**: `EventImpactModeler`
- **Task 4**: `FinancialInclusionForecaster`
- **Task 5**: `DashboardDataLoader`, `DashboardVisualizations`

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
