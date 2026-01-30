# Ethiopia Financial Inclusion Forecasting

Project for forecasting financial inclusion in Ethiopia.

## Project Structure

```
├── .github/workflows/
│   └── unittests.yml
├── data/
│   ├── raw/                      # Starter dataset
│   │   ├── ethiopia_fi_unified_data.csv
│   │   └── reference_codes.csv
│   └── processed/                # Analysis-ready data
├── notebooks/
│   └── README.md
├── src/
│   ├── __init__.py
├── dashboard/
│   └── app.py
├── tests/
│   └── __init__.py
├── models/
├── reports/
│   └── figures/
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

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

### Running Tests

```bash
pytest tests/
```

## License

[Add your license here]
