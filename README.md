# Infectious Disease Forecasting

A locally run infectious disease case-count forecasting project built around a SARIMA, Fourier seasonal regression, and neural-network ensemble workflow.

The project is designed for public health analysts who want to upload daily, weekly, quarterly, or yearly case-count data and generate reproducible forecasts from a command line or local Flask web app.

## Features

- Load `.csv`, `.txt`, `.tsv`, `.xlsx`, and `.xls` files
- Clean dates and case-count columns
- Aggregate and resample to daily, weekly, quarterly, or yearly time series
- Handle missing dates and missing values
- Optional robust outlier smoothing
- Forecast with:
  - SARIMA / SARIMAX
  - Fourier seasonal regression
  - scikit-learn MLP neural network
- Combine forecasts with weighted mean, mean, or median ensemble methods
- Export forecast outputs to CSV, Excel, and PNG
- Run from the command line or a local web interface
- Extend with modular data providers for CDC, ADHS, or local files

## Project structure

```text
infectious-disease-forecasting/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
├── run_forecast.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── ensemble.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── pipeline.py
│   └── utils.py
├── data_sources/
│   ├── __init__.py
│   ├── base_provider.py
│   ├── cdc_provider.py
│   ├── adhs_provider.py
│   └── local_file_provider.py
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── app.js
├── examples/
│   └── sample_case_data.csv
├── outputs/
│   ├── forecasts/
│   ├── plots/
│   └── uploads/
├── tests/
│   └── test_forecasting_pipeline.py
└── docs/
    └── original_forecasting_suite.py
```

## Installation

### Option 1: pip

```bash
git clone <your-repo-url>
cd infectious-disease-forecasting
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: conda

```bash
git clone <your-repo-url>
cd infectious-disease-forecasting
conda env create -f environment.yml
conda activate infectious-disease-forecasting
```

## Required input data format

At minimum, your file needs:

| column | description |
|---|---|
| date | Date of event, report, week end, quarter, or year |
| cases | Case count or other numeric outcome |

Optional columns:

| column | description |
|---|---|
| disease | Disease, pathogen, subtype, or condition |
| geography | County, region, state, facility, or other spatial unit |

Example:

```csv
date,cases,disease,county
2024-10-06,13,Influenza A,Example County
2024-10-13,18,Influenza A,Example County
```

## Run the command-line pipeline

```bash
python run_forecast.py \
  --input examples/sample_case_data.csv \
  --date-col date \
  --value-col cases \
  --frequency weekly \
  --horizon 12 \
  --seasonal-period 52
```

Outputs are written to:

```text
outputs/forecasts/<input_name>_forecast.csv
outputs/forecasts/<input_name>_forecast.xlsx
outputs/plots/<input_name>_forecast.png
```

To run only Fourier and MLP models, which is faster for testing:

```bash
python run_forecast.py \
  --input examples/sample_case_data.csv \
  --date-col date \
  --value-col cases \
  --no-sarima
```

## Run the local web app

```bash
python app/app.py
```

Open your browser to:

```text
http://127.0.0.1:5000
```

Use the form to upload a file, set the date/case-count columns, choose frequency and model settings, and download the forecast CSV, Excel workbook, or PNG plot.

## Model methodology

### SARIMA

The SARIMA model uses `statsmodels` SARIMAX with a small candidate grid. Models are scored on a holdout window and then refit on the full series before forecasting. A log1p transform is enabled by default to stabilize variance in count data.

### Fourier seasonal regression

The Fourier model estimates sine/cosine seasonal terms using least squares. This is useful as a transparent baseline for recurring annual or weekly seasonal patterns.

### MLP neural network

The MLP model converts the time series into lagged windows and trains a lightweight `scikit-learn` neural network. It recursively predicts future periods.

### Ensemble

Forecasts are combined using a weighted mean by default. Mean and median ensemble options are also available. The ensemble interval is based on cross-model forecast spread and should be treated as a heuristic interval, not a formal probabilistic confidence interval.

## Data providers

Provider classes live in `data_sources/`.

- `LocalFileProvider`: reads user-provided files
- `CDCDataGovProvider`: pulls public rows from `data.cdc.gov` Socrata datasets by resource ID
- `CDCWonderProvider`: explicit placeholder for database-specific CDC WONDER XML workflows
- `ADHSPublicFileProvider`: reads direct public ADHS CSV/XLS/XLSX download URLs

Example CDC data.gov usage:

```python
from data_sources.cdc_provider import CDCDataGovProvider

provider = CDCDataGovProvider()
result = provider.fetch(
    resource_id="vbim-akqf",
    limit=1000,
)
print(result.data.head())
```

## Adding a new data provider

1. Create a new provider in `data_sources/`.
2. Subclass `BaseDataProvider`.
3. Implement `fetch()` and return a `ProviderResult`.
4. Standardize the output so it contains at least one date column and one numeric value column.
5. Add validation and provider-specific documentation.

## Known limitations

- Forecasts are only as reliable as the input surveillance data.
- Report delays, changes in testing behavior, case definition changes, and missing reporting periods can degrade forecasts.
- SARIMA can be slow or unstable on short or sparse time series.
- The MLP model is lightweight and intended as a baseline, not a fully tuned deep-learning model.
- Ensemble intervals are based on model spread and are not calibrated prediction intervals.
- ADHS dashboards may not expose stable direct CSV APIs; use direct downloadable public files when available.

## Future improvements

- Add calibrated probabilistic intervals with conformal prediction or bootstrapping
- Add backtesting over multiple rolling forecast origins
- Add Poisson/negative binomial generalized additive models
- Add holiday, school calendar, wastewater, hospitalization, and climate covariates
- Add automated model cards and forecast quality flags
- Add Dockerfile and GitHub Actions CI
- Add optional FastAPI backend and richer front-end column mapping

## Development

Run tests:

```bash
pytest
```
