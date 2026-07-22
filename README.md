# Food Inflation Forecasting MVP

This repository is a phase-one hackathon submission for forecasting Botswana food inflation over the next 12 months using economic and trade indicators from the provided CSV datasets.

## Phase 1 objective

The project is designed to produce a credible baseline forecast and a documented, explainable feature-engineering workflow. The core story is that global trade and oil shocks affect Botswana food inflation through a delayed transmission chain involving shipping conditions, import costs, policy conditions, and retail inflation persistence.

## What the workflow does

- loads the raw source datasets from the repository
- converts the daily Baltic Dry Index into richer monthly signals instead of relying only on a monthly mean
- creates lagged economic features for the transmission mechanism from global input pressures to domestic inflation
- trains baseline models for short-horizon inflation forecasting
- offers a judge-facing driver selector so the same pipeline can be interpreted through either a logistics lens (`bdi`) or an energy-cost lens (`brent`)
- writes outputs for the submission package, including a forecast CSV and PDF reports

## Datasets used

The current implementation integrates variables from multiple datasets, including:

- Baltic Dry Index daily series for shipping and supply chain stress
- Brent crude monthly series for energy import pressure
- Botswana policy rate series for domestic macro conditions
- FAO Botswana price series for general and food CPI
- HCP indicators as linkage features for the human-capital memo

This breadth is important because the scoring rubric rewards cross-dataset integration and economic reasoning.

## Feature engineering design

The feature engineering workflow is intentionally more detailed than a naive month-average merge. For the daily BDI data, the pipeline extracts:

- monthly mean
- monthly volatility
- monthly range
- coefficient of variation
- month-to-month return
- monthly directional change
- extreme-day count
- rolling 3-month summary signals

These features are then merged with Brent prices, policy rates, and food index data. Lagged versions of the key variables are created at 1, 3, 6, and 12 months to reflect delayed pass-through from global variables to domestic food inflation.

## Phase 1 submission deliverables

The repo already generates the key artifacts expected for the submission:

- 12-month forecast CSV in the [outputs](outputs) folder
- feature engineering report PDF in the [reports](reports) folder
- HCP linkage memo PDF in the [reports](reports) folder
- evaluation plots in the [figures](figures) folder

For judge review, the workflow now supports three focus modes:

- `all` — combined baseline using both shipping and oil context
- `bdi` — logistics-focused view for short-run import and shipping transmission analysis
- `brent` — oil-cost-focused view for longer-term energy price pass-through analysis

When a judge selects `bdi`, the model emphasises the BDI-derived shipping and trade-stress variables. When they select `brent`, the run shifts focus toward Brent oil price and its lagged pass-through structure for longer-horizon macro analysis.

## Repository structure

- [src/preprocessing.py](src/preprocessing.py) — data loading, feature engineering, and lag construction
- [src/models.py](src/models.py) — SARIMAX and LSTM baseline training
- [src/evaluation.py](src/evaluation.py) — output generation and report production
- [app.py](app.py) — browser-based dashboard for running the workflow
- [src/tkinter_gui.py](src/tkinter_gui.py) — legacy desktop dashboard fallback
- [requirements.txt](requirements.txt) — pinned dependency list

## Environment setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the package dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Run the project

### Option 1: Use the Tkinter desktop dashboard

```bash
source .venv/bin/activate
python app.py
```

Use the dashboard to:

- upload a CSV file if needed
- choose the analysis focus from the dropdown (`all`, `bdi`, or `brent`)
- run the preprocessing stage
- train the models
- evaluate the results
- inspect the output log

### Option 2: Run the scripts directly

```bash
source .venv/bin/activate
python src/preprocessing.py
python src/models.py --driver bdi
python src/evaluation.py --driver bdi
```

You can swap `bdi` with `brent` or `all` depending on the analysis lens you want to present.

## Expected outputs

Running the workflow produces:

- forecast metrics in the terminal
- a 12-month prediction CSV in [outputs/predictions.csv](outputs/predictions.csv)
- visualisations in [figures](figures)
- PDF reports in [reports](reports)

## Hackathon submission note

This repository is intentionally scoped to a strong phase-one MVP. The emphasis is on a reproducible pipeline, transparent feature engineering, and a defensible forecast baseline rather than a complex production system.
