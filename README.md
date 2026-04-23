# AI Econ DAIOE Months V2

This project fetches and processes employment data from the SCB (Statistics Sweden) API.

## Setup

1. Ensure you have Python installed (>= 3.14 recommended).
2. Install dependencies (e.g., using `uv` or `pip`):
   ```bash
   uv sync
   ```

## Usage

Run the main script to fetch data and save it as a Parquet file:
```bash
python main.py
```

The output will be saved to `data/scb_months.parquet`.

## Project Structure

- `main.py`: Main script for data fetching and processing.
- `data/`: Directory where the processed data is stored.
- `scb_months_15_to_25.ipynb`: Original research notebook (prototype).
