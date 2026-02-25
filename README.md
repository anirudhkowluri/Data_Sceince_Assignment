# Data Science Assignment: Market Sentiment Prediction from Trading Activity

This repository contains an end-to-end machine learning workflow that predicts **market sentiment classes** (e.g., Fear, Greed) from aggregated trading behavior.

The main analysis is implemented in a Jupyter notebook (`notebook_1.ipynb`) using:
- Data loading and cleaning
- Feature engineering from raw trade logs
- Dataset merge with a Fear & Greed index
- Multi-class classification using a Random Forest model
- Evaluation and visualization (confusion matrix, feature importance, correlation heatmap)

---

## Repository Structure

- `notebook_1.ipynb` — Primary analysis notebook containing the complete workflow.
- `csv_files/fear_greed_index.csv` — Sentiment index dataset (`timestamp`, `value`, `classification`, `date`).
- `csv_files/historical_data.csv` — Raw historical trade-level dataset.
- `outputs/Confusion_matrix.png` — Saved confusion matrix visualization.
- `outputs/Model_output.png` — Saved feature-importance plot.
- `outputs/Corelation matrix.png` — Saved feature-correlation heatmap.
- `ds_report.pdf` — Project report document.
- `notebook_link` — Google Colab link to run/view the notebook online.

---

## Problem Statement

The project aims to learn the relationship between daily aggregated trading activity and market sentiment category.  
Given daily trading metrics, the model predicts one of the sentiment classes:
- Fear
- Greed
- Extreme Fear
- Extreme Greed
- Neutral

---

## Data Summary

### 1) Fear & Greed Index
- File: `csv_files/fear_greed_index.csv`
- Rows: **2,644**
- Columns: `timestamp`, `value`, `classification`, `date`
- Class distribution:
  - Fear: 781
  - Greed: 633
  - Extreme Fear: 508
  - Neutral: 396
  - Extreme Greed: 326

### 2) Historical Trades
- File: `csv_files/historical_data.csv`
- Rows: **211,224**
- Contains trade/account metadata including side, price, size, fee, and timestamps.

---

## Methodology

The notebook workflow is:

1. **Import libraries** (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`).
2. **Load both datasets**.
3. **Clean and standardize dates**:
   - Convert sentiment `date` to datetime.
   - Parse trade timestamp (`Timestamp IST`) and derive a normalized `date` field.
4. **Aggregate trades to daily level**, creating features:
   - `total_volume`
   - `avg_price`
   - `net_pnl`
   - `trade_count`
   - `buy_trades`
   - `sell_trades`
   - `buy_sell_ratio`
5. **Merge** daily trade features with sentiment labels on `date`.
6. **Train/test split** with stratification.
7. **Scale features** using `StandardScaler`.
8. **Train model**: `RandomForestClassifier` with balanced class weights.
9. **Evaluate** via classification report and confusion matrix.
10. **Interpret** via feature-importance and correlation heatmaps.

---

## Model Configuration

The notebook trains a Random Forest with:
- `n_estimators=200`
- `max_depth=8`
- `class_weight='balanced'`
- `random_state=42`

Target variable: `classification` (multi-class sentiment label).

---

## Results Snapshot

From the notebook output:
- Test accuracy is approximately **0.40**.
- Performance varies by class, with better recall on **Extreme Fear** than on some other labels.

Detailed metrics are available directly in the notebook’s classification report output.

---

## How to Run

### Option A: Google Colab
Use the link in `notebook_link`.

### Option B: Local Jupyter
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open `notebook_1.ipynb` and run all cells in order.

> Note: In the notebook, data is loaded with absolute-style paths (`/fear_greed_index.csv`, `/historical_data.csv`).
> If running locally, update these to:
> - `csv_files/fear_greed_index.csv`
> - `csv_files/historical_data.csv`

---

## Potential Improvements

- Perform hyperparameter tuning (e.g., `GridSearchCV`/`RandomizedSearchCV`).
- Add time-aware validation to better reflect temporal dynamics.
- Engineer richer market microstructure features.
- Try additional models (XGBoost/LightGBM/CatBoost, calibrated linear baselines).
- Address class imbalance further with sampling strategies and threshold analysis.

---

## Notes

- Output plots are already included in the `outputs/` folder.
- A detailed report is available in `ds_report.pdf`.

