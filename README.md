# Understanding European Social Trends Through Machine Learning
## Predicting Voter Turnout Across Time and the East-West Divide

Tree-based machine learning models applied to European Social Survey (ESS) data from 2004 and 2023 to predict voter turnout and compare Eastern and Western Europe.

---

## Data

This project uses the **European Social Survey (ESS)**, a cross-national survey conducted across Europe. The data be found in the repository.

| File | Round | Year | Description |
|---|---|---|---|
| `ESS2.csv` | Round 2 | 2004 | Place in `data/raw/` |
| `ESS11.csv` | Round 11 | 2023 | Place in `data/raw/` |
| `ESS2e03_6 codebook.html` | Round 2 | 2004 | Place in `codebooks/` |
| `ESS11e04_1 codebook.html` | Round 11 | 2023 | Place in `codebooks` |


## File Overview

### `main.py`
Entry point for training and evaluating a single model or running grid search. Configure `year`, `region`, `grid_search`, `shap`, and `model_type` at the top of the file.

### `resampling.py`
Runs a permutation test comparing mean absolute SHAP values between Eastern and Western Europe. Produces `resampling_results.json` with observed differences and p-values for each predictor.

### `data_preparation_2004.py` / `data_preparation_2023.py`
Loads and cleans ESS data for 2004 and 2023 respectively. Filters to eligible voters, replaces non-response codes with NaN, maps the vote target to binary, and returns train/test splits.

### `modeling.py`
Contains models for grid search (`lgbm`, `rf`, `xgb`) and pre-tuned LightGBM models for each region and year. Handles training, evaluation, and saving outputs.

### `shap_analysis.py`
Computes SHAP values and saves a beeswarm plot. 

### `variable_cleaner.py`
Parses the ESS codebook HTML to extract non-response codes (marked with `*`) and replaces them with NaN in the dataframe.

---

## Usage

### Training a model

Open `scripts/main.py` and set the configs at the top:

```python
year = 2023         # 2004, 2023
region = "all"      # "all", "east", "west"
shap = True
grid_search = False
random_state = 42
model_type = "lgbm"   # "dummy", "rf", "xgb", "lgbm"
```

Then run from the project root:

```bash
python -m scripts.main
```

Outputs are saved to `output/{region}_{model_name}/` and include:
- `classification_report.txt`
- `confusion_matrix.png`
- `model.pkl`
- `X_test.pkl`
- `shap_beeswarm.png` (if `shap = True`)

### Running the permutation test

Set `year` in `scripts/resampling.py` and run:

```bash
python -m scripts.resampling
```

Results are saved to `output/permutation_test_{YEAR}/resampling_results.json`.

---

### HPC Setup 

```bash
module load Python/3.11.3-GCCcore-12.3.0
python3 -m venv "$HOME/bachelor2026/venv"
source "$HOME/bachelor2026/venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
```

### Submitting a SLURM job
Grid search and resampling can be ran using an HPC using the job scripts

```bash
sbatch resampling.job
sbatch gridsearch.job
```

---

## Requirements

```
pandas==3.0.2
numpy==2.4.4
scikit-learn==1.8.0
shap==0.51.0
matplotlib==3.10.8
beautifulsoup4==4.14.3
xgboost==3.2.0
lightgbm==4.6.0
category_encoders==2.9.0
```