# Import libraries
import os
from .modeling import train_model
from .shap_analysis import run_shap

# Configs
year = 2023         # 2004, 2023
region = "all"      # "all", "east", "west"
shap = True
grid_search = False
random_state = 42
model_type = "lgbm"   # "dummy", "rf", "xgb", "lgbm"

# Define variables based on configs
countries = {
    "all": ["BE", "FR", "DE", "NL", "IE", "GR", "PT", "ES", "AT", "FI", "SE", "EE", "HU", "PL", "SK", "SI"],
    "east": ["EE", "HU", "PL", "SK", "SI"],
    "west": ["BE", "FR", "DE", "NL", "IE", "GR", "PT", "ES", "AT", "FI", "SE"]
}    

data = {
    2004: {
        "data_path": "../data/raw/ESS2.csv",
        "codebook": "../codebooks/ESS2e03_6 codebook.html"
    },
    2023: {
        "data_path": "../data/raw/ESS11.csv",
        "codebook": "../codebooks/ESS11e04_1 codebook.html"
    }
}

country = countries[region]
data_path = data[year]["data_path"]
codebook = data[year]["codebook"]
model_name = model_type if grid_search else f"{region}_{year}_lgbm"
output_dir = f"../output/{region}_{model_name}"

# Data function import
if year == 2004:
    from .data_preparation_2004 import load_data
else:
    from .data_preparation_2023 import load_data

# Run training
os.makedirs(output_dir, exist_ok=True)

X_train, X_test, y_train, y_test, df = load_data(data_path, codebook, country, random_state)
model = train_model(X_train, X_test, y_train, y_test, df, model_name, output_dir, grid_search, random_state)

if shap:
    run_shap(model, X_test, output_dir, model_name)