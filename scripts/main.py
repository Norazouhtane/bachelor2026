from data_preparation import load_data
from model import train_model
from shap_analysis import run_shap
import os

# Configs
country      = "DE"
model_name   = "lgbm"      
data_path    = "data/raw/ESS11.csv"
codebook     = "codebooks/ESS11e04_1 codebook.html"
shap         = False
grid_search  = False


output_dir = f"output/{country}_{model_name}"
os.makedirs(output_dir, exist_ok=True)

X_train, X_test, y_train, y_test, df = load_data(data_path, codebook, country)
model = train_model(X_train, X_test, y_train, y_test, df, model_name, output_dir, grid_search)

if shap:
    run_shap(model, X_test, output_dir)