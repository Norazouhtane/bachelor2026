import os
import numpy as np
import pandas as pd
from data_preparation import load_data
from modeling import fit_model
from shap_analysis import shap_values
from variable_cleaner import parse_ess_codebook

# Configs
old_eu       = ["BE", "DE", "FR", "IT", "NL"]
new_eu       = ["HR", "BG", "SK", "SI", "PL"]
model_name   = "hist"      
data_path    = "../data/raw/ESS11.csv"
codebook     = "../codebooks/ESS11e04_1 codebook.html"
repeat       = 1000
random_state = 42
predictor    = "polintr"

output_dir = f"../output/resampling_test"
os.makedirs(output_dir, exist_ok=True)


# Non-response dict for cleaning
nonresponse_dict = parse_ess_codebook(codebook)


# Computing mean SHAP for founding and more recent EU countries
X_train_old, X_test_old, y_train_old, y_test_old, df_old = load_data(
    data_path, codebook, old_eu, random_state, nonresponse_dict=nonresponse_dict
)

X_train_new, X_test_new, y_train_new, y_test_new, df_new = load_data(
    data_path, codebook, new_eu, random_state, nonresponse_dict=nonresponse_dict
)

model_old = fit_model(X_train_old, y_train_old, df_old, model_name, random_state)
model_new = fit_model(X_train_new, y_train_new, df_new, model_name, random_state)

shap_old = shap_values(model_old, X_test_old)
shap_new = shap_values(model_new, X_test_new)

shap_old_df = pd.DataFrame(shap_old, columns=X_test_old.columns)
shap_new_df = pd.DataFrame(shap_new, columns=X_test_new.columns)

old_eu_mean = shap_old_df[predictor].mean()
new_eu_mean = shap_new_df[predictor].mean()
mean_shap_diff = new_eu_mean - old_eu_mean

print(f"Mean SHAP for founding EU \n {predictor}: {old_eu_mean}")
print(f"Mean SHAP for newer EU \n {predictor}: {new_eu_mean}")
print(f"Difference between the two \n {predictor}: {mean_shap_diff}")