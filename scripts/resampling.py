import os
import json
import random
import numpy as np
import pandas as pd
from .modeling import fit_model
from .shap_analysis import shap_values
from .variable_cleaner import parse_ess_codebook


# Configs
year = 2004     # 2004, 2023
repeat = 1000   # number of runs
random_state = 42

old_eu       = ["BE", "FR", "DE", "NL", "IE", "GR", "PT", "ES", "AT", "FI", "SE"]
new_eu       = ["EE", "HU", "PL", "SK", "SI"]

data = {
    2004: {
        "data_path": "../data/raw/ESS2.csv",
        "codebook": "../codebooks/ESS2e03_6 codebook.html",
        "model": "all_2004_lgbm",
        "predictors": ['polintr', 'clsprty', 'lrscale', 'stfdem', 'trstprl', 'ppltrst', 'eisced', 'agea', 'hincfel', 'mbtru','nwsppol', 'health']
    },
    2023: {
        "data_path": "../data/raw/ESS11.csv",
        "codebook": "../codebooks/ESS11e04_1 codebook.html",
        "model": "all_2023_lgbm",
        "predictors": ['polintr', 'clsprty', 'lrscale', 'stfdem', 'trstprl', 'ppltrst', 'eisced', 'agea', 'hincfel', 'mbtru', 'nwspol', 'health']
    }
}

data_path = data[year]["data_path"]
codebook = data[year]["codebook"]
model_name = data[year]["model"]
predictors = data[year]["predictors"]
output_dir = f"../output/permutation_test_{year}"

# Data function import
if year == 2004:
    from .data_preparation_2004 import load_data
else:
    from .data_preparation_2023 import load_data


os.makedirs(output_dir, exist_ok=True)


# Non-response dict for cleaning
nonresponse_dict = parse_ess_codebook(codebook) 

# Computing mean SHAP for founding and more recent EU members
X_train_old, X_test_old, y_train_old, y_test_old, df_old = load_data(
    data_path, codebook, old_eu, random_state, nonresponse_dict=nonresponse_dict
)

X_train_new, X_test_new, y_train_new, y_test_new, df_new = load_data(
    data_path, codebook, new_eu, random_state, nonresponse_dict=nonresponse_dict
)

model_old = fit_model(X_train_old, y_train_old, df_old, model_name, random_state)
model_new = fit_model(X_train_new, y_train_new, df_new, model_name, random_state)

shap_old_df = pd.DataFrame(shap_values(model_old, X_test_old), columns=X_test_old.columns)
shap_new_df = pd.DataFrame(shap_values(model_new, X_test_new), columns=X_test_new.columns)

observed_diff = {}
for predictor in predictors:
    old_eu_mean = shap_old_df[predictor].abs().mean()
    new_eu_mean = shap_new_df[predictor].abs().mean()
    observed_diff[predictor] = new_eu_mean - old_eu_mean

    print(f"Mean SHAP for founding EU \n {predictor}: {old_eu_mean:.4f}")
    print(f"Mean SHAP for newer EU \n {predictor}: {new_eu_mean:.4f}")
    print(f"Difference between the two \n {predictor}: {observed_diff[predictor]:.4f}")


# Performing permutation test
countries = old_eu + new_eu
n_old = len(old_eu)
difference = {predictor: [] for predictor in predictors}
rdm = random.Random(random_state)

for i in range(repeat):
    re_old = rdm.sample(countries, n_old)
    re_new = [c for c in countries if c not in re_old]

    X_train_re_old, X_test_re_old, y_train_re_old, y_test_re_old, df_re_old = load_data(
        data_path, codebook, re_old, random_state, nonresponse_dict=nonresponse_dict
    )

    X_train_re_new, X_test_re_new, y_train_re_new, y_test_re_new, df_re_new = load_data(
        data_path, codebook, re_new, random_state, nonresponse_dict=nonresponse_dict
    )

    model_re_old = fit_model(X_train_re_old, y_train_re_old, df_re_old, model_name, random_state)
    model_re_new = fit_model(X_train_re_new, y_train_re_new, df_re_new, model_name, random_state)

    shap_re_old = shap_values(model_re_old, X_test_re_old)
    shap_re_new = shap_values(model_re_new, X_test_re_new)

    shap_re_old_df = pd.DataFrame(shap_values(model_re_old, X_test_re_old), columns=X_test_re_old.columns)
    shap_re_new_df = pd.DataFrame(shap_values(model_re_new, X_test_re_new), columns=X_test_re_new.columns)

    for predictor in predictors:
        mean_re_shap_diff = shap_re_new_df[predictor].abs().mean() - shap_re_old_df[predictor].abs().mean()
        difference[predictor].append(mean_re_shap_diff)

print(f"Resampling done")

# Calculating p-values
results = []
for predictor in predictors:
    null_dist = np.array(difference[predictor])
    count = 0
    for diff in null_dist:
        if abs(diff) >= abs(observed_diff[predictor]): # two-tailed
            count+=1
    p_value = count/repeat
    significant = p_value < 0.05

    results.append({
        "predictor": predictor,
        "observed_diff": observed_diff[predictor],
        "p_value": p_value,
        "significant": significant
    })

results_path = os.path.join(output_dir, "resampling_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)
