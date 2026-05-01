import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

def run_shap(model, X_test, output_dir, model_name):
    """
    MAKE DOCSTRING
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, max_display = 20, plot_size = 1.5, show=False)
    beeswarm_path = os.path.join(output_dir, f"shap_beeswarm{model_name}.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {beeswarm_path}")

def shap_values(model, X_test):
    """
    MAKE DOCSTRING
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    raw_shap = shap_values.values

    return raw_shap

