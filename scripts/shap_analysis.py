import os
import shap
import pandas as pd
import matplotlib.pyplot as plt


def run_shap(model, X_test, output_dir, model_name):
    """
    Compute SHAP values and save a beeswarm plot to output_dir.

    Parameters:
        model: Fitted classifier.
        X_test (DataFrame): Test features.
        output_dir (string): Path to directory where the plot is saved.
        model_name (string): Variable to name the output file.
    
    Return:
        None
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Create SHAP beeswarm plot
    shap.summary_plot(shap_values, X_test, max_display = 30, plot_size = 1.0, show=False)
    beeswarm_path = os.path.join(output_dir, f"shap_beeswarm{model_name}.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {beeswarm_path}")

def shap_values(model, X_test):
    """
    Compute and return SHAP values for a fitted model.

    Parameters:
        model: Fitted classifier.
        X_test (DataFrame): Test features.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    raw_shap = shap_values.values

    return raw_shap

