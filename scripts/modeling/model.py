import os
import pickle
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def choose_model(model_name):
    if model_name == "lgbm":
        return lgb.LGBMClassifier(random_state=42)
    elif model_name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "hist":
        return HistGradientBoostingClassifier()
    

def train_model(X_train, X_test, y_train, y_test, df, model_name, output_dir, grid_search):

    model = choose_model(model_name)
    weights = df.loc[X_train.index, 'pspwght']

    if grid_search:
        model = GridSearchCV(model, parameters[model_name], cv=5, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=weights)
        print(f"  Best params: {model.best_params_}")
        model = model.best_estimator_  
    else:
        model.fit(X_train, y_train, sample_weight=weights)
   

    # Predicting class
    y_pred = model.predict(X_test)

    # Classification report
    classification_rep = classification_report(y_test, y_pred, target_names=["No", "Yes"])
    print(classification_rep)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(classification_rep)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No", "Yes"], ax=ax)
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"))

    # Save model
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    return model

parameters = {
    "lgbm": {
        "n_estimators": [100, 300],
        "max_depth":    [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    },
    "logistic": {
        "C": [0.01, 0.1, 1, 10],
    }
}