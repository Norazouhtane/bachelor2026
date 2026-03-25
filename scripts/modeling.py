import os
import pickle
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def choose_model(model_name, random_state):
    if model_name == "lgbm":
        return lgb.LGBMClassifier(random_state=random_state)
    elif model_name == "logistic":
        return LogisticRegression(random_state=random_state)
    elif model_name == "hist":
        return HistGradientBoostingClassifier(random_state=random_state)
    elif model_name == "rf":
        return RandomForestClassifier(random_state=random_state)
    

def train_model(X_train, X_test, y_train, y_test, df, model_name, output_dir, grid_search, random_state):

    model = choose_model(model_name, random_state)
    weights = df.loc[X_train.index, 'pspwght']

    if grid_search:
        model = GridSearchCV(model, parameters[model_name], cv=5, scoring="f1_weighted", n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=weights)
        print(f"  Best params: {model.best_params_}")
        print(f"  Best CV score: {model.best_score_:.4f}")
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
        "max_bin": [70, 80, 90],
        "num_leaves":    [30, 45, 60],
        "learning_rate": [0.65, 0.7, 0.75],
    },
    "hist": {
        "learning_rate": [0.1, 0.15, 0.20],
        "max_depth":    [3, 5, 10],
        "min_samples_leaf": [1, 2, 3],
    },
    "logistic": {
        "C": [0.01, 0.1, 1, 10],
    },
     "rf": {
        "max_features": ["sqrt", "log2"],
        "max_depth":    [3, 5, 10],
        "min_samples_leaf": [1, 2, 3],
    },
}