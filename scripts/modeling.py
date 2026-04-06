import os
import pickle
import pandas as pd
#import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def choose_model(model_name, random_state):
    """
    MAKE DOCSTRING
    """

    if model_name == "lgbm":
        return lgb.LGBMClassifier(learning_rate=0.7, max_depth=10, n_estimators=150, num_leaves=30, random_state=random_state)
    elif model_name == "hist11":
        return HistGradientBoostingClassifier(learning_rate=0.25, max_depth=3, max_iter=100, min_samples_leaf=1, random_state=random_state, class_weight="balanced")
    elif model_name == "hist1":
        return HistGradientBoostingClassifier(learning_rate=0.3, max_depth=2, max_iter=100, min_samples_leaf=3, random_state=random_state, class_weight="balanced")
    elif model_name == "rf":
        return RandomForestClassifier(max_depth=50, max_features=None, min_samples_leaf=1, n_estimators=10, random_state=random_state)
    

def train_model(X_train, X_test, y_train, y_test, df, model_name, output_dir, grid_search, random_state):
    """
    MAKE DOCSTRING
    """

    model = choose_model(model_name, random_state)
    weights = df.loc[X_train.index, 'anweight']

    if grid_search:
        model = GridSearchCV(model, parameters[model_name], cv=5, scoring="f1_macro", n_jobs=-1)
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

def fit_model(X_train, y_train, df, model_name, random_state):
    """
    MAKE DOCSTRING
    """
    model = choose_model(model_name, random_state)
    weights = df.loc[X_train.index, 'anweight']
    model.fit(X_train, y_train, sample_weight=weights)
    
    return model

parameters = {
    "lgbm": {
        "max_depth": [-1, 10, 15, 20],
        "num_leaves":    [15, 30, 45],
        "learning_rate": [0.7, 0.75, 0.8],
        "n_estimators": [100, 150, 200]
    },
    "hist": {
        "learning_rate": [0.25, 0.3, 0.35],
        "max_depth":    [1, 2, 3, 5],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_iter": [100, 150, 200, 250, 300]
    },
     "rf": {
        "max_features": ["sqrt", "log2", None],
        "max_depth":    [15, 20, 50, 100],
        "min_samples_leaf": [1, 2, 3],
        "n_estimators": [5, 10, 100, 200]
    },
}