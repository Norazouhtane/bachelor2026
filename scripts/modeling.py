import os
import json
import pickle
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def choose_model(model_name, random_state):
    """
    MAKE DOCSTRING
    """

    if model_name == "lgbm":
        return lgb.LGBMClassifier(random_state=random_state)
    elif model_name == "lgbm11":
        return lgb.LGBMClassifier(learning_rate=0.7, max_depth=10, n_estimators=150, num_leaves=30, random_state=random_state)
    elif model_name == "hist1":
        return HistGradientBoostingClassifier(learning_rate=0.3, max_depth=2, max_iter=100, min_samples_leaf=3, random_state=random_state, class_weight="balanced")
    elif model_name == "hist11":
        return HistGradientBoostingClassifier(learning_rate=0.25, max_depth=3, max_iter=100, min_samples_leaf=1, random_state=random_state, class_weight="balanced")
    elif model_name == "rf":
        return RandomForestClassifier(random_state=random_state)
    elif model_name == "xgb":
        return xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1, random_state=random_state)
    elif model_name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=random_state)
    

def train_model(X_train, X_test, y_train, y_test, df, model_name, output_dir, grid_search, random_state):
    """
    MAKE DOCSTRING
    """

    model = choose_model(model_name, random_state)
    weights = df.loc[X_train.index, 'anweight']

    if grid_search:
        parameters["xgb"]["scale_pos_weight"] = [sum(y_train == 0) / sum(y_train == 1)]
        model = GridSearchCV(model, parameters[model_name], cv=5, scoring="f1_macro", n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=weights)
        print(f"  Best params: {model.best_params_}")
        print(f"  Best CV score: {model.best_score_:.4f}")
        with open(os.path.join(output_dir, "best_params.json"), "w") as f:
            json.dump({
                "best_params": model.best_params_,
                "best_CV_score": model.best_score_,
                "search_parameters": parameters[model_name]
            }, f, indent=2)
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
    "rf": {
        "max_features": ["sqrt", "log2"],
        "max_depth":    [5, 10, 15, 20],
        "min_samples_leaf": [10, 15, 20, 25],
        "n_estimators": [250, 300, 350, 400],
        "class_weight": ["balanced"]
    },
    "lgbm": {
        "max_depth": [11, 13, 15], #doc: <=0 means no limit
        "num_leaves":    [12, 15, 17],
        "learning_rate": [0.05, 0.1, 0.15],
        "min_data_in_leaf": [40, 50, 60],
        "class_weight": ["balanced"],
        "n_jobs": [1],
        #"n_estimators": [100]
        #"num_threads": [8]
        #"n_estimators": [100, 150, 200],
    },
    "xgb": {
        "max_depth": [6, 8, 10, 12],
        "learning_rate": [0.04, 0.05, 0.1, 0.2], #also called eta
        "subsample": [0.5, 0.75, 1.0],
        "objective": ["binary:hinge"]
    }
}