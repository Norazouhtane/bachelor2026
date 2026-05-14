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
from sklearn.ensemble import RandomForestClassifier


def choose_model(model_name, random_state):
    """
    MAKE DOCSTRING
    """

    if model_name == "lgbm":
        return lgb.LGBMClassifier(random_state=random_state)
    elif model_name == "rf":
        return RandomForestClassifier(random_state=random_state)
    elif model_name == "xgb":
        return xgb.XGBClassifier(random_state=random_state)
    elif model_name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=random_state)
    elif model_name == "all_2004_lgbm":
        return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.1, max_depth=25, min_data_in_leaf=100, n_jobs=1, num_leaves=90, random_state=random_state)
    elif model_name == "all_2023_lgbm":
        return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.1, max_depth=20, min_data_in_leaf=80, n_jobs=1, num_leaves=50, random_state=random_state)    
    elif model_name == "east_2004_lgbm":
        return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, max_depth=15, min_data_in_leaf=70, n_jobs=1, num_leaves=50, random_state=random_state)    
    elif model_name == "east_2023_lgbm":
        return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, max_depth=15, min_data_in_leaf=80, n_jobs=1, num_leaves=40, random_state=random_state)    
    elif model_name == "west_2004_lgbm":
        return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.05, max_depth=25, min_data_in_leaf=80, n_jobs=1, num_leaves=70, random_state=random_state)    
    elif model_name == "west_2023_lgbm":
        return lgb.LGBMClassifier(class_weight="balanced", learning_rate=0.15, max_depth=15, min_data_in_leaf=90, n_jobs=1, num_leaves=50, random_state=random_state)    
    

def train_model(X_train, X_test, y_train, y_test, df, model_name, output_dir, grid_search, random_state):
    """
    MAKE DOCSTRING
    """

    model = choose_model(model_name, random_state)
    weights = df.loc[X_train.index, 'anweight']

    if grid_search:
        parameters["xgb"]["scale_pos_weight"] = [sum(y_train == 1) / sum(y_train == 0)]
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
    plt.close(fig)

    # Save model
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Save X_test
    with open(os.path.join(output_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)

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
        "max_depth":    [15, 20, 25, 30],
        "min_samples_leaf": [5, 10, 15, 20, 25],
        "n_estimators": [300, 350, 400, 450, 500, 550],
        "class_weight": ["balanced"]
    },
    "lgbm": {
        "max_depth": [5, 10, 15, 20, 25, 30], #doc: <=0 means no limit
        "num_leaves":    [20, 30, 40, 50, 60],
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25],
        "min_data_in_leaf": [70, 80, 90, 100, 110],
        "class_weight": ["balanced"],
        "n_jobs": [1],
        #"n_estimators": [100]
        #"num_threads": [8]
        #"n_estimators": [100, 150, 200],
    },
    "xgb": {
        "max_depth": [2, 4, 7, 10, 12, 15, 20, 25, 30],
        "learning_rate": [0.01, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7], #also called eta
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "objective": ["binary:hinge"]
    },
        
}