import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

warnings.filterwarnings("ignore")


def load_processed_creditcard_data(processed_dir="data/processed"):
    path = ROOT_DIR / processed_dir / "creditcard_processed.csv"
    return pd.read_csv(path)


def load_processed_fraud_data(processed_dir="data/processed"):
    path = ROOT_DIR / processed_dir / "fraud_data_processed.csv"
    return pd.read_csv(path)


def prepare_creditcard_dataset(processed_dir="data/processed"):
    df = load_processed_creditcard_data(processed_dir)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype("int64")
    return X, y


def prepare_fraud_dataset(processed_dir="data/processed"):
    df = load_processed_fraud_data(processed_dir)
    X = df.drop(columns=["class"])
    y = df["class"].astype("int64")
    return X, y


def describe_dataset(name, X, y):
    print(f"### Dataset: {name}")
    print(f"Rows: {len(X):,}, Features: {X.shape[1]}")

    distribution = y.value_counts().sort_index()

    print("Target class distribution:")
    print(distribution.to_string())

    print("Percent distribution:")
    print((distribution / len(y) * 100).round(3).to_string())

    ratio = distribution.iloc[0] / distribution.iloc[1]
    print(f"Imbalance ratio = {ratio:.1f} : 1")


def split_scale_resample(X, y, random_state=42, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(
        X_train_scaled,
        y_train,
    )  # type: ignore

    print("Before SMOTE:", np.bincount(y_train.values))
    print("After SMOTE:", np.bincount(y_train_res))
    print("-" * 72)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        X_train_res,
        y_train_res,
        scaler,
    )


def evaluate_model(clf, X_test, y_test, model_name, dataset_name):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(
        f"{dataset_name} | {model_name} | "
        f"AP: {ap:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}"
    )
    print(cm)
    print("-" * 72)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "average_precision": ap,
        "f1_score": f1,
        "roc_auc": roc,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }


def cross_validate_model(X, y, base_model, n_splits=5, random_state=42):
    metrics = {
        "average_precision": [],
        "f1_score": [],
        "roc_auc": [],
    }

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(
            X_train_scaled,
            y_train,
        )  # type: ignore

        clf = clone(base_model)
        clf.fit(X_train_res, y_train_res)

        y_val_pred = clf.predict(X_val_scaled)
        y_val_proba = clf.predict_proba(X_val_scaled)[:, 1]

        metrics["average_precision"].append(
            average_precision_score(y_val, y_val_proba)
        )
        metrics["f1_score"].append(f1_score(y_val, y_val_pred))
        metrics["roc_auc"].append(roc_auc_score(y_val, y_val_proba))

        print(
            f"Fold {fold} | "
            f"AP {metrics['average_precision'][-1]:.4f} | "
            f"F1 {metrics['f1_score'][-1]:.4f} | "
            f"ROC-AUC {metrics['roc_auc'][-1]:.4f}"
        )

    summary = {
        "mean_average_precision": np.mean(metrics["average_precision"]),
        "std_average_precision": np.std(metrics["average_precision"]),
        "mean_f1_score": np.mean(metrics["f1_score"]),
        "std_f1_score": np.std(metrics["f1_score"]),
        "mean_roc_auc": np.mean(metrics["roc_auc"]),
        "std_roc_auc": np.std(metrics["roc_auc"]),
    }

    print("CV summary:", summary)
    print("-" * 72)

    return summary


def tune_xgboost(X, y, random_state=42):
    candidate_params = [
        {"n_estimators": 100, "max_depth": 3},
        {"n_estimators": 150, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 7},
    ]

    best_score = -np.inf
    best_params = None

    print("Tuning XGBoost...")

    for params in candidate_params:
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            **params,
        )

        summary = cross_validate_model(X, y, model)

        if summary["mean_average_precision"] > best_score:
            best_score = summary["mean_average_precision"]
            best_params = params

    print(f"Best params: {best_params} | AP {best_score:.4f}")
    return best_params


def run_pipeline():
    results = []

    datasets = [
        ("CreditCard", prepare_creditcard_dataset),
        ("FraudData", prepare_fraud_dataset),
    ]

    for name, fn in datasets:
        X, y = fn("data/processed")
        describe_dataset(name, X, y)

        X_train_s, X_test_s, y_train, y_test, X_res, y_res, scaler = (
            split_scale_resample(X, y)
        )

        baseline = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

        baseline.fit(X_res, y_res)

        results.append(
            evaluate_model(
                baseline,
                X_test_s,
                y_test,
                "LogReg",
                name,
            )
        )

        best_params = tune_xgboost(X, y)

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            **best_params,
        )

        model.fit(X_res, y_res)

        results.append(
            evaluate_model(
                model,
                X_test_s,
                y_test,
                "XGBoost",
                name,
            )
        )

    summary_df = pd.DataFrame(results)
    print(summary_df)


if __name__ == "__main__":
    run_pipeline()