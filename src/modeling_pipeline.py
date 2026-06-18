import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
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


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError('Model does not expose feature importances.')

    importance = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot.barh(ax=ax, color='#3182bd')
    ax.invert_yaxis()
    ax.set_title('Top Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved feature importance plot to {save_path}')
    plt.close(fig)
    return importance


def shap_explainability(model, X_test, y_test, feature_names, save_dir=None, sample_size=1000):
    """
    Generate SHAP plots and display them directly in the notebook.
    
    Parameters:
        model: Trained model (e.g., XGBoost, RandomForest, etc.).
        X_test: Test dataset (features).
        y_test: True labels for the test dataset.
        feature_names: List of feature names.
        save_dir: Directory to save generated SHAP plots (default: None).
        sample_size: Number of samples to use for SHAP analysis (default: 1000).
    """
    # Sample a subset of the test set for SHAP analysis
    sample_idx = np.random.RandomState(42).choice(X_test.shape[0], min(sample_size, X_test.shape[0]), replace=False)
    X_sample = X_test.iloc[sample_idx]

    # Create SHAP explainer and compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Generate and display the SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    if save_dir:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir_path / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.show()  # Ensure plot is displayed in the notebook

    # Identify one true positive, false positive, and false negative using the full test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    # Find indices for true positive, false positive, and false negative
    indices = {
        'true_positive': np.where((y_test == 1) & (y_pred == 1))[0],
        'false_positive': np.where((y_test == 0) & (y_pred == 1))[0],
        'false_negative': np.where((y_test == 1) & (y_pred == 0))[0],
    }

    selected = {}
    for label, idx_array in indices.items():
        if len(idx_array) > 0:
            selected[label] = idx_array[0]

    # Generate and display SHAP force plots for selected instances
    for label, idx in selected.items():
        instance = X_test.iloc[[idx]]
        if idx in sample_idx:
            instance_shap_values = shap_values[np.where(sample_idx == idx)[0][0]]
        else:
            instance_shap_values = explainer.shap_values(instance)[0]

        # Display the force plot directly in the notebook
        shap.force_plot(
            explainer.expected_value,
            instance_shap_values,
            instance,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir_path / f'shap_force_plot_{label}.png', dpi=150, bbox_inches='tight')
        plt.show()

    return {
        'feature_names': feature_names,
        'shap_values': shap_values,
        'selected_instances': selected,
        'confusion_matrix': cm,
    }

def run_pipeline():
    results = []
    output_dir = ROOT_DIR / 'models' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_configs = [
        ('CreditCard', prepare_creditcard_dataset),
        ('FraudData', prepare_fraud_dataset),
    ]

    for dataset_name, prepare_fn in dataset_configs:
        X, y = prepare_fn('data/processed')
        describe_dataset(dataset_name, X, y)

        X_train_scaled, X_test_scaled, y_train, y_test, X_train_res, y_train_res, scaler = split_scale_resample(X, y)

        baseline = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        baseline.fit(X_train_res, y_train_res)
        results.append(evaluate_model(baseline, X_test_scaled, y_test, 'LogisticRegression', dataset_name))

        # Tune XGBoost on the full dataset and then fit the best ensemble on the train split.
        best_params = tune_xgboost(X, y)
        ensemble = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
            **best_params, # type: ignore
        )
        ensemble.fit(X_train_res, y_train_res)
        results.append(evaluate_model(ensemble, X_test_scaled, y_test, 'XGBoost', dataset_name))

        # Save feature importance for the chosen model
        feature_names = X.columns.tolist()
        importance = plot_feature_importance(ensemble, feature_names, save_path=output_dir / f'{dataset_name}_feature_importance.png')
        importance.to_csv(output_dir / f'{dataset_name}_feature_importance.csv')

        # Save the best model to disk for later explainability if it's the fraud dataset.
        if dataset_name == 'FraudData':
            import joblib
            joblib.dump(ensemble, ROOT_DIR / 'models' / 'best_fraud_xgb.joblib')
            print('Saved FraudData ensemble model artifact.')

            shap_dir = ROOT_DIR / 'models' / 'shap_fraud'
            X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
            shap_explainability(ensemble, X_test_df, y_test.reset_index(drop=True), feature_names, shap_dir)

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / 'model_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved model comparison summary to {summary_path}')
    print(summary_df)


if __name__ == '__main__':
    run_pipeline()
