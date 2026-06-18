import os
import pandas as pd
import numpy as np

from src.geolocation import map_ip_to_country


def clean_creditcard_data(file_path):
    """Loads and cleans the credit card fraud dataset."""
    print(f"Cleaning credit card data from {file_path}...")
    df = pd.read_csv(file_path)

    n_dups = df.duplicated().sum()
    if n_dups > 0:
        print(f"Removing {n_dups} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicates found.")

    n_nans = df.isnull().sum().sum()
    if n_nans > 0:
        print(f"Imputing {n_nans} missing values with median.")
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    return df


def clean_fraud_data(file_path):
    """Loads and cleans the e-commerce fraud dataset."""
    print(f"Cleaning e-commerce fraud data from {file_path}...")
    df = pd.read_csv(file_path)

    n_dups = df.duplicated().sum()
    if n_dups > 0:
        print(f"Removing {n_dups} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)

    df["signup_time"] = pd.to_datetime(df["signup_time"])
    df["purchase_time"] = pd.to_datetime(df["purchase_time"])
    df["ip_address"] = df["ip_address"].astype("int64")

    n_nans = df.isnull().sum().sum()
    if n_nans > 0:
        print(f"Imputing {n_nans} missing values.")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ["int64", "float64", "<M8[ns]"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    return df


def integrate_geolocation(fraud_df, ip_df_path):
    """Adds country mapping from IP addresses."""
    print(f"Mapping IPs using {ip_df_path}...")
    ip_df = pd.read_csv(ip_df_path)

    return map_ip_to_country(
        fraud_df,
        ip_df,
        ip_int_col="ip_address",
        country_col="country",
    )


def engineer_features(fraud_df):
    """Creates fraud detection features."""
    df = fraud_df.copy()

    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    df["time_since_signup"] = (
        df["purchase_time"] - df["signup_time"]
    ).dt.total_seconds()

    device_counts = df["device_id"].value_counts()
    df["device_sharing_count"] = df["device_id"].map(device_counts)

    ip_counts = df["ip_address"].value_counts()
    df["ip_sharing_count"] = df["ip_address"].map(ip_counts)

    df = df.sort_values("purchase_time").reset_index(drop=True)

    df["device_velocity"] = (
        df.groupby("device_id")["purchase_time"].diff().dt.total_seconds()
    )

    df["ip_velocity"] = (
        df.groupby("ip_address")["purchase_time"].diff().dt.total_seconds()
    )

    df["device_velocity"] = df["device_velocity"].fillna(-1)
    df["ip_velocity"] = df["ip_velocity"].fillna(-1)

    return df


def run_preprocessing_pipeline(
    raw_dir="data/raw",
    processed_dir="data/processed",
):
    """Runs full preprocessing pipeline."""
    os.makedirs(processed_dir, exist_ok=True)

    cc_df = clean_creditcard_data(
        os.path.join(raw_dir, "creditcard.csv")
    )
    cc_df["Amount"] = np.log1p(cc_df["Amount"])

    cc_df.to_csv(
        os.path.join(processed_dir, "creditcard_processed.csv"),
        index=False,
    )

    fraud_df = clean_fraud_data(
        os.path.join(raw_dir, "fraud_data.csv")
    )

    fraud_df = integrate_geolocation(
        fraud_df,
        os.path.join(raw_dir, "IpAddress_to_Country.csv"),
    )

    fraud_df = engineer_features(fraud_df)

    categorical_cols = ["source", "browser", "sex", "country"]

    fraud_df = pd.get_dummies(
        fraud_df,
        columns=categorical_cols,
        drop_first=True,
    )

    fraud_df = fraud_df.drop(
        columns=["user_id", "device_id", "signup_time", "purchase_time"]
    )

    fraud_df.to_csv(
        os.path.join(processed_dir, "fraud_data_processed.csv"),
        index=False,
    )

    print("Preprocessing complete.")
