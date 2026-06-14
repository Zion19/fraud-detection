import os
import sys
import pandas as pd
import numpy as np

# Ensure src path is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.geolocation import map_ip_to_country

def clean_creditcard_data(file_path):
    """Loads and cleans the credit card fraud dataset."""
    print(f"Cleaning credit card data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check for duplicates
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        print(f"Removing {n_dups} duplicate rows from credit card data.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicates found in credit card data.")
        
    # Check for NaNs
    n_nans = df.isnull().sum().sum()
    if n_nans > 0:
        print(f"Warning: Found {n_nans} missing values in credit card data. Imputing with median.")
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
    return df

def clean_fraud_data(file_path):
    """Loads and cleans the e-commerce fraud dataset."""
    print(f"Cleaning e-commerce fraud data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Check for duplicates
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        print(f"Removing {n_dups} duplicate rows from e-commerce fraud data.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicates found in e-commerce fraud data.")
        
    # Correct data types
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['ip_address'] = df['ip_address'].astype('int64')
    
    # Check for NaNs
    n_nans = df.isnull().sum().sum()
    if n_nans > 0:
        print(f"Warning: Found {n_nans} missing values in e-commerce fraud data. Imputing with median/mode.")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64', '<M8[ns]']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    
    return df

def integrate_geolocation(fraud_df, ip_df_path):
    """Integrates geolocation country mapping into the e-commerce dataset."""
    print(f"Integrating geolocation data using lookup file {ip_df_path}...")
    ip_df = pd.read_csv(ip_df_path)
    
    # Use helper to map IP to country
    fraud_df = map_ip_to_country(fraud_df, ip_df, ip_int_col='ip_address', country_col='country')
    return fraud_df

def engineer_features(fraud_df):
    """Engineers time, frequency, sharing, and velocity features for e-commerce data."""
    print("Engineering features for e-commerce fraud data...")
    df = fraud_df.copy()
    
    # 1. Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    
    # 2. Sharing features (frequency)
    # Total count of transactions sharing same device_id
    device_sharing = df['device_id'].value_counts()
    df['device_sharing_count'] = df['device_id'].map(device_sharing)
    
    # Total count of transactions sharing same ip_address
    ip_sharing = df['ip_address'].value_counts()
    df['ip_sharing_count'] = df['ip_address'].map(ip_sharing)
    
    # 3. Velocity features (consecutive transaction intervals on same device/IP)
    # Sort by purchase_time first to calculate intervals correctly
    df = df.sort_values(by='purchase_time').reset_index(drop=True)
    
    # Device velocity: difference in seconds between consecutive purchases on same device
    df['device_velocity'] = df.groupby('device_id')['purchase_time'].diff().dt.total_seconds()
    # IP velocity: difference in seconds between consecutive purchases on same IP
    df['ip_velocity'] = df.groupby('ip_address')['purchase_time'].diff().dt.total_seconds()
    
    # Fill NaNs with -1 to indicate "first transaction" (no previous transaction)
    df['device_velocity'] = df['device_velocity'].fillna(-1)
    df['ip_velocity'] = df['ip_velocity'].fillna(-1)
    
    return df

def run_preprocessing_pipeline(raw_dir='data/raw', processed_dir='data/processed'):
    """Orchestrates the data preprocessing pipeline and saves the outputs."""
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Process Credit Card Dataset
    cc_raw_path = os.path.join(raw_dir, 'creditcard.csv')
    cc_df = clean_creditcard_data(cc_raw_path)
    
    # Apply log1p transformation to skewed Amount column
    cc_df['Amount'] = np.log1p(cc_df['Amount'])
    
    # Save processed credit card dataset
    cc_out_path = os.path.join(processed_dir, 'creditcard_processed.csv')
    cc_df.to_csv(cc_out_path, index=False)
    print(f"Processed credit card dataset saved to {cc_out_path}")
    
    # 2. Process E-commerce Dataset
    fraud_raw_path = os.path.join(raw_dir, 'fraud_data.csv')
    ip_raw_path = os.path.join(raw_dir, 'IpAddress_to_Country.csv')
    
    fraud_df = clean_fraud_data(fraud_raw_path)
    fraud_df = integrate_geolocation(fraud_df, ip_raw_path)
    fraud_df = engineer_features(fraud_df)
    
    # Log distribution metrics of engineered features
    print("E-commerce processed shape:", fraud_df.shape)
    
    # Categorical One-Hot Encoding
    categorical_cols = ['source', 'browser', 'sex', 'country']
    print(f"One-hot encoding categorical variables: {categorical_cols}...")
    fraud_df_encoded = pd.get_dummies(fraud_df, columns=categorical_cols, drop_first=True)
    
    # Drop columns that are no longer needed (IDs and datetimes)
    cols_to_drop = ['user_id', 'device_id', 'signup_time', 'purchase_time']
    fraud_df_encoded = fraud_df_encoded.drop(columns=cols_to_drop)
    
    # Convert dummy boolean columns to float/int
    dummy_cols = [c for c in fraud_df_encoded.columns if c not in [
        'purchase_value', 'age', 'ip_address', 'class', 
        'hour_of_day', 'day_of_week', 'time_since_signup',
        'device_sharing_count', 'ip_sharing_count', 'device_velocity', 'ip_velocity'
    ]]
    for col in dummy_cols:
        fraud_df_encoded[col] = fraud_df_encoded[col].astype(int)
        
    # Save processed e-commerce dataset
    fraud_out_path = os.path.join(processed_dir, 'fraud_data_processed.csv')
    fraud_df_encoded.to_csv(fraud_out_path, index=False)
    print(f"Processed e-commerce dataset saved to {fraud_out_path}")

if __name__ == '__main__':
    run_preprocessing_pipeline()
