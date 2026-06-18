import os
import sys

# Ensure src path is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import run_preprocessing_pipeline

if __name__ == '__main__':
    print("Starting data preprocessing...")
    run_preprocessing_pipeline()
    print("Preprocessing completed!")
