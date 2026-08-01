import os
import urllib.request
import pandas as pd

def download_dataset():
    url = "https://raw.githubusercontent.com/shahmi0519/Ecommerce-Churn-Prediction/main/data/ecommerce_customer_data_custom_ratios.csv"
    dest_dir = "c:/Users/vadla/OneDrive/Documents/internship/Customer-Behavior-Analysis/dataset"
    dest_path = os.path.join(dest_dir, "ecommerce_customer_data_custom_ratios.csv")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Downloading dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Dataset successfully downloaded and saved to {dest_path}")
        
        # Verify the download by loading the dataset
        df = pd.read_csv(dest_path)
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nData info:")
        df.info()
        
    except Exception as e:
        print(f"Error downloading or verifying dataset: {e}")

if __name__ == "__main__":
    download_dataset()
