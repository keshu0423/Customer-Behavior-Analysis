import pandas as pd
import numpy as np

df = pd.read_csv("c:/Users/vadla/OneDrive/Documents/internship/Customer-Behavior-Analysis/dataset/ecommerce_customer_data_custom_ratios.csv")

print("Checking duplicates:")
print("Duplicate rows count:", df.duplicated().sum())

print("\nChecking if Customer Age and Age columns are identical:")
identical_age = (df['Customer Age'] == df['Age']).all()
print("Are they identical?", identical_age)
if not identical_age:
    diff_count = (df['Customer Age'] != df['Age']).sum()
    print(f"Number of differences: {diff_count}")
    print(df[['Customer Age', 'Age']].head(10))

print("\nChecking if Total Purchase Amount equals Product Price * Quantity:")
total_calc = df['Product Price'] * df['Quantity']
matches = (df['Total Purchase Amount'] == total_calc).all()
print("Are they identical?", matches)
if not matches:
    diff_count = (df['Total Purchase Amount'] != total_calc).sum()
    print(f"Number of differences: {diff_count}")
    print(df[['Product Price', 'Quantity', 'Total Purchase Amount']].head(10))
    # Check if there is a multiplier or offset
    ratio = df['Total Purchase Amount'] / total_calc
    print("Ratio summary:")
    print(ratio.describe())

print("\nReturns column value counts (including nulls):")
print(df['Returns'].value_counts(dropna=False))

print("\nChurn column value counts:")
print(df['Churn'].value_counts(dropna=False))

print("\nPayment Method value counts:")
print(df['Payment Method'].value_counts())

print("\nProduct Category value counts:")
print(df['Product Category'].value_counts())
