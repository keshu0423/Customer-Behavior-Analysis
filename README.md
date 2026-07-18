# Customer Behavior Analysis — E-commerce Segmentation & Churn BI

This repository contains an end-to-end data analytics and business intelligence project focused on understanding customer purchasing patterns, segmenting users, and identifying primary drivers of customer churn. This project is a submission for the **InternSpark Data Analytics Internship at Alfido Tech**.

---

## 📌 Project Overview & Objective

The primary objective is to transform raw e-commerce transaction data into a strategic business intelligence asset. By analyzing customer behaviors, transaction histories, returns, and payment preferences, we aim to:
1. Identify primary risk factors driving customer churn.
2. Segment the customer base using RFM (Recency, Frequency, Monetary) modeling to enable targeted marketing.
3. Formulate data-backed, actionable business recommendations to improve customer lifetime value (CLV) and retention for Alfido Tech.

---

## 📊 Dataset Overview

* **Source:** [Kaggle Customer Behavior Analysis Dataset](https://www.kaggle.com/datasets/bhanupratapbiswas/customer-behavior-analysis)
* **Dataset Size:** 250,000 transaction records
* **Unique Customers:** 49,673
* **Target Variable:** `Churn` (Binary: `0 = Retained`, `1 = Churned`)
* **Schema details:**
  * `Customer ID` (Integer): Unique customer identifier.
  * `Purchase Date` (Datetime): Timestamp of purchase.
  * `Product Category` (String): Clothing, Books, Electronics, Home.
  * `Product Price` (Float): Unit price.
  * `Quantity` (Integer): Purchase quantity.
  * `Total Purchase Amount` (Float): Total transaction value (including taxes/discounts).
  * `Payment Method` (String): Credit Card, PayPal, Cash, Crypto.
  * `Returns` (Integer): 1 if returned, 0 otherwise (NaNs handled).
  * `Age` (Integer): Customer age.
  * `Gender` (String): Male, Female.
  * `Churn` (Integer): 1 if churned, 0 otherwise.

---

## ⚙️ Technologies Used

* **Language:** Python 3.14
* **Data Processing & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Modeling & Segmentation:** Scikit-Learn (Rank-based Quintiles for RFM)
* **PDF Report Generation:** ReportLab 5.0
* **Interactive Notebook:** Jupyter Notebook, nbformat, nbconvert
* **Environment Manager:** uv (Astral)

---

## 🚀 Installation & Reproducibility

To replicate the analysis, run the notebook, or generate the PDF report locally, follow these steps:

### 1. Prerequisites
Ensure you have the `uv` tool installed (or standard `python` and `pip`).

### 2. Set Up Virtual Environment & Install Dependencies
Clone the repository and install packages from `requirements.txt`:
```bash
# Clone the repository
git clone https://github.com/your-username/Customer-Behavior-Analysis.git
cd Customer-Behavior-Analysis

# Initialize a virtual environment and install packages
uv venv
uv pip install -r requirements.txt
```

### 3. Run the Pipeline
To download the dataset, execute data cleaning, generate plots, compile the PDF, and create the pre-executed notebook:
```bash
# Download dataset
uv run python scratch/download_data.py

# Run complete analysis
uv run python scratch/run_analysis.py
```

---

## 📈 Analysis Workflow & Results

### 1. Data Cleaning Decisions
* **Redundancy:** Dropped `Customer Age` as it was identical to the `Age` column.
* **Missing Values:** Imputed 47,596 missing values in the `Returns` column as `0` (No Return), avoiding dropping 19% of the records.
* **Types:** Converted dates to datetime objects, and flags (`Returns`, `Churn`) to integers.
* **Standardization:** Stripped spaces and converted category values to Title Case.

### 2. Feature Engineering
We generated:
* `Total Spending` = `Product Price * Quantity`
* `Customer Lifetime Value` (CLV) = Sum of transactions per customer.
* `Average Order Value` (AOV) = Average spent per transaction per customer.
* `Purchase Frequency` = Total orders per customer.
* `Recency` = Days since last purchase relative to the max date.
* `Customer Age Group` = Binned ages (`Under 25`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`).

### 3. Customer Segments (RFM Profile)

| Segment | Recency (d) | Frequency | CLV ($) | AOV ($) | Return % | Churn % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **High Value (Champions)** | 114.7 | 8.35 | $2,965.73 | $355.20 | 40.5% | 19.7% |
| **Loyal Customers** | 204.0 | 6.55 | $2,425.43 | $370.29 | 40.2% | 20.3% |
| **New Customers** | 98.7 | 1.94 | $811.23 | $418.16 | 40.4% | 20.7% |
| **At Risk** | 305.8 | 5.56 | $2,076.32 | $373.44 | 40.4% | 19.4% |
| **Lost Customers** | 413.4 | 2.52 | $975.31 | $387.03 | 39.6% | 20.1% |

### 4. Key Churn Indicators
* **Returns:** Customers who returned products have a **23.3%** churn rate, compared to **16.8%** for non-returners.
* **Payment Methods:** Crypto checkouts lead with **20.3%** churn, while Credit Card has the lowest.
* **Transaction drop-off:** 50.6% of customers buy only once, indicating a retention bottleneck at the second purchase.

---

## 🖼️ Code & Output Screenshot Locations

For your presentation and submission screenshots, capture the following steps in the notebook `Customer_Behavior_Analysis.ipynb` or the terminal:
1. **Data Loading:** Screenshot of cell 2 (loading data, printing shape, and data types).
2. **Data Cleaning:** Screenshot of cell 3 showing check for duplicates, handling missing values, and type conversions.
3. **EDA & Visualizations:** Screenshots of cell 5 displaying summary tables and the 2x2 multi-plot dashboard (Age distribution, Category sales, Monthly spending, Correlation Heatmap).
4. **RFM Segmentation:** Screenshot of cell 6 showing segment mapping logic, counts, and segment profile table.
5. **Insights & Recommendations:** Screenshot of the bottom markdown cells summarizing the findings.

---

## 💡 Five Actionable Recommendations for Alfido Tech

1. **Return-to-Loyalty Recovery Program:** Automatically email a 10-15% discount code or free return shipping exchange options when a return is registered.
2. **VIP Club for Champions:** Retain the top-spending segment with priority customer service, free delivery, and early product previews.
3. **Optimize Checkout Flows:** default checkouts to credit cards and PayPal, offering simple incentives for saving payment options.
4. **Second-Purchase WELCOME Offer:** Send a 15% discount for a second purchase within 14 days of their first order.
5. **Free Shipping Threshold:** Implement free shipping for orders over $150 to drive Average Order Value (AOV).

---

## 📁 Repository Structure

```
Customer-Behavior-Analysis/
│── README.md
│── Customer_Behavior_Analysis.ipynb
│── dataset/
│   └── ecommerce_customer_data_custom_ratios.csv
│── cleaned_dataset.csv
│── images/
│   ├── 1_age_distribution.png
│   ├── 2_product_category.png
│   ├── 3_monthly_spending.png
│   ├── 4_payment_method.png
│   ├── 5_box_purchase_by_category.png
│   ├── 6_correlation_matrix.png
│   ├── 7_gender_distribution.png
│   ├── 8_age_vs_spending.png
│   ├── 9_rfm_segments.png
│   └── 10_churn_by_category.png
│── report/
│   └── report.pdf
│── requirements.txt
└── scratch/
    ├── download_data.py
    ├── execute_notebook.py
    ├── inspect_data.py
    └── run_analysis.py
```

---

## 🧑‍💻 Author

* **Role:** Data Analyst Intern (InternSpark Data Analytics Internship)
* **Partner Firm:** Alfido Tech
* **Date:** July 2026
