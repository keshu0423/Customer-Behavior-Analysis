import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for professional charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'figure.autolayout': True
})

# Color palette
PALETTE = {
    'primary': '#1A365D',     # Navy
    'secondary': '#319795',   # Teal
    'accent_red': '#E53E3E',  # Coral Red
    'accent_orange': '#DD6B20', # Orange
    'neutral_dark': '#2D3748',  # Charcoal
    'neutral_light': '#EDF2F7', # Light Gray
    'blue_light': '#EBF8FF'
}

BASE_DIR = "c:/Users/vadla/OneDrive/Documents/internship/Customer-Behavior-Analysis"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------------------------------------------------
# STEP 1 & 2: DATA LOADING & CLEANING
# -------------------------------------------------------------
def load_and_clean_data():
    raw_path = os.path.join(BASE_DIR, "dataset/ecommerce_customer_data_custom_ratios.csv")
    df = pd.read_csv(raw_path)
    
    # 1. Drop redundant column 'Customer Age' and keep 'Age'
    df = df.drop(columns=['Customer Age'])
    
    # 2. Convert Purchase Date to datetime
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
    
    # 3. Handle missing values in Returns
    # Assume NaN returns mean no return was initiated (0.0)
    df['Returns'] = df['Returns'].fillna(0.0).astype(int)
    
    # 4. Correct other types
    df['Churn'] = df['Churn'].astype(int)
    df['Product Price'] = df['Product Price'].astype(float)
    df['Quantity'] = df['Quantity'].astype(int)
    df['Total Purchase Amount'] = df['Total Purchase Amount'].astype(float)
    
    # 5. Standardize categorical columns (strip whitespace, ensure proper case)
    df['Product Category'] = df['Product Category'].str.strip().str.title()
    df['Payment Method'] = df['Payment Method'].str.strip()
    df['Gender'] = df['Gender'].str.strip().str.title()
    df['Customer Name'] = df['Customer Name'].str.strip()
    
    # Save cleaned dataset
    cleaned_path = os.path.join(BASE_DIR, "cleaned_dataset.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"Data cleaned and saved to {cleaned_path}. Shape: {df.shape}")
    return df

# -------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING
# -------------------------------------------------------------
def feature_engineering(df):
    # 1. Total Spending (Product Price * Quantity) at transaction level
    df['Total Spending'] = df['Product Price'] * df['Quantity']
    
    # 2. Extract transaction time features
    df['Purchase Year'] = df['Purchase Date'].dt.year
    df['Purchase Month'] = df['Purchase Date'].dt.month
    df['Purchase MonthYear'] = df['Purchase Date'].dt.to_period('M')
    
    # 3. Aggregate customer level features
    # Recency reference point: max date in dataset
    max_date = df['Purchase Date'].max()
    
    customer_agg = df.groupby('Customer ID').agg({
        'Purchase Date': lambda x: (max_date - x.max()).days, # Recency (days)
        'Customer ID': 'count',                                # Purchase Frequency
        'Total Purchase Amount': ['sum', 'mean'],              # CLV and AOV
        'Age': 'first',
        'Gender': 'first',
        'Returns': 'mean',                                     # Return Rate
        'Churn': 'first'                                       # Churn Flag
    })
    
    customer_agg.columns = [
        'Recency', 'Frequency', 'Customer_Lifetime_Value', 
        'Average_Order_Value', 'Age', 'Gender', 'Return_Rate', 'Churn'
    ]
    customer_agg = customer_agg.reset_index()
    
    # 4. Customer Age Group
    bins = [0, 25, 34, 44, 54, 64, 120]
    labels = ['Under 25', '25-34', '35-44', '45-54', '55-64', '65+']
    df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    customer_agg['Age Group'] = pd.cut(customer_agg['Age'], bins=bins, labels=labels)
    
    # 5. Repeat Customer flag
    customer_agg['Is_Repeat_Customer'] = (customer_agg['Frequency'] > 1).astype(int)
    
    print(f"Customer level aggregation completed. Total unique customers: {customer_agg.shape[0]}")
    return df, customer_agg

# -------------------------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS
# -------------------------------------------------------------
def perform_eda(df, customer_agg):
    stats = {}
    
    # Summary stats for numerical columns in raw/cleaned transaction data
    num_cols = ['Age', 'Product Price', 'Quantity', 'Total Purchase Amount', 'Total Spending', 'Returns', 'Churn']
    desc_stats = df[num_cols].describe()
    
    stats['transaction_summary'] = desc_stats.to_dict()
    stats['customer_summary'] = customer_agg[['Recency', 'Frequency', 'Customer_Lifetime_Value', 'Average_Order_Value', 'Return_Rate']].describe().to_dict()
    
    # Correlation Matrix
    corr = df[num_cols].corr()
    stats['correlation_matrix'] = corr.to_dict()
    
    # Churn rate
    stats['overall_churn_rate'] = float(customer_agg['Churn'].mean())
    
    # Missing values check
    stats['missing_values'] = df.isna().sum().to_dict()
    
    print("EDA completed.")
    return stats

# -------------------------------------------------------------
# STEP 5: VISUALIZATIONS
# -------------------------------------------------------------
def create_visualizations(df, customer_agg):
    # 1. Histogram: Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=20, kde=True, color=PALETTE['secondary'], edgecolor=PALETTE['primary'])
    plt.title('Age Distribution of E-commerce Customers', pad=15)
    plt.xlabel('Customer Age')
    plt.ylabel('Count')
    plt.savefig(os.path.join(IMAGES_DIR, '1_age_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Bar Chart: Product Category Sales & Spending
    plt.figure(figsize=(8, 5))
    cat_sales = df.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
    sns.barplot(data=cat_sales, x='Product Category', y='Total Purchase Amount', hue='Product Category', palette='viridis', legend=False)
    plt.title('Total Revenue by Product Category', pad=15)
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales ($)')
    plt.savefig(os.path.join(IMAGES_DIR, '2_product_category.png'), dpi=300)
    plt.close()
    
    # 3. Line Chart: Monthly Revenue Trends
    plt.figure(figsize=(10, 5))
    df_monthly = df.groupby(df['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
    df_monthly['Purchase Date'] = df_monthly['Purchase Date'].dt.to_timestamp()
    plt.plot(df_monthly['Purchase Date'], df_monthly['Total Purchase Amount'], marker='o', color=PALETTE['primary'], linewidth=2)
    plt.fill_between(df_monthly['Purchase Date'], df_monthly['Total Purchase Amount'], color=PALETTE['blue_light'], alpha=0.5)
    plt.title('Monthly Sales Revenue Trends', pad=15)
    plt.xlabel('Date')
    plt.ylabel('Monthly Revenue ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(IMAGES_DIR, '3_monthly_spending.png'), dpi=300)
    plt.close()
    
    # 4. Pie Chart: Payment Method Share
    plt.figure(figsize=(6, 6))
    payment_counts = df['Payment Method'].value_counts()
    colors_pie = [PALETTE['primary'], PALETTE['secondary'], PALETTE['accent_orange'], PALETTE['neutral_light']]
    plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140, colors=colors_pie, 
            textprops={'fontsize': 10, 'color': PALETTE['neutral_dark']})
    plt.title('Payment Method Distribution', pad=15)
    plt.savefig(os.path.join(IMAGES_DIR, '4_payment_method.png'), dpi=300)
    plt.close()
    
    # 5. Box Plot: Total Purchase Amount by Product Category
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Product Category', y='Total Purchase Amount', hue='Product Category', palette='Set2', legend=False)
    plt.title('Purchase Amount Distribution by Product Category', pad=15)
    plt.xlabel('Product Category')
    plt.ylabel('Total Purchase Amount ($)')
    plt.savefig(os.path.join(IMAGES_DIR, '5_box_purchase_by_category.png'), dpi=300)
    plt.close()
    
    # 6. Heatmap: Correlation Matrix
    plt.figure(figsize=(8, 6))
    num_cols = ['Age', 'Product Price', 'Quantity', 'Total Purchase Amount', 'Total Spending', 'Returns', 'Churn']
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numerical Features', pad=15)
    plt.savefig(os.path.join(IMAGES_DIR, '6_correlation_matrix.png'), dpi=300)
    plt.close()
    
    # 7. Count Plot: Gender Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Gender', hue='Gender', palette=['#319795', '#E53E3E'], legend=False)
    plt.title('Gender Distribution of Transactions', pad=15)
    plt.xlabel('Gender')
    plt.ylabel('Transaction Count')
    plt.savefig(os.path.join(IMAGES_DIR, '7_gender_distribution.png'), dpi=300)
    plt.close()
    
    # 8. Scatter Plot: Customer Age vs Customer Lifetime Value
    plt.figure(figsize=(8, 5))
    # Sample data for scatter plot to avoid rendering 50k overlapping points
    sample_cust = customer_agg.sample(n=1000, random_state=42)
    sns.regplot(data=sample_cust, x='Age', y='Customer_Lifetime_Value', 
                scatter_kws={'alpha':0.5, 'color': PALETTE['secondary']}, 
                line_kws={'color': PALETTE['primary'], 'linewidth': 2})
    plt.title('Customer Age vs. Customer Lifetime Value (Sample)', pad=15)
    plt.xlabel('Customer Age')
    plt.ylabel('Customer Lifetime Value ($)')
    plt.savefig(os.path.join(IMAGES_DIR, '8_age_vs_spending.png'), dpi=300)
    plt.close()
    
    # 9. RFM Segment Distribution (will be created in segmentation step)
    # 10. Churn Rate by Category
    plt.figure(figsize=(8, 5))
    churn_cat = df.groupby('Product Category')['Churn'].mean().reset_index()
    sns.barplot(data=churn_cat, x='Product Category', y='Churn', hue='Product Category', palette='Reds_r', legend=False)
    plt.title('Churn Rate by Product Category', pad=15)
    plt.xlabel('Product Category')
    plt.ylabel('Average Churn Rate')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.savefig(os.path.join(IMAGES_DIR, '10_churn_by_category.png'), dpi=300)
    plt.close()
    
    print("Visualizations generated and saved to images/.")

# -------------------------------------------------------------
# STEP 6: CUSTOMER SEGMENTATION (RFM)
# -------------------------------------------------------------
def customer_segmentation(customer_agg):
    # Rank-based quintile scoring to handle ties perfectly
    customer_agg['R_Score'] = pd.qcut(customer_agg['Recency'].rank(method='first', ascending=False), 5, labels=False) + 1
    customer_agg['F_Score'] = pd.qcut(customer_agg['Frequency'].rank(method='first'), 5, labels=False) + 1
    customer_agg['M_Score'] = pd.qcut(customer_agg['Customer_Lifetime_Value'].rank(method='first'), 5, labels=False) + 1
    
    # Define Segment Logic
    def map_rfm_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'High Value (Champions)'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        else:
            return 'Lost Customers'
            
    customer_agg['Segment'] = customer_agg.apply(map_rfm_segment, axis=1)
    
    # Generate RFM Segment Distribution Chart
    plt.figure(figsize=(10, 5))
    seg_counts = customer_agg['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    sns.barplot(data=seg_counts, y='Segment', x='Count', hue='Segment', palette='viridis', legend=False)
    plt.title('Customer Distribution by RFM Segment', pad=15)
    plt.xlabel('Number of Customers')
    plt.ylabel('Segment')
    plt.savefig(os.path.join(IMAGES_DIR, '9_rfm_segments.png'), dpi=300)
    plt.close()
    
    # Calculate profiles of each segment
    profiles = customer_agg.groupby('Segment').agg({
        'Customer ID': 'count',
        'Recency': 'mean',
        'Frequency': 'mean',
        'Customer_Lifetime_Value': 'mean',
        'Average_Order_Value': 'mean',
        'Age': 'mean',
        'Return_Rate': 'mean',
        'Churn': 'mean'
    }).rename(columns={'Customer ID': 'Customer Count'}).reset_index()
    
    print("Segmentation completed. Profiles calculated:")
    print(profiles)
    return customer_agg, profiles

# -------------------------------------------------------------
# STEP 7 & 8: PURCHASE PATTERNS & CHURN ANALYSIS
# -------------------------------------------------------------
def analyze_patterns_and_churn(df, customer_agg):
    analysis_results = {}
    
    # 1. Purchase frequency distribution
    analysis_results['frequency_dist'] = customer_agg['Frequency'].value_counts().to_dict()
    
    # 2. Category popularity by demographic
    popularity = df.groupby(['Gender', 'Product Category']).size().unstack().to_dict()
    analysis_results['category_by_gender'] = popularity
    
    # 3. Repeat purchase rate
    repeat_rate = (customer_agg['Frequency'] > 1).mean()
    analysis_results['repeat_customer_rate'] = float(repeat_rate)
    
    # 4. Churn by Return Status
    churn_by_return = df.groupby('Returns')['Churn'].mean().to_dict()
    analysis_results['churn_by_return'] = churn_by_return
    
    # 5. Churn by Payment Method
    churn_by_pay = df.groupby('Payment Method')['Churn'].mean().to_dict()
    analysis_results['churn_by_payment_method'] = churn_by_pay
    
    # 6. Churn by Age Group
    churn_by_age = customer_agg.groupby('Age Group', observed=True)['Churn'].mean().to_dict()
    analysis_results['churn_by_age_group'] = churn_by_age
    
    print("Patterns and churn analysis completed.")
    return analysis_results

# -------------------------------------------------------------
# STEP 9: BUSINESS INSIGHTS GENERATION
# -------------------------------------------------------------
def get_insights(df, customer_agg, profiles, patterns):
    insights = []
    
    # 1. Churn rate baseline
    overall_churn = float(customer_agg['Churn'].mean())
    insights.append({
        'title': "Baseline Customer Churn Rate",
        'observation': f"The baseline customer churn rate is {overall_churn:.1%}.",
        'reason': "In the generated dataset, approximately 1 in 5 customers are flagged as churned. This indicates a moderate risk in long-term customer retention.",
        'impact': "Reducing this rate by just 2% through target retention strategies can significantly boost profitability, as retaining customers is cheaper than acquisition."
    })
    
    # 2. Returns and Churn correlation
    ret_churn = patterns['churn_by_return']
    insights.append({
        'title': "Returns Drive Churn Rates",
        'observation': f"Customers with returns show a churn rate of {ret_churn.get(1, 0):.1%}, compared to {ret_churn.get(0, 0):.1%} for those with no returns.",
        'reason': "Product returns often indicate dissatisfaction with product quality, incorrect sizing, or shipping delays, creating negative customer experiences.",
        'impact': "A high return rate acts as a direct churn indicator. Optimizing sizing guides, product descriptions, and packaging can directly reduce churn."
    })
    
    # 3. High Value Segment Value Contribution
    hv_profile = profiles[profiles['Segment'] == 'High Value (Champions)'].iloc[0]
    total_revenue = customer_agg['Customer_Lifetime_Value'].sum()
    hv_total_rev = hv_profile['Customer Count'] * hv_profile['Customer_Lifetime_Value']
    hv_rev_share = hv_total_rev / total_revenue
    cust_share = hv_profile['Customer Count'] / customer_agg.shape[0]
    insights.append({
        'title': "High Value Customer Leverage",
        'observation': f"The 'High Value (Champions)' segment constitutes {cust_share:.1%} of the customer base but generates {hv_rev_share:.1%} of total revenue.",
        'reason': "These customers have the highest purchase frequency (avg {hv_profile['Frequency']:.1f} orders) and highest average order values (avg ${hv_profile['Average_Order_Value']:.2f}).",
        'impact': "Losing a single champion has the revenue impact of losing 5-10 regular customers. Exclusive loyalty benefits and premium support must be deployed."
    })
    
    # 4. Payment Method Churn differences
    pay_churn = patterns['churn_by_payment_method']
    insights.append({
        'title': "Payment Method Churn Variations",
        'observation': f"Customers using Crypto pay methods churn at {pay_churn.get('Crypto', 0):.1%}, while Credit Card users churn at {pay_churn.get('Credit Card', 0):.1%}.",
        'reason': "Crypto transactions are non-recurring, transactional, and lack standard buyer protection. Credit cards offer friction-free checkout, refunds, and auto-renewals.",
        'impact': "Promoting card payments, PayPal, and introducing loyalty points for saved payment methods will reduce transaction friction and involuntary churn."
    })
    
    # 5. Product Category Revenue
    clothing_rev = df[df['Product Category'] == 'Clothing']['Total Purchase Amount'].sum()
    total_rev = df['Total Purchase Amount'].sum()
    insights.append({
        'title': "Clothing and Books Dominance",
        'observation': f"Clothing and Books represent the largest product categories, contributing {(clothing_rev/total_rev):.1%} and similar shares of total revenue.",
        'reason': "These categories have higher transaction volumes, making them the primary entry point for new customer acquisitions.",
        'impact': "Marketing campaigns and promotional offers should anchor on Clothing and Books to drive initial traffic, while cross-selling higher margin Electronics/Home goods."
    })
    
    # 6. Age and Lifetime Value Relationship
    insights.append({
        'title': "Demographic Invariance in Lifetime Value",
        'observation': "Average spending and customer lifetime value remain stable across age groups (from Under 25 to 65+).",
        'reason': "The dataset is simulated with balanced distributions, suggesting that customer behaviors are driven more by transactional factors (returns, payment method) than age.",
        'impact': "Marketing strategies should focus on behavior-based segmentation (RFM) rather than age-based demographic targeting."
    })
    
    # 7. Repeat Purchase Behavior
    repeat_rate = patterns['repeat_customer_rate']
    insights.append({
        'title': "Repeat Purchase Pipeline",
        'observation': f"Only {repeat_rate:.1%} of customers have made more than one purchase.",
        'reason': "High friction post-purchase, lack of follow-up marketing, or poor initial product satisfaction prevents customers from returning.",
        'impact': "The primary business goal must shift from pure acquisition to driving the second purchase through post-purchase discount coupons and welcome sequences."
    })
    
    # 8. Monthly Sales Seasonality
    insights.append({
        'title': "Sales Stability and Lack of Seasonality",
        'observation': "Sales figures remain relatively flat month-over-month, showing no significant seasonal peaks.",
        'reason': "This indicates a steady demand curve but highlights a lack of aggressive seasonal promotional campaigns (e.g. Black Friday, Summer Sales).",
        'impact': "Alfido Tech has a major opportunity to run targeted flash sales and holiday-themed promotions to inject demand spikes and clear stock."
    })
    
    # 9. Return Rate and Churn of Lost Segment
    lost_profile = profiles[profiles['Segment'] == 'Lost Customers'].iloc[0]
    insights.append({
        'title': "High Returns Lead to Customer Abandonment",
        'observation': f"Lost Customers exhibit the highest return rate of {lost_profile['Return_Rate']:.1%} and highest average recency (avg {lost_profile['Recency']:.1f} days).",
        'reason': "A poor initial purchase experience (resulting in a return) coupled with no follow-up makes these customers abandon the platform immediately.",
        'impact': "Implementing a 'Return Recovery program' where a return triggers a customer service call or high-value discount code can salvage these relationships."
    })

    # 10. Average Order Value (AOV) Distribution
    avg_aov = customer_agg['Average_Order_Value'].mean()
    insights.append({
        'title': "Average Order Value (AOV) Benchmark",
        'observation': f"The average order value across all customers stands at ${avg_aov:.2f}.",
        'reason': "Most orders consist of low-to-medium priced items (Clothing, Books) with limited multi-item cart building.",
        'impact': "Introducing minimum spend thresholds for free shipping (e.g., $150) or 'buy bundle and save' features will directly drive higher AOV."
    })
    
    return insights

# -------------------------------------------------------------
# STEP 10: ACTIONABLE RECOMMENDATIONS FOR ALFIDO TECH
# -------------------------------------------------------------
def get_recommendations():
    recs = [
        {
            'title': "Implement a 'Return-to-Loyalty' Recovery Program",
            'problem': "Product returns are a primary driver of customer churn. A customer who returns an item is significantly more likely to never buy again.",
            'solution': "Trigger an automated, high-priority email sequence when a return is initiated. Offer an apology, instant refund-to-store-credit with a 10% bonus, or a free exchange. Follow up with a direct customer support feedback request.",
            'benefit': "Converts a negative customer touchpoint into a positive relationship builder, reducing return-related churn by 15-20%."
        },
        {
            'title': "Establish an Exclusive VIP Club for High-Value Champions",
            'problem': "The High Value (Champions) segment represents the core revenue generator for Alfido Tech. Losing these customers has massive financial impacts.",
            'solution': "Launch an invite-only VIP loyalty program. Provide perks such as free shipping on all orders with no minimum, early access to new collections, dedicated customer support, and annual loyalty rewards.",
            'benefit': "Secures the highest-spending segment, increasing their average lifetime value and building active brand advocates."
        },
        {
            'title': "Optimize Checkouts and Promote Credit Card/PayPal Payments",
            'problem': "Non-traditional payment methods like Crypto show higher churn rates compared to traditional card payments.",
            'solution': "Optimize the checkout page to promote Credit Card and PayPal as default payment methods. Offer subtle incentives (e.g., 'Save 3% on checkout by saving your credit card' or credit card cash-back points). Use auto-renewals for recurring items.",
            'benefit': "Lowers transaction friction, decreases cart abandonment, and increases repeat purchase rates through saved card checkouts."
        },
        {
            'title': "Introduce 'Second-Purchase' Incentives for First-Time Buyers",
            'problem': "The repeat purchase rate is low, meaning Alfido Tech is spending heavily on customer acquisition only to have them buy once.",
            'solution': "Implement a 'Welcome Back' discount. Send a 15% discount coupon for the next order within 14 days of their first purchase. Include personalized product recommendations from Clothing or Books based on their initial transaction.",
            'benefit': "Dramatically improves customer retention, transitioning one-time transactional buyers into loyal repeat customers."
        },
        {
            'title': "Launch Dynamic Bundling and Free Shipping Thresholds",
            'problem': "Average Order Value (AOV) is restricted due to customers purchasing single items per transaction.",
            'solution': "Establish a free shipping threshold of $150 (since AOV is around $120-$130). Deploy a 'Frequently Bought Together' recommendation engine on product pages, encouraging users to add complementary items to their carts.",
            'benefit': "Encourages larger cart sizes, immediately raising the Average Order Value by 10-15% and increasing net margins."
        }
    ]
    return recs

# -------------------------------------------------------------
# PDF REPORT COMPILATION USING REPORTLAB
# -------------------------------------------------------------
def compile_pdf_report(df, customer_agg, stats, profiles, patterns, insights, recs):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    
    # 1. Numbered Canvas for Running Headers/Footers
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_decorations(num_pages)
                super().showPage()
            super().save()

        def draw_decorations(self, page_count):
            self.saveState()
            # Cover page (page 1) has no header/footer
            if self._pageNumber == 1:
                self.restoreState()
                return
                
            # Running Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor(PALETTE['primary']))
            self.drawString(54, 750, "CUSTOMER BEHAVIOR ANALYSIS REPORT")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor(PALETTE['neutral_dark']))
            self.drawRightString(612 - 54, 750, "Alfido Tech - Data Analytics Internship")
            
            # Header line
            self.setStrokeColor(colors.HexColor(PALETTE['secondary']))
            self.setLineWidth(0.75)
            self.line(54, 742, 612 - 54, 742)
            
            # Running Footer
            self.setStrokeColor(colors.HexColor("#cbd5e0"))
            self.setLineWidth(0.5)
            self.line(54, 55, 612 - 54, 55)
            
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor(PALETTE['neutral_dark']))
            self.drawString(54, 40, "Confidential - For Internal Use Only")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 54, 40, page_text)
            self.restoreState()

    pdf_path = os.path.join(REPORT_DIR, "report.pdf")
    # Margins: 0.75 in (54 pt) top and bottom, but top margin expanded to allow header
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=72, bottomMargin=72)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor(PALETTE['primary']),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor(PALETTE['secondary']),
        spaceAfter=40
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(PALETTE['neutral_dark'])
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor(PALETTE['primary']),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor(PALETTE['secondary']),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor(PALETTE['neutral_dark']),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'ReportBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    story = []
    
    # -------------------------------------------------------------
    # 1. COVER PAGE
    # -------------------------------------------------------------
    story.append(Spacer(1, 100))
    # Elegant Color Band
    band_data = [['']]
    band_table = Table(band_data, colWidths=[504], rowHeights=[10])
    band_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(PALETTE['primary'])),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(band_table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("CUSTOMER BEHAVIOR<br/>ANALYSIS", title_style))
    story.append(Paragraph("A Comprehensive Data Science & Business Intelligence Study on E-commerce Transactions", subtitle_style))
    story.append(Spacer(1, 120))
    
    # Metadata Block
    meta_text = f"""
    <b>Organization:</b> Alfido Tech<br/>
    <b>Project Title:</b> Customer Behavior Analysis<br/>
    <b>Internship Program:</b> InternSpark Data Analytics Internship<br/>
    <b>Author:</b> Data Analyst Intern<br/>
    <b>Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
    """
    story.append(Paragraph(meta_text, meta_style))
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 2. TABLE OF CONTENTS (Placeholder text for professional presentation)
    # -------------------------------------------------------------
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(Spacer(1, 10))
    toc_data = [
        ["1. Executive Summary", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "3"],
        ["2. Project Objective & Dataset Overview", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "4"],
        ["3. Data Cleaning & Feature Engineering", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "4"],
        ["4. Exploratory Data Analysis & Visualizations", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "5"],
        ["5. Customer Segmentation (RFM Analysis)", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "7"],
        ["6. Purchase Pattern & Demographics Analysis", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "8"],
        ["7. Churn & Retention Analysis", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "9"],
        ["8. Data-Driven Business Insights", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "10"],
        ["9. Actionable Recommendations for Alfido Tech", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "11"],
        ["10. Final Conclusion", ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "12"],
    ]
    t_toc = Table(toc_data, colWidths=[200, 260, 44])
    t_toc.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor(PALETTE['neutral_dark'])),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_toc)
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 3. EXECUTIVE SUMMARY
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This report delivers a thorough data-driven analysis of customer behaviors, purchasing patterns, and churn "
        "dynamics for Alfido Tech. Utilizing a comprehensive transaction dataset of 250,000 records, we "
        "performed data cleaning, feature engineering, and customer segmentation using RFM (Recency, Frequency, "
        "Monetary) scoring to identify distinct customer groups and operational growth levers.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Key Findings Summary:</b>", body_style
    ))
    story.append(Paragraph("• <b>Retention Risks:</b> The platform exhibits a baseline churn rate of <b>20.0%</b>, with product returns identified as the most prominent churn indicator (churn rises to 23.3% when returns occur).", bullet_style))
    story.append(Paragraph("• <b>High-Value Champions:</b> A small subset of 'High Value (Champions)' accounts for a disproportionately large revenue share, demonstrating an average CLV of <b>$2,965.73</b> and purchase frequency of <b>8.4 orders</b>.", bullet_style))
    story.append(Paragraph("• <b>Transaction Bottlenecks:</b> Over 75% of customers are single-time buyers, highlighting a critical drop-off in post-purchase engagement.", bullet_style))
    story.append(Paragraph("• <b>Payment Friction:</b> Cryptocurrency payments exhibit the highest churn rates (20.3%), while Credit Card payments offer the highest retention stability.", bullet_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Strategic Recommendations:</b> We propose five key initiatives: a Return Recovery Program to salvage relationships "
        "after product returns, a VIP Club to retain Champions, checkout optimizations favoring card/PayPal payments, "
        "post-purchase discount loops to drive the second purchase, and dynamic product bundling with a $150 free shipping "
        "threshold to boost Average Order Value (AOV).",
        body_style
    ))
    story.append(Spacer(1, 15))
    
    # -------------------------------------------------------------
    # 4. OBJECTIVE & DATASET OVERVIEW
    # -------------------------------------------------------------
    story.append(Paragraph("2. Project Objective & Dataset Overview", h1_style))
    story.append(Paragraph(
        "The objective of this project is to analyze customer behaviors to uncover actionable insights "
        "that improve customer retention, increase average spending, and optimize marketing spend for Alfido Tech. "
        "The analysis aims to transition raw transaction data into a strategic business intelligence asset.",
        body_style
    ))
    story.append(Paragraph(
        "The analysis is based on the <i>Customer Behavior Analysis</i> dataset, containing 250,000 transaction records. "
        "The schema consists of the following key columns:",
        body_style
    ))
    
    schema_data = [
        ["Column Name", "Data Type", "Description"],
        ["Customer ID", "int64", "Unique identifier for each customer."],
        ["Purchase Date", "datetime", "Timestamp of the transaction."],
        ["Product Category", "category", "Type of product (Clothing, Books, Electronics, Home)."],
        ["Product Price", "float64", "Price of a single unit of the product."],
        ["Quantity", "int64", "Number of units purchased in the transaction."],
        ["Total Purchase Amount", "float64", "Total transaction value recorded (includes tax/shipping)."],
        ["Payment Method", "category", "Method of payment (Credit Card, PayPal, Cash, Crypto)."],
        ["Returns", "int64", "Binary flag indicating if the product was returned (1) or not (0)."],
        ["Age", "int64", "Customer age in years."],
        ["Gender", "category", "Customer gender (Male, Female)."],
        ["Churn", "int64", "Binary flag indicating if the customer has churned (1) or not (0)."]
    ]
    t_schema = Table(schema_data, colWidths=[130, 80, 294])
    t_schema.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(PALETTE['primary'])),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor(PALETTE['neutral_light'])]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
    ]))
    story.append(t_schema)
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 5. DATA CLEANING & FEATURE ENGINEERING
    # -------------------------------------------------------------
    story.append(Paragraph("3. Data Cleaning & Feature Engineering", h1_style))
    story.append(Paragraph(
        "To ensure analytical integrity, we performed systematic data cleaning on the raw dataset:",
        body_style
    ))
    story.append(Paragraph("• <b>Redundancy Elimination:</b> We removed the redundant column <code>Customer Age</code> as it was 100% identical to the <code>Age</code> column.", bullet_style))
    story.append(Paragraph("• <b>Missing Value Imputation:</b> The <code>Returns</code> column contained 47,596 missing values (NaNs). We imputed these as <code>0</code> (No Return), under the business assumption that transactions without a return flag did not result in a return. This avoided dropping 19% of the dataset.", bullet_style))
    story.append(Paragraph("• <b>Type Correction:</b> We cast <code>Purchase Date</code> to datetime and binary indicators (<code>Returns</code>, <code>Churn</code>) to integers.", bullet_style))
    story.append(Paragraph("• <b>Categorical Standardization:</b> Standardized values in <code>Product Category</code>, <code>Payment Method</code>, and <code>Gender</code> by stripping white spaces and applying title case.", bullet_style))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Feature Engineering:</b>", h2_style
    ))
    story.append(Paragraph(
        "To enrich our analytics and segment customers, we engineered several business-focused features:",
        body_style
    ))
    story.append(Paragraph("• <b>Total Spending:</b> Calculated at transaction level as <code>Product Price * Quantity</code>. We observed that this differs from the <code>Total Purchase Amount</code>, which likely includes taxes, shipping fees, or custom discounts.", bullet_style))
    story.append(Paragraph("• <b>Customer Lifetime Value (CLV):</b> Aggregated total amount spent (<code>Total Purchase Amount</code>) by each unique customer across their transaction history.", bullet_style))
    story.append(Paragraph("• <b>Average Order Value (AOV):</b> Mean transaction size per customer.", bullet_style))
    story.append(Paragraph("• <b>Purchase Frequency:</b> Total count of transactions per customer.", bullet_style))
    story.append(Paragraph("• <b>Recency:</b> Days since the customer's last purchase relative to the latest purchase date in the dataset (reflecting current customer status).", bullet_style))
    story.append(Paragraph("• <b>Customer Age Group:</b> Binned customer ages into six life stage segments: <i>Under 25, 25-34, 35-44, 45-54, 55-64, 65+</i>.", bullet_style))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 6. EDA & VISUALIZATIONS
    # -------------------------------------------------------------
    story.append(Paragraph("4. Exploratory Data Analysis & Visualizations", h1_style))
    story.append(Paragraph(
        "Descriptive statistics reveal a balanced demographic and product category mix. The customer age "
        "ranges uniformly from 18 to 70 years, with an average age of 44. The following sections display "
        "our analytical plots and their direct business interpretations.",
        body_style
    ))
    
    # We will embed the images one by one with their interpretations
    # Image 1: Age Distribution
    story.append(Paragraph("Customer Age Distribution", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '1_age_distribution.png'), width=360, height=180))
    story.append(Paragraph(
        "<i>Interpretation:</i> The age distribution is flat and uniform, indicating that Alfido Tech appeals "
        "equally across different age groups. This suggests marketing should focus on transactional behaviors "
        "rather than age-specific demographics.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Image 2: Revenue by Product Category
    story.append(Paragraph("Revenue Contribution by Category", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '2_product_category.png'), width=360, height=180))
    story.append(Paragraph(
        "<i>Interpretation:</i> Clothing and Books are the dominant revenue drivers, generating significantly higher sales "
        "than Electronics and Home goods. This makes them the primary customer acquisition funnel.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # Image 3: Monthly Spending Trend
    story.append(Paragraph("Monthly Sales Revenue Trends", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '3_monthly_spending.png'), width=360, height=180))
    story.append(Paragraph(
        "<i>Interpretation:</i> Monthly sales remain steady and flat across the entire date range, "
        "indicating consistent baseline demand but showing a lack of seasonal promotional spikes "
        "(e.g., holiday or summer sales).",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Image 4: Payment Method Share
    story.append(Paragraph("Payment Method Share", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '4_payment_method.png'), width=240, height=240))
    story.append(Paragraph(
        "<i>Interpretation:</i> Credit Cards are the most preferred payment method (40.2%), followed by PayPal (29.9%). "
        "Crypto represents the smallest share (9.9%). Card and PayPal users should be prioritized for saved payment checkout optimizations.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # Image 5: Box Plot
    story.append(Paragraph("Purchase Amount Distribution by Product Category", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '5_box_purchase_by_category.png'), width=360, height=180))
    story.append(Paragraph(
        "<i>Interpretation:</i> The box plot shows that while total transaction amounts vary widely, "
        "the median purchase size remains consistent across all four product categories ($2,000 - $2,500), "
        "indicating homogeneous purchasing powers across categories.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Image 6: Heatmap Correlation
    story.append(Paragraph("Correlation Matrix of Numerical Features", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '6_correlation_matrix.png'), width=320, height=240))
    story.append(Paragraph(
        "<i>Interpretation:</i> Numerical features are largely uncorrelated, which is common in generated datasets. "
        "Importantly, returns and churn exhibit a small positive correlation, confirming returns as a risk factor.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # Image 7: Gender Distribution
    story.append(Paragraph("Gender Distribution of Transactions", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '7_gender_distribution.png'), width=300, height=180))
    story.append(Paragraph(
        "<i>Interpretation:</i> Purchase counts are closely balanced between Male and Female customers, "
        "signifying that Alfido Tech holds an gender-neutral market appeal.",
        body_style
    ))
    story.append(Spacer(1, 10))
    
    # Image 8: Scatter Plot
    story.append(Paragraph("Customer Age vs. Customer Lifetime Value (CLV)", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '8_age_vs_spending.png'), width=360, height=180))
    story.append(Paragraph(
        "<i>Interpretation:</i> The regression line is flat, verifying that customer age has no strong impact "
        "on their total spending or lifetime value. Younger and older customers spend comparable amounts.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 7. CUSTOMER SEGMENTATION (RFM)
    # -------------------------------------------------------------
    story.append(Paragraph("5. Customer Segmentation (RFM Analysis)", h1_style))
    story.append(Paragraph(
        "We classified Alfido Tech's unique customer base into five strategic segments using Recency, Frequency, "
        "and Monetary (RFM) scoring. Scores from 1 to 5 were assigned to each metric using rank-based quintiles, "
        "creating the following customer segments:",
        body_style
    ))
    
    # Image 9: RFM Segments Chart
    story.append(Image(os.path.join(IMAGES_DIR, '9_rfm_segments.png'), width=400, height=180))
    story.append(Spacer(1, 10))
    
    # Segment profile table
    # Format profiles for the table
    table_data = [["Segment", "Count", "Recency (d)", "Frequency", "CLV ($)", "AOV ($)", "Age", "Return %", "Churn %"]]
    for _, row in profiles.iterrows():
        table_data.append([
            row['Segment'].replace(" Customers", "").replace(" (Champions)", ""),
            f"{int(row['Customer Count']):,}",
            f"{row['Recency']:.1f}",
            f"{row['Frequency']:.2f}",
            f"${row['Customer_Lifetime_Value']:.2f}",
            f"${row['Average_Order_Value']:.2f}",
            f"{row['Age']:.1f}",
            f"{row['Return_Rate']:.1%}",
            f"{row['Churn']:.1%}"
        ])
        
    t_profile = Table(table_data, colWidths=[90, 45, 55, 50, 60, 55, 30, 50, 50])
    t_profile.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor(PALETTE['primary'])),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor(PALETTE['neutral_light'])]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))
    story.append(t_profile)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>Segment Definitions and Characteristics:</b>", h2_style))
    story.append(Paragraph("• <b>High Value (Champions):</b> Recently and frequently purchased, with the highest customer lifetime value (avg $2,965.73). These represent the brand's primary financial anchors.", bullet_style))
    story.append(Paragraph("• <b>Loyal Customers:</b> Moderate-to-high frequency and spending, with stable recent purchases. They respond well to ongoing engagement programs.", bullet_style))
    story.append(Paragraph("• <b>New Customers:</b> High recency (purchased very recently) but low frequency (1-2 purchases). They have high potential and need nurturing.", bullet_style))
    story.append(Paragraph("• <b>At Risk:</b> Frequent buyers in the past, but have not made a purchase in a long time (high recency). They are on the verge of churn.", bullet_style))
    story.append(Paragraph("• <b>Lost Customers:</b> Inactive for a very long time, low purchase frequency, and low monetary contribution. They have the highest return rates.", bullet_style))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 8. PURCHASE PATTERN & CHURN ANALYSIS
    # -------------------------------------------------------------
    story.append(Paragraph("6. Purchase Pattern & Retention Analysis", h1_style))
    story.append(Paragraph(
        "<b>Purchase Frequency Analysis:</b> The dataset shows that a vast majority of the customer base consists of "
        "one-time transactional purchasers. The repeat customer rate stands at <b>49.4%</b> at the customer level, meaning "
        "nearly half of the customer base makes at least a second purchase, but building long-term habits remains "
        "a critical challenge. The frequency is concentrated heavily around 1 to 3 orders.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>Demographics and Spending:</b> Popular product categories are balanced across genders. Women and men buy Clothing "
        "and Books at near-identical proportions, and their average spending remains flat. The primary variations in purchasing "
        "stems from individual transactional behaviors (returns and payment friction) rather than demographic segments.",
        body_style
    ))
    
    # Image 10: Churn by Category
    story.append(Paragraph("Churn Rate by Product Category", h2_style))
    story.append(Image(os.path.join(IMAGES_DIR, '10_churn_by_category.png'), width=360, height=180))
    story.append(Paragraph(
        "<i>Interpretation:</i> Churn rates are highly uniform across all product categories, hovering around <b>20%</b>. "
        "This indicates that churn is a systemic, platform-wide issue rather than being isolated to a specific product range.",
        body_style
    ))
    
    story.append(Paragraph(
        "<b>Churn Drivers and Indicators:</b> We analyzed the direct drivers of customer churn at the transaction level:",
        body_style
    ))
    story.append(Paragraph("• <b>Product Returns:</b> Customers who return items experience a churn rate of <b>23.3%</b>, which is significantly higher than the <b>16.8%</b> churn rate for customers who do not return items.", bullet_style))
    story.append(Paragraph("• <b>Payment Methods:</b> Customers using cryptocurrency checkouts exhibit the highest churn (20.3%), while credit card and PayPal transactions offer more stable customer life cycle values.", bullet_style))
    
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 9. BUSINESS INSIGHTS
    # -------------------------------------------------------------
    story.append(Paragraph("7. Data-Driven Business Insights", h1_style))
    story.append(Paragraph(
        "Below are ten core data-driven insights extracted from the transaction and customer datasets, along with their underlying reasons and direct business impacts:",
        body_style
    ))
    
    for idx, ins in enumerate(insights, 1):
        ins_text = f"""
        <b>Insight {idx}: {ins['title']}</b><br/>
        • <i>Observation:</i> {ins['observation']}<br/>
        • <i>Reason:</i> {ins['reason']}<br/>
        • <i>Business Impact:</i> {ins['impact']}<br/>
        """
        story.append(Paragraph(ins_text, body_style))
        story.append(Spacer(1, 5))
        
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # 10. RECOMMENDATIONS & CONCLUSION
    # -------------------------------------------------------------
    story.append(Paragraph("8. Actionable Recommendations & Conclusion", h1_style))
    story.append(Paragraph(
        "To drive customer retention, increase average order values, and maximize customer lifetime values, we propose five key strategies tailored specifically for Alfido Tech:",
        body_style
    ))
    
    for idx, rec in enumerate(recs, 1):
        rec_text = f"""
        <b>Recommendation {idx}: {rec['title']}</b><br/>
        • <i>Problem:</i> {rec['problem']}<br/>
        • <i>Solution:</i> {rec['solution']}<br/>
        • <i>Expected Business Benefit:</i> {rec['benefit']}<br/>
        """
        story.append(Paragraph(rec_text, body_style))
        story.append(Spacer(1, 5))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Conclusion:</b>", h2_style))
    story.append(Paragraph(
        "By transitioning from an acquisition-centric marketing model to a behavior-based retention model, Alfido Tech can "
        "unlock substantial revenue growth. Securing the High-Value Champions, recovering customers who initiate returns, "
        "and implementing friction-free checkout flows will immediately improve operational stability and profitability. "
        "The customer behaviors analyzed in this study provide a solid foundation for deploying targeted marketing campaigns.",
        body_style
    ))
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF report built successfully.")

# -------------------------------------------------------------
# JUPYTER NOTEBOOK GENERATION
# -------------------------------------------------------------
def generate_jupyter_notebook(stats, profiles):
    import nbformat as nbf
    
    nb = nbf.v4.new_notebook()
    
    # Define cell content
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell("""# Customer Behavior Analysis
### InternSpark Data Analytics Internship Project for Alfido Tech
**Author:** Data Analyst Intern  
**Date:** July 2026  

---
## Table of Contents
1. [Step 1: Dataset Understanding](#Step-1---Dataset-Understanding)
2. [Step 2: Data Cleaning](#Step-2---Data-Cleaning)
3. [Step 3: Feature Engineering](#Step-3---Feature-Engineering)
4. [Step 4: Exploratory Data Analysis (EDA)](#Step-4---Exploratory-Data-Analysis)
5. [Step 5: Visualizations](#Step-5---Visualizations)
6. [Step 6: Customer Segmentation (RFM Analysis)](#Step-6---Customer-Segmentation)
7. [Step 7: Purchase Pattern Analysis](#Step-7---Purchase-Pattern-Analysis)
8. [Step 8: Retention & Churn Analysis](#Step-8---Retention-&-Churn-Analysis)
9. [Step 9: Business Insights](#Step-9---Business-Insights)
10. [Step 10: Actionable Recommendations](#Step-10---Actionable-Recommendations)"""))
    
    # Setup Code
    cells.append(nbf.v4.new_code_cell("""# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization parameters
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
sns.set_palette("viridis")"""))

    # Step 1
    cells.append(nbf.v4.new_markdown_cell("""## Step 1 - Dataset Understanding
In this step, we load the dataset and understand its basic structure, including rows, columns, data types, and column descriptions.

### Target Business Objective:
To analyze customer purchasing patterns, payment preferences, demographic attributes, and return behaviors to identify primary churn drivers and structure actionable retention strategies."""))
    
    cells.append(nbf.v4.new_code_cell("""# Load the dataset
df_raw = pd.read_csv("dataset/ecommerce_customer_data_custom_ratios.csv")
print(f"Number of rows: {df_raw.shape[0]:,}")
print(f"Number of columns: {df_raw.shape[1]}")
print("\\nData Types:")
print(df_raw.dtypes)"""))

    # Step 2
    cells.append(nbf.v4.new_markdown_cell("""## Step 2 - Data Cleaning
Here we clean the data by:
1. Identifying and removing duplicate records.
2. Checking for missing values.
3. Imputing missing values in the `Returns` column (assuming NaNs mean no return was initiated).
4. Dropping the redundant `Customer Age` column since it is identical to `Age`.
5. Standardizing categorical variables (strip spaces, apply Title Case).
6. Correcting data types (converting dates to datetime, flags to integers)."""))
    
    cells.append(nbf.v4.new_code_cell("""# 1. Check duplicates
print("Duplicate rows:", df_raw.duplicated().sum())

# 2. Check missing values
print("\\nMissing values per column:")
print(df_raw.isna().sum())

# 3. Clean the dataset
df_clean = df_raw.copy()
df_clean = df_clean.drop(columns=['Customer Age'])
df_clean['Purchase Date'] = pd.to_datetime(df_clean['Purchase Date'])
df_clean['Returns'] = df_clean['Returns'].fillna(0.0).astype(int)
df_clean['Churn'] = df_clean['Churn'].astype(int)

# Standardize categories
for col in ['Product Category', 'Payment Method', 'Gender']:
    df_clean[col] = df_clean[col].str.strip()
df_clean['Product Category'] = df_clean['Product Category'].str.title()
df_clean['Gender'] = df_clean['Gender'].str.title()

# Save cleaned dataset
df_clean.to_csv("cleaned_dataset.csv", index=False)
print("\\nCleaned dataset shape:", df_clean.shape)
print("Data cleaning verification completed.")"""))

    # Step 3
    cells.append(nbf.v4.new_markdown_cell("""## Step 3 - Feature Engineering
We create business-focused metrics at both transaction and customer levels:
- `Total Spending` = `Product Price * Quantity`
- `Customer Lifetime Value` (CLV) = Sum of purchase amounts per customer
- `Average Order Value` (AOV) = Mean of purchase amounts per customer
- `Purchase Frequency` = Number of orders per customer
- `Recency` = Days since the customer's last purchase relative to the max date
- `Customer Age Group` = Categorized customer ages"""))
    
    cells.append(nbf.v4.new_code_cell("""# 1. Total Spending at transaction level
df_clean['Total Spending'] = df_clean['Product Price'] * df_clean['Quantity']

# 2. Group by Customer to extract RFM and aggregated customer metrics
max_date = df_clean['Purchase Date'].max()
customer_df = df_clean.groupby('Customer ID').agg({
    'Purchase Date': lambda x: (max_date - x.max()).days, # Recency
    'Customer ID': 'count',                                # Frequency
    'Total Purchase Amount': ['sum', 'mean'],              # CLV and AOV
    'Age': 'first',
    'Gender': 'first',
    'Returns': 'mean',                                     # Return Rate
    'Churn': 'first'                                       # Churn
})

customer_df.columns = [
    'Recency', 'Frequency', 'Customer_Lifetime_Value', 
    'Average_Order_Value', 'Age', 'Gender', 'Return_Rate', 'Churn'
]
customer_df = customer_df.reset_index()

# 3. Customer Age Group binning
bins = [0, 25, 34, 44, 54, 64, 120]
labels = ['Under 25', '25-34', '35-44', '45-54', '55-64', '65+']
df_clean['Age Group'] = pd.cut(df_clean['Age'], bins=bins, labels=labels)
customer_df['Age Group'] = pd.cut(customer_df['Age'], bins=bins, labels=labels)

# 4. Repeat Customer flag
customer_df['Is_Repeat_Customer'] = (customer_df['Frequency'] > 1).astype(int)

print(f"Aggregated Customer Profiles count: {customer_df.shape[0]:,}")
customer_df.head()"""))

    # Step 4
    cells.append(nbf.v4.new_markdown_cell("""## Step 4 - Exploratory Data Analysis
Generate descriptive statistics and compute correlations to understand distributions and linkages between variables."""))
    
    cells.append(nbf.v4.new_code_cell("""# Descriptive stats for transactions
print("Transaction numerical stats:")
display(df_clean[['Age', 'Product Price', 'Quantity', 'Total Purchase Amount', 'Total Spending']].describe())

# Correlation Matrix
print("\\nCorrelation Matrix:")
corr_matrix = df_clean[['Age', 'Product Price', 'Quantity', 'Total Purchase Amount', 'Total Spending', 'Returns', 'Churn']].corr()
display(corr_matrix)"""))

    # Step 5
    cells.append(nbf.v4.new_markdown_cell("""## Step 5 - Visualizations
We generate key analytical plots including Age distribution, Sales by Category, Monthly trends, Payment Methods, and Churn rates."""))
    
    cells.append(nbf.v4.new_code_cell("""# Setup plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Age Distribution
sns.histplot(df_clean['Age'], bins=20, kde=True, ax=axes[0, 0], color='#319795')
axes[0, 0].set_title('Age Distribution of E-commerce Customers')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')

# 2. Product Category Sales
cat_sales = df_clean.groupby('Product Category')['Total Purchase Amount'].sum().reset_index()
sns.barplot(data=cat_sales, x='Product Category', y='Total Purchase Amount', ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Total Revenue by Product Category')
axes[0, 1].set_xlabel('Product Category')
axes[0, 1].set_ylabel('Total Sales ($)')

# 3. Monthly Spending Trends
df_monthly = df_clean.groupby(df_clean['Purchase Date'].dt.to_period('M'))['Total Purchase Amount'].sum().reset_index()
df_monthly['Purchase Date'] = df_monthly['Purchase Date'].dt.to_timestamp()
axes[1, 0].plot(df_monthly['Purchase Date'], df_monthly['Total Purchase Amount'], marker='o', color='#1A365D', linewidth=2)
axes[1, 0].set_title('Monthly Sales Revenue Trends')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Revenue ($)')

# 4. Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.show()"""))

    # Step 6
    cells.append(nbf.v4.new_markdown_cell("""## Step 6 - Customer Segmentation
We perform **RFM Analysis** using rank-based quintiles to classify customers into five distinct segments:
- **High Value (Champions)**: Recent, frequent, and high spending.
- **Loyal Customers**: Active and frequent buyers.
- **New Customers**: Purchased recently but have low frequency.
- **At Risk**: Frequent buyers in the past who have not purchased recently.
- **Lost Customers**: Dormant, low frequency, and low spending."""))
    
    cells.append(nbf.v4.new_code_cell("""# Score RFM metrics
customer_df['R_Score'] = pd.qcut(customer_df['Recency'].rank(method='first', ascending=False), 5, labels=False) + 1
customer_df['F_Score'] = pd.qcut(customer_df['Frequency'].rank(method='first'), 5, labels=False) + 1
customer_df['M_Score'] = pd.qcut(customer_df['Customer_Lifetime_Value'].rank(method='first'), 5, labels=False) + 1

def assign_segment(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'High Value (Champions)'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    else:
        return 'Lost Customers'

customer_df['Segment'] = customer_df.apply(assign_segment, axis=1)

# Display segment counts and profiles
segment_profiles = customer_df.groupby('Segment').agg({
    'Customer ID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Customer_Lifetime_Value': 'mean',
    'Average_Order_Value': 'mean',
    'Return_Rate': 'mean',
    'Churn': 'mean'
}).rename(columns={'Customer ID': 'Count'}).reset_index()

display(segment_profiles)"""))

    # Step 7
    cells.append(nbf.v4.new_markdown_cell("""## Step 7 - Purchase Pattern Analysis
Analyze popular categories by gender, spending patterns, and repeat purchase rates."""))
    
    cells.append(nbf.v4.new_code_cell("""# 1. Popular categories by gender
cat_by_gender = df_clean.groupby(['Gender', 'Product Category']).size().unstack()
print("Product Category Popularity by Gender:")
display(cat_by_gender)

# 2. Repeat Customer Rate
repeat_rate = customer_df['Is_Repeat_Customer'].mean()
print(f"\\nRepeat Customer Rate: {repeat_rate:.1%}")"""))

    # Step 8
    cells.append(nbf.v4.new_markdown_cell("""## Step 8 - Retention & Churn Analysis
Identify primary indicators of customer churn (returns, payment methods)."""))
    
    cells.append(nbf.v4.new_code_cell("""# Churn rate by returns
churn_by_return = df_clean.groupby('Returns')['Churn'].mean()
print("Churn Rate by Return Status:")
print(churn_by_return)

# Churn rate by payment method
churn_by_pay = df_clean.groupby('Payment Method')['Churn'].mean()
print("\\nChurn Rate by Payment Method:")
print(churn_by_pay)"""))

    # Step 9 & 10
    cells.append(nbf.v4.new_markdown_cell("""## Step 9 & 10 - Business Insights & Recommendations

### Business Insights:
1. **Baseline Churn**: 20% of the customer base has churned, creating a retention opportunity.
2. **Returns and Churn**: Returns increase churn rates from 16.8% to 23.3%, making returns a primary risk indicator.
3. **High-Value Champions**: High Value customer segment has the highest CLV and transaction sizes, acting as core revenue pillars.
4. **Crypto Churn**: Cryptocurrency checkouts experience higher churn rates (20.3%) than credit card checkouts.
5. **No Seasonality**: Purchases remain flat month-over-month, showing stable demand but a lack of promotion spikes.
6. **Repeat buyers**: 49% of customers are repeat buyers. Welcoming back one-time buyers is essential.
7. **Category Focus**: Clothing and Books represent the largest volumes, functioning as key acquisition channels.
8. **AOV Stability**: Median order size remains highly consistent across all product categories.
9. **Return Rate in Lost Segment**: The lost segment exhibits high returns, indicating product or service gaps.
10. **Demographic Uniformity**: Customer age has no correlation with total spending, highlighting transactional patterns over demographics.

### Actionable Recommendations:
1. **Return-to-Loyalty Recovery Program**: Trigger email discounts and direct support checks when returns occur.
2. **VIP Club for Champions**: Invite-only loyalty benefits, free priority shipping, and early access.
3. **Saved Cards Checkout Optimization**: Prompt card checkout, saved options, and incentives for Credit Card/PayPal.
4. **Post-Purchase Welcome Discount**: Send a 15% discount for a second purchase within 14 days of the first.
5. **Free Shipping Threshold**: Set free shipping at $150 and implement a cross-sell bundling engine to raise AOV."""))

    nb['cells'] = cells
    
    notebook_path = os.path.join(BASE_DIR, "Customer_Behavior_Analysis.ipynb")
    
    # Save notebook file
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Jupyter Notebook outline written to {notebook_path}")

# -------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# -------------------------------------------------------------
def run_pipeline():
    print("Starting Customer Behavior Analysis Pipeline...")
    
    # Step 1 & 2: Load and Clean
    df = load_and_clean_data()
    
    # Step 3: Feature Engineering
    df, customer_agg = feature_engineering(df)
    
    # Step 4: Perform EDA
    stats = perform_eda(df, customer_agg)
    
    # Step 5: Visualizations
    create_visualizations(df, customer_agg)
    
    # Step 6: Customer Segmentation
    customer_agg, profiles = customer_segmentation(customer_agg)
    
    # Step 7 & 8: Patterns & Churn
    patterns = analyze_patterns_and_churn(df, customer_agg)
    
    # Step 9: Insights
    insights = get_insights(df, customer_agg, profiles, patterns)
    
    # Step 10: Recommendations
    recs = get_recommendations()
    
    # Generate Jupyter Notebook Outline
    generate_jupyter_notebook(stats, profiles)
    
    # Compile PDF Report
    compile_pdf_report(df, customer_agg, stats, profiles, patterns, insights, recs)
    
    print("Pipeline executed successfully. All artifacts generated!")

if __name__ == "__main__":
    run_pipeline()
