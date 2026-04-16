# 📊 Customer Behavior Analysis

## 🔍 Overview

This project analyzes customer behavior data to identify purchasing patterns, customer satisfaction trends, and churn factors. The goal is to generate actionable insights that help businesses improve customer retention and decision-making.

---

## 🎯 Problem Statement

Businesses often struggle to understand why customers leave (churn) and how different factors like satisfaction, spending, and region affect behavior. This project aims to solve that by analyzing customer data and uncovering key patterns.

---

## 🚀 Features

* 📈 Customer behavior analysis
* 🔍 Churn analysis by region
* 📊 Correlation analysis between features
* 🧩 Customer segmentation insights
* 📉 Data visualization using charts and graphs
* 📝 Business insight report generation

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## 📁 Project Structure

```
Customer-Behavior-Analysis/
│
├── data/
│   └── customer_data.csv              # Raw dataset
│
├── output/
│   ├── churn_by_region.png
│   ├── correlation_matrix.png
│   ├── satisfaction_vs_churn.png
│   ├── spend_vs_frequency.png
│   └── customer_data_segmented.csv   # Processed data
│
├── reports/
│   └── business_insight_report.md    # Final insights
│
├── src/
│   ├── main.py                      # Core analysis logic
│   ├── generate_sample_data.py      # Data generation script
│   └── web_app.py                   # (Optional) Web interface
│
├── .gitignore
└── README.md
```

---

## 🔄 Project Workflow

1. Data is collected or generated using `generate_sample_data.py`
2. Data is processed and analyzed in `main.py`
3. Visualizations are generated and stored in `output/`
4. Insights are documented in `reports/business_insight_report.md`

---

## 📊 Key Insights

* Customers with low satisfaction scores show higher churn rates
* Spending patterns vary significantly across regions
* Frequent buyers tend to have higher retention
* Strong correlation exists between satisfaction and churn

---

## 📸 Visualizations

### Churn by Region

<img width="1200" height="750" alt="churn_by_region" src="https://github.com/user-attachments/assets/1d5f4112-763f-47e8-be52-0d0507eba46c" />


### Correlation Matrix

<img width="1350" height="1050" alt="correlation_matrix" src="https://github.com/user-attachments/assets/db100d55-dc5b-4706-a7a5-07b8b0e7d689" />

### Satisfaction vs Churn

<img width="1200" height="750" alt="satisfaction_vs_churn" src="https://github.com/user-attachments/assets/0755b56d-5782-4346-9230-a2ab22eb8eb9" />


### Spend vs Frequency

<img width="1200" height="750" alt="spend_vs_frequency" src="https://github.com/user-attachments/assets/fd924d0b-5aad-407e-a892-92f42ab48311" />


---

## ▶️ How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/JeslinSajan/Customer-Behavior-Analysis.git
```

### 2. Navigate to project folder

```
cd Customer-Behavior-Analysis
```

### 3. Install dependencies

```
pip install pandas numpy matplotlib seaborn
```

### 4. Run analysis

```
python src/main.py
```

---

## 🌐 Web App

Run the interactive dashboard using:

streamlit run src/web_app.py

## 💡 Future Improvements

* Implement machine learning models for churn prediction
* Add customer segmentation using clustering (K-Means)
* Build an interactive dashboard (Streamlit)
* Deploy as a web application

---

## 👨‍💻 Author

**Jeslin Sajan**
