#  SkySatisfy — Airline Passenger Satisfaction Prediction

A machine learning mini-pipeline that predicts **airline passenger satisfaction** using structured survey + flight data.  
This project is implemented in **two notebooks** where **Airline Passenger Satisfaction — Feature Scaling & Feature Selection Optimization is a continuation and enhancement of Airline Passenger Satisfaction — Data Cleaning & Baseline Logistic Regression**.

---

## 📌 What this project does
✅ Loads and explores the airline dataset  
✅ Handles missing values (e.g., removing rows with missing arrival delay)  
✅ Encodes categorical features (Gender, Customer Type, Type of Travel, Class)  
✅ Detects outliers (IQR + visual analysis)  
✅ Applies feature scaling (MinMax 0–1)  
✅ Trains a **Logistic Regression** classifier  
✅ Improves the model using **feature selection techniques**:
- RFE / RFECV (feature elimination)
- SelectKBest (top-k important features)
- Correlation filtering (removing highly correlated features)

---

## 🧠 Notebooks Overview (Part 1 → Part 2)

### **1) Airline Passenger Satisfaction — Data Cleaning & Baseline Logistic Regression (Assignment1.ipynb)**
**Focus:** cleaning and building the first baseline model  
Key steps:
- Missing values removal (Arrival Delay in Minutes)
- Encoding target + categorical features
- Outlier detection (IQR + histograms/boxplots)
- Baseline Logistic Regression training and evaluation
- Scaling features using MinMaxScaler (0–1)

✅ Output: baseline performance + cleaned/scaled dataset experiments

---

### **2) Airline Passenger Satisfaction — Feature Scaling & Feature Selection Optimization (Assignment2.ipynb)**
**This notebook continues Airline Passenger Satisfaction — Data Cleaning & Baseline Logistic Regression** with stronger preprocessing and feature selection.  
Key steps:
- Feature encoding + scaling (MinMax)
- Train/test split (80/20)
- Logistic Regression training & evaluation
- Feature selection experiments:
  - **RFE / RFECV**
  - **SelectKBest**
  - **Correlation heatmap** + removing highly correlated features

✅ Output: comparison of feature selection strategies (in notebook results, SelectKBest gives ~80% accuracy depending on k/features)
