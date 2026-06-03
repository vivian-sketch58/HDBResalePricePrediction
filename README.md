# 🏠 End-to-End ML Pipelines for HDB Resale Price Prediction

An enterprise-grade, secure, and highly efficient machine learning pipeline designed to predict Housing & Development Board (HDB) flat resale prices in Singapore. This project delivers robust, data-driven price forecasts to bring transparency and certainty to the public housing market.

---

## 📌 Project Overview

Navigating the real estate market can be complex. This project automates the entire machine learning lifecycle - from raw data ingestion and secure preprocessing to model training, evaluation, and prediction - to accurately predict HDB resale prices. 

By transforming historical housing data into actionable insights, this pipeline suports:

* **Prospective Buyers:** Better budgeting and realistic price expectations
* **For Current Owners:** Data-driven property valuation for informed selling decisions

## 🚀 Key Features

* **End-to-End Automation:** Fully integrated pipeline from data ingestion to prediction.
* **Modular Design:** Separation of data preparation, modeling, and execution (`main.py`)
* **Scalable & Efficient:** Supports parallel processing (e.g., `n_jobs=-1`)
* **Config-Driven:** Flexible parameter tuning via `config.yaml`
* **Reproducible:** Clean `.py` pipeline (converted from notebooks)

## ⚙️ Prerequisites & Installation

### 🔹 Prerequisites
* Python 3.11+
* Anaconda / Miniconda (recommended)

### 🔹 Key Libraries
* pandas
* numpy
* scikit-learn
* PyYAML

### 🔹 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/vivian-sketch58/HDBResalePricePrediction.git
cd HDBResalePricePrediction

# 2. Create environment
conda create -n hdbenv python=3.11
conda activate hdbenv

# Install dependencies
conda install pandas numpy scikit-learn pyyaml

# 3. Or using pip:
pip install -r requirements.txt
```

## 📁 Project Repository Structure
```
HDBresalepricepredictions/
│
├── data/
│   ├── resale_transactions.csv
│
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   ├── config.yaml
│
├── .gitignore
├── eda.ipynb
├── main.py
├── requirements.txt
└── README.md
```


## 🚀 Instructions for Executing the Pipeline

### 🔹 Step 1: Configure Parameters
Edit `config.yaml` to control:
* Input/output file paths
* Feature selection
* Model hyperparameters

### 🔹 Step 2: Run the Pipeline
`python main.py`

### 🔹 Step 3: Output
* Cleaned dataset
* Trained model
* Evaluation metrics
* Predictions


## 🔄 Logical Steps/Flow of the Pipeline

The pipeline follows a structured workflow:
1. **Data Loading**
    * Load raw dataset from CSV
2. **Data Preparation**
    * Drop duplicates
    * Handle missing values    
    * Encode categorical variables
    * Scale numerical features
3. **Feature Engineering**
    * Create derived features (e.g., storey_range,remaining_lease_months)
4. **Model Training**
    * Train regression models (i.e. Linear Regression, Ridge, and lasso)
5. **Model Evaluation**
    * Evaluate using metrics such as RMSE and R²
6. **Prediction**
    * Generate resale price predictions


## 🔍 Key Findings from EDA

* Prevalence of mid-sized flats (3/4/5-room)
* Pproperty values is broad spectrum with right-skewed (higher price properties that extend the tail )
* Resale listings are influenced by seasonal or market factors rather than a consistent monthly trend
* Location (town impact) is a strong price driver
* Remaining lease significantly affects price
* Larger floor area → higher resale value
* Categorical features (flat type, town) are highly influential

## 🧩Feature Handling

| Feature Type | Processing Method | Description                            |
| ------------ | ----------------- | -------------------------------------- |
| Numerical    | StandardScaler    | Normalize features for model stability |
| Categorical  | OneHotEncoder     | Convert nominal categories             |
| Ordinal      | OrdinalEncoder    | Encode ordered features (e.g., story_level)               |
| Missing Data | Imputation | Handle null values  (e.g., town_name)                   |
| Engineered   | Custom Features   | e.g., remaining_lease_months   |


## 🤖 Model Choices

**Linear Regression**

Baseline model
Interpretable, simple and fast

**ridge regression  (L2 Regularization)**
* Reduces overfitting by penalizing large coefficients
* Helps control overfitting, especially with many correlated features
* Handles multicollinearity effectively by distributing weights across correlated variables
* Improves generalization performance on validation/test data

**lasso regression (L1 Regularization)**
* Performs automatic feature selection by shrinking some coefficients to zero
* Produces simpler and more interpretable models
* Useful when identifying the most important predictors
* Can reduce noise by eliminating irrelevant features

### 👉 Models were selected to balance:
* Interpretability
* Performance
* Scalability


## 📈 Evaluation of Models

### 🔹 Evaluation Metrics Used

- **MAE (Mean Absolute Error)**  
  Average absolute difference between predicted and actual values  

- **MSE (Mean Squared Error)**  
  Average squared difference; penalizes larger errors more heavily  

- **RMSE (Root Mean Squared Error)**  
  Square root of MSE; interpretable in the same unit as the target variable  

- **R² Score**  
  Measures the proportion of variance in the target variable explained by the model  

### 🏆 Summary of Evaluation

The **Ridge Regression model (alpha = 0.1)** achieved the best overall performance on the test set:

- **MAE:** 41,272.73  
- **MSE:** 2,896,767,086.77  
- **RMSE:** 53,821.62  
- **R² Score:** 0.8656  

This model delivered the **highest R² score**, explaining approximately **86.6% of the variance** in HDB resale prices, indicating strong predictive performance and generalization.

## 🚀 Deployment Considerations

* **Scalability:**
    * Use batch or API-based inference
* **Performance:**
    * Optimize model loading and prediction latency
* **Integration:**
    * Can integrate with web apps (e.g., Flask, FastAPI)
* **Monitoring:**
    * Track model drift and retrain periodically
* **Security:**
    * Ensure safe handling of housing data



