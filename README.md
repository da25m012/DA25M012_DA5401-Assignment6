# DA25M012_DA5401-Assignment6
Data Analytics Lab Assignment-6 - Imputation via Regression for Missing Data

## Author
Name : G C V Sairam  
Roll No : DA25M012  
Department : Data Science and Artificial Intelligence, M.Tech  

## Problem Statement

We are working on a credit card risk assessment project with a dataset containing missing values in several important feature columns. The presence of missing data prevents immediate application of classification methods. We need to implement various strategies for handling this missing data and then use the resulting clean datasets to train and evaluate a classification model.

## Dataset

- **Source**: UCI Credit Card Dataset
- **Size**: 30,000 data points with 25 columns
- **Target Variable**: `default.payment.next.month` (binary classification)
- **Features**: Demographic information, payment history, bill amounts, and payment amounts

## Files

- `DA25M012_DALab_Assgn6.ipynb`: Main Jupyter notebook with complete implementation
- `UCI_Credit_Card.csv`: Dataset file


## Project Structure

### Part A: Data Preprocessing and Imputation

#### 1. Data Preparation
- Load and explore the dataset
- Artificially introduce MAR (Missing At Random) values in `BILL_AMT2` and `PAY_AMT5` columns
- Missingness depends on the `AGE` column using a sigmoid probability function

#### 2. Imputation Strategies

**Strategy 1: Baseline Imputation (Median)**
- Replace missing values with median of available data
- Preferred over mean imputation due to robustness to outliers

**Strategy 2: Linear Regression Imputation**
- Use linear regression to predict missing values in `BILL_AMT2`
- Train on complete cases, predict missing values
  
**Strategy 3: Non-Linear Regression Imputation**
- Use Decision Tree Regressor for more complex relationships
- Handle non-linear patterns in the data

**Strategy 4: Listwise Deletion**
- Remove all rows with missing values
- Serves as baseline comparison (reduces dataset from 30,000 to 26,751 rows)

### Part B: Model Training and Performance Assessment

- Split each imputed dataset into training (80%) and testing (20%) sets
- Apply feature scaling using StandardScaler
- Train Logistic Regression models with class balancing
- Evaluate using classification reports and key metrics

### Part C: Comparative Analysis

Comprehensive comparison of all four approaches across multiple performance metrics.

## Key Results

## 📊 Results Summary

| Model | Accuracy | Macro F1 | Weighted F1 | Notes |
|:------|:---------:|:---------:|:------------:|:------|
| Median Imputation | 0.6828 | 0.6204 | 0.7063 | Simple but effective baseline |
| Linear Regression Imputation | 0.6858 | **0.6235** | **0.7078** | Best overall performance |
| Non-Linear Regression Imputation | 0.6837 | 0.6200 | 0.7086 | Slightly overfits; no clear gain |
| Listwise Deletion | 0.6844 | 0.6296 | 0.7063 | Reduced data size, minor drop in recall |


## 🧠 Efficacy Discussion (Summary)
- **Listwise Deletion** discards data, which slightly reduces generalization capability due to smaller sample size.  
- **Imputation-based models** preserve full data, improving stability and balance.  
- **Linear Regression Imputation** provides the best trade-off between bias and variance, outperforming both simpler (median) and more complex (non-linear) approaches.  
- The relationship between predictors and the imputed feature appears largely **linear**, explaining why non-linear methods did not offer a performance advantage.

---

## ✅ Final Recommendation
**Linear Regression Imputation** is recommended as the optimal approach for this dataset.  
It:
- Retains all observations without introducing bias from data loss.  
- Captures relationships among features for more realistic imputations.  
- Provides the highest and most stable performance across metrics.  

Thus, it offers an ideal balance between **data completeness, interpretability, and predictive accuracy**.
