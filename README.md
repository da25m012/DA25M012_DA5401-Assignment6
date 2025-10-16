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
- Use Random Forest Regressor for more complex relationships
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

### Performance Comparison Table

| Model | Accuracy | Precision_0 | Recall_0 | F1_0 | Precision_1 | Recall_1 | F1_1 | Macro_Avg_F1 | Weighted_Avg_F1 | Support_1 |
|-------|----------|-------------|----------|------|-------------|----------|------|--------------|-----------------|-----------|
| Median Imputation | 0.6830 | 0.8629 | 0.7009 | 0.7735 | 0.3801 | 0.6223 | 0.4720 | 0.6227 | 0.7049 | 1366 |
| Linear Regression Imputation | 0.6845 | 0.8703 | 0.6968 | 0.7740 | 0.3805 | 0.6420 | 0.4778 | 0.6259 | 0.7074 | 1349 |
| Non-Linear Regression Imputation | 0.6777 | 0.8786 | 0.6851 | 0.7699 | 0.3585 | 0.6502 | 0.4622 | 0.6160 | 0.7043 | 1278 |
| Listwise Deletion | 0.6926 | 0.8796 | 0.7013 | 0.7804 | 0.3861 | 0.6619 | 0.4877 | 0.6341 | 0.7157 | 1183 |

## Key Findings

1. **Listwise Deletion** performed best overall, achieving the highest accuracy (0.6926) and F1-score for class 1 (0.4877)
2. **Linear Regression Imputation** showed slightly better performance than median imputation
3. **Non-Linear Regression Imputation** underperformed compared to simpler methods
4. All models showed good performance for class 0 (non-default) but struggled with class 1 (default) prediction
5. The recall for class 1 was consistently higher than precision, indicating better sensitivity than precision for default prediction
