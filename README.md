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

### Performance Comparison
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Weighted Avg F1 |
|-------|----------|-------------------|------------------|-------------------|----------------|
| Median Imputation | 0.6802 | 0.3675 | 0.6187 | 0.4611 | 0.7037 |
| Linear Regression Imputation | 0.6813 | 0.3691 | 0.6217 | 0.4632 | 0.7048 |
| Non-Linear Regression Imputation | 0.6815 | 0.3693 | 0.6217 | 0.4634 | 0.7049 |
| Listwise Deletion | 0.6900 | 0.3804 | 0.6494 | 0.4798 | 0.7133 |

### Key Insights
1. **Listwise Deletion** shows slightly better performance metrics but reduces dataset size from 1327 to 1178 samples
2. **Non-Linear Regression Imputation** performs marginally better than linear regression
3. All imputation methods preserve the full dataset while maintaining comparable performance
4. The small performance differences suggest that the missing data mechanism is well-handled by all methods

## Recommendations

**Preferred Method: Non-Linear Regression Imputation** 


### Why Non-Linear Regression Imputation?
1. **Data Retention**: Maintains all 30,000 data points
2. **Flexibility**: Captures complex, non-linear relationships in the data
3. **Performance**: Slightly better than linear regression imputation
4. **Generalizability**: Avoids biases from data deletion
5. **Robustness**: Preserves statistical power and minority class representation

### Trade-off Considerations
- **Listwise Deletion**: Better metrics but risks information loss and selection bias
- **Linear Regression**: Simpler but may miss complex patterns
- **Non-Linear Regression**: Best balance of performance and data preservation
