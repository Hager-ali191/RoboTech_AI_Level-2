# E-Commerce Customer Churn Prediction Analysis

## 📋 Project Overview
A comprehensive machine learning project analyzing customer behavior data to predict customer churn in an e-commerce platform. This project implements a complete data science pipeline from exploratory analysis to model deployment.

## 📊 Dataset Description

### Original Features
| Feature | Type | Description | Preprocessing Applied |
|---------|------|-------------|----------------------|
| **CustomerID** | Identifier | Unique customer ID | Removed (not used in modeling) |
| **Churn** | Target | Binary (1=churned, 0=active) | None |
| **Tenure** | Numerical | Months with company | KNN imputation, outlier handling |
| **PreferredLoginDevice** | Categorical | Most used login device | One-hot encoding |
| **CityTier** | Ordinal | City classification (1-3) | Used as-is |
| **WarehouseToHome** | Numerical | Distance in km | KNN imputation, outlier capping |
| **PreferredPaymentMode** | Categorical | Payment method | Ordinal encoding |
| **Gender** | Categorical | Male/Female | One-hot encoding |
| **HourSpendOnApp** | Numerical | Daily hours on app | Mode imputation |
| **NumberOfDeviceRegistered** | Numerical | Device count | Used as-is |
| **PreferedOrderCat** | Categorical | Preferred category | Ordinal encoding |
| **SatisfactionScore** | Ordinal | Rating 1-5 | Used as-is |
| **MaritalStatus** | Categorical | Status | One-hot encoding |
| **NumberOfAddress** | Numerical | Address count | Outlier handling |
| **Complain** | Binary | Complaint flag | Used as-is |
| **OrderAmountHikeFromlastYear** | Numerical | % increase | KNN imputation |
| **CouponUsed** | Numerical | Coupon count | Median imputation |
| **OrderCount** | Numerical | Total orders | KNN imputation |
| **DaySinceLastOrder** | Numerical | Days since last order | KNN imputation, outlier handling |
| **CashbackAmount** | Numerical | Total cashback | Scaling, outlier handling |

### Engineered Features
| Feature | Formula | Purpose |
|---------|---------|---------|
| **AvgOrdersPerHour** | `HourSpendOnApp / OrderCount` | Engagement efficiency metric |
| **AvgCashBackperCategory** | `CashbackAmount / OrderCount` | Cashback effectiveness |
| **LoyaltyScore** | `1 if Tenure ≤ 12 else 2 if ≤ 24 else 3` | Customer loyalty categorization |

## 🚀 Project Pipeline

### 1. Data Loading & Initial Inspection
```python
# Code structure
data_frame = pd.read_csv("E Commerce Dataset.csv")
print(f"Dataset shape: {data_frame.shape}")
print(f"Columns: {list(data_frame.columns)}")
```
### 2. Exploratory Data Analysis (EDA)
#### 2.1 Data Quality Check
- Missing values analysis (1856 total nulls)
- Duplicate detection (556 duplicates found)
- Data types validation
- Outlier detection using box plots

#### 2.2 Univariate Analysis
- Target distribution: 16.84% churn rate
- Numerical features: Histograms with skewness metrics
- Categorical features: Bar charts with percentages
- Feature statistics summary

#### 2.3 Bivariate Analysis
- Box plots: Numerical features vs Churn
- Count plots: Categorical features vs Churn
- KDE plots: Distribution differences between churn groups
- Strip plots: Individual data points visualization

#### 2.4 Multivariate Analysis
- Correlation heatmap (numerical features)
- ANOVA for numerical features correlation with target
- Chi-square tests for categorical features correlation
- Grouped visualizations for feature interactions


### 3. Data Preprocessing Pipeline
#### 3.1 Missing Value Treatment
```python
# Strategy per feature type
null_columns_mode = ['HourSpendOnApp']
null_columns_knn_imputer = ['OrderAmountHikeFromlastYear', 'WarehouseToHome', 
                           'OrderCount', 'DaySinceLastOrder', 'Tenure']
null_columns_median = ['CouponUsed']
```

#### 3.2 Data Type Conversion
- Converted float columns to integers where appropriate
- Ensured consistent data types for modeling

#### 3.3 Outlier Handling
- IQR method with 1.5 multiplier
- Capping at lower and upper bounds
- Visual verification using box plots

#### 3.4 Feature Engineering
- Created 3 new features from existing ones
- Added domain-specific metrics
- Enhanced model's predictive power

#### 3.5 Duplicate Removal
- Removed 556 duplicate records
- Ensured data uniqueness


### 4. Feature Encoding & Scaling
#### 4.1 Encoding Strategy
```python
# Small cardinality features
encoding_columns_small = ['Gender', 'PreferredLoginDevice', 'MaritalStatus']
# One-hot encoding applied

# Large cardinality features
encoding_columns_large = ['PreferedOrderCat', 'PreferredPaymentMode']
# Ordinal encoding applied
```

#### 4.2 Scaling Methods Tested
- Standard Scaler: For normally distributed features
- Min-Max Scaler: For bounded ranges
- Robust Scaler: For outlier-robust scaling


### 5. Feature Selection
#### 5.1 Numerical Features Correlation
- ANOVA F-tests with target
- Top correlated features identified

#### 5.2 Categorical Features Correlation
- Chi-square tests with target
- Significant categorical features identified

#### 5.3 Multicollinearity Check
- Correlation matrix analysis
- Feature independence verification


### 6. Machine Learning Implementation
####6.1 Model Architecture
```
python
class MachineLearning:
    """Comprehensive ML class with methods for:
    - Data analysis and visualization
    - Preprocessing operations
    - Model training and evaluation
    - Hyperparameter tuning
    """
```

#### 6.2 Implemented Models
- Model	Library	Hyperparameter Grid
- Logistic Regression	sklearn	C, penalty, solver
- SVM	sklearn	C, kernel, gamma
- KNN	sklearn	n_neighbors, weights, metric
- Decision Tree	sklearn	criterion, max_depth, min_samples_split
- Random Forest	sklearn	n_estimators, max_depth, criterion
- XGBoost	xgboost	n_estimators, learning_rate, max_depth
- Naive Bayes	sklearn	var_smoothing

#### 6.3 Training Strategy
- Train-test split: 80-20 ratio
- Random state: 42 for reproducibility
- Cross-validation: 5-fold for hyperparameter tuning
- GridSearchCV: Exhaustive parameter search

#### 6.4 Evaluation Metrics
- Primary: Accuracy, Precision, Recall, F1-Score
- Secondary: ROC-AUC, Confusion Matrix
- Class Imbalance Handling: Balanced accuracy

### 7. Model Comparison Framework
####7.1 Evaluation Pipeline
```python
def best_parameters(self, text, model_type, params, model, 
                   x_train, x_test, y_train, y_test):
    """Unified evaluation function for all models"""
```

#### 7.2 Performance Tracking
- Train vs test performance comparison
- Overfitting/underfitting detection
- Best parameters logging
- Metric visualization

###8. Visualization System
#### 8.1 Plot Types Implemented
- Distribution plots: Histogram, KDE, Box, Violin
- Categorical plots: Bar, Count, Pie
- Relationship plots: Scatter, Strip, Swarm
- Matrix plots: Heatmap, Correlation

#### 8.2 Custom Visualization Methods
```python
def bar_plot(self, column, data_frame):  # Percentage bar plots
def histogram_plot(self, column, data_frame):  # With skewness
def count_plot(self, column, hue, data_frame):  # With percentages
def pixel_bar_plot(self, data_frame, x, label_x, label_y):  # Interactive
```


### 9. Key Technical Decisions
#### 9.1 Data Splitting
- Stratified sampling considered for class imbalance
- Random state fixed for reproducibility
- Feature-target separation before preprocessing

#### 9.2 Preprocessing Order
- Handle missing values
- Remove duplicates
- Handle outliers
- Engineer features
- Encode categorical variables
- Scale numerical features

#### 9.3 Model Selection Criteria
- Performance on test set
- Generalization capability
- Computational efficiency
- Interpretability for business

### 10. Code Organization Structure

```python
# Main sections in the notebook:
# 1. Libraries Import
# 2. MachineLearning Class Definition
# 3. Parameters Guide (hyperparameter grids)
# 4. Data Loading
# 5. Data Information & EDA
# 6. Data Analysis (Visualizations)
# 7. Preprocessing Pipeline
# 8. Model Implementation & Evaluation
# 9. Results Analysis
```
#### Technical Implementation Details
##### Custom MachineLearning Class
The project implements a custom MachineLearning class with methods for:

- Data inspection and quality checks
- Statistical analysis and visualization
- Preprocessing operations (handling nulls, outliers, encoding)
- Model training with hyperparameter optimization
- Comprehensive evaluation metrics
- Results comparison and reporting
  
##### Parameter Optimization Strategy
For each model, a comprehensive parameter grid is defined:

```python
# Example: Random Forest parameters
random_forest = {
    'n_estimators': [100, 200, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

##### Cross-Validation Approach
- 5-fold cross-validation for hyperparameter tuning
- Stratified K-Fold for classification tasks
- GridSearchCV for exhaustive parameter search
- RandomizedSearchCV for larger parameter spaces


### 📁 Project Structure

```text
e-commerce-churn-prediction/
├── notebooks/
│   └── E_Commerce_Project.ipynb      # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py         # Preprocessing functions
│   ├── feature_engineering.py        # Feature creation
│   ├── model_training.py            # Model training pipeline
│   └── visualization.py             # Plotting functions
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Cleaned dataset
├── models/                           # Saved models
├── reports/                          # Analysis reports
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```
### 🚦 Running the Project
#### Quick Start
- Install dependencies:
```
pip install -r requirements.txt
```
- Run the analysis:
```python
# Open and run the notebook cells sequentially
# or execute the main script
python src/main.py
```
- Development Setup
```
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

### 🔄 Pipeline Summary
- Data Loading → 2. EDA → 3. Preprocessing → 4. Feature Engineering → 5. Encoding/Scaling → 6. Feature Selection → 7. Model Training → 8. Hyperparameter Tuning → 9. Evaluation → 10. Results Analysis

### 📊 Output Generated
Comprehensive EDA visualizations
Model performance comparisons
Feature importance rankings
Preprocessing transformation logs
Best model parameters and metrics
