# House Price Prediction Using XGBoost and SHAP

## Overview

This project focuses on building an interpretable machine learning model using XGBoost to predict house prices. The dataset is preprocessed and modeled using XGBoostRegressor, and SHAP (SHapley Additive exPlanations) is employed to provide global and local interpretability of the model's predictions.

## Dataset

The dataset used in this assignment is the Ames Housing Dataset, a well-known dataset for regression tasks involving house price prediction. It contains numerous features including zoning information, building type, year built, and sale price.

- Target Variable: `SalePrice`
- Features: Numeric and categorical predictors related to house properties.

## Problem Statement

The task is to:
1. Preprocess the dataset to handle missing values and categorical variables.
2. Train a powerful and efficient regression model using XGBoost.
3. Use SHAP to understand the model's predictions by visualizing feature importances and individual prediction explanations.

## Methodology

### 1. Preprocessing
- Missing values were imputed appropriately based on domain knowledge.
- Categorical variables were encoded using `OrdinalEncoder`.
- Numeric features were standardized using `StandardScaler`.

### 2. Model Training
- An `XGBRegressor` model was trained on the preprocessed dataset.
- Cross-validation was used to tune hyperparameters.

### 3. Model Interpretability with SHAP
- SHAP values were computed using `TreeExplainer`.
- Visualizations include:
  - SHAP Summary Plot
  - SHAP Dependence Plot
  - SHAP Force Plot for a single prediction

## Libraries Used

- Python 3
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `shap`
- `matplotlib`, `seaborn`

## Visualizations

- SHAP Summary Plot: Displays overall feature importance based on SHAP values.
- SHAP Dependence Plot: Shows how a feature's value affects the prediction.
- SHAP Force Plot: Provides a local explanation of a single prediction instance.

## How to Run

1. Clone the repository or download the notebook.
2. Install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
   ```
3. Open the notebook `Main.ipynb` and run all the cells.

## Results

- The XGBoost model demonstrated high accuracy on the validation set.
- SHAP helped in uncovering which features had the most influence, such as `OverallQual`, `GrLivArea`, and `YearBuilt`.

## Key Takeaways

- XGBoost is a powerful tool for regression tasks, especially when paired with robust preprocessing.
- SHAP provides a principled, game-theoretic approach to model interpretability, crucial for transparency in ML predictions.
