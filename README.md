# Flight Delay Prediction (ML project, Arrival Delay)

![EDA](https://img.shields.io/badge/EDA-Exploratory%20Data%20Analysis-purple)
![Feature Engineering](https://img.shields.io/badge/Feature-Engineering-blueviolet)
![Class Imbalance](https://img.shields.io/badge/Class%20Imbalance-SMOTE-orange)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458)
![Scikit-Learn](https://img.shields.io/badge/Library-ScikitLearn-F7931E)
![Seaborn](https://img.shields.io/badge/Library-Seaborn-9cf)

## Project Overview

This project focuses on predicting flight arrival delays using machine learning techniques for US SDF airport.  
The problem is formulated in two ways:

- **Binary classification** — predicting whether a flight arrives with a delay greater than 15 minutes. (ArrDel15 column)
- **Regression** — predicting the exact arrival delay time in minutes. (ArrDelayMinutes column)

The analysis is based on the full raw U.S. flight dataset, with a dedicated focus on **SDF (Louisville International Airport)**.  
To enhance predictive performance, the dataset is enriched with historical weather data obtained from NOAA.

The project includes:
- Data collection and merging
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Handling class imbalance (SMOTE/class weights)
- Model training and hyperparameter tuning
- Model comparison and evaluation
- Feature importance analysis and interpretation

The goal is to build reliable models capable of early delay prediction and to analyze the key factors influencing flight punctuality.

---

## Project Structure

data/ -> raw and processed datasets

**data/model_eval -> model final scores**

01-08 -> Jupyter notebooks (Merging data, preprocessing, EDA, modeling, experiments, evaluation)

models/ -> saved trained models for different tasks

img/ -> visualizations used in project

---

> [!NOTE]
> File name indicators:
> 
> - `_bal_` → model trained using class weighting (`class_weight='balanced'`)
> - `_dep_` → model trained with Departure-related information (prediction performed immediately after takeoff)
> - `_no_dep_` → model trained without any Departure-related features or potential leakage from post-departure information
> 
> _dt_, _lr_, _xgb_, _rf_, _LassoReg_, _RidgeReg_, _LinearSVR_, _linreg_ -> model names (Decision Tree, Logistic Regression, XGBoost default library, Random Forest, Lasso Regression, RidgeRegression, Linear SVR (SVM), Linear Regression)

---

> [!WARNING]
> Some code cells — particularly those related to model training — may take a considerable amount of time to complete.

---

## Dataset Description

- Full U.S. flight dataset for all 2024 flights (column descriptions available in `data/raw_data_column_info`)
- Focused subset: flights related to SDF airport in 2024  
- Additional historical weather features integrated from NOAA   
- Target variables:
  - `ArrDel15` (classification, binary)
  - `ArrDelayMinutes` (regression)

---

## Methodology

1. Data cleaning and preprocessing
2. Flight and Weather datasets merging
3. Feature engineering (time-based, route-based, weather features)
4. EDA analysis
5. Handling class imbalance  
6. Model training and cross-validation  
7. Evaluation using appropriate metrics  

---

> [!IMPORTANT]
> Model Training Strategy:
>
> - All models were trained and evaluated using **5-fold stratified cross-validation** to ensure robust and reliable performance estimation while preserving class distribution.
>
> - Hyperparameter optimization was performed through multiple sequential runs of **Grid Search**.  
>   For very large hyperparameter spaces, **Randomized Search** or **Bayesian Optimization** was used as a computationally efficient alternative.
>
> - For classification tasks, **Average Precision (AP)** was selected as the primary optimization metric.  
>   AP is generally more informative than ROC-AUC for imbalanced datasets, as it better reflects model performance on the minority class and focuses on precision–recall trade-offs.
>
> - For regression tasks, **Mean Absolute Error (MAE)** was used as the primary metric.  
>   MAE was chosen due to its robustness to extreme values and outliers, providing a more stable estimate of typical prediction error compared to squared-error metrics.

> [!NOTE]
> Resampling and Data Leakage Notice:
>
> - All over- and under-sampling techniques (including SMOTE used in this project) were applied **strictly to the training data only**, and performed separately within each cross-validation fold.
>
> - Resampling was never applied before cross-validation or on the full dataset.  
>   Performing oversampling prior to data splitting introduces **data leakage**, as synthetic samples may contain information from validation folds.
>
> - Applying SMOTE (or any resampling technique) outside the fold-wise training process is a common methodological mistake and can artificially inflate performance metrics.

## Models Used

**Classification task**
- Logistic Regression  
- Random Forest  
- XGBoost

**Regression task**
- Linear Regression
- Lasso CV Regression
- Ridge CV Regression
- Random Forest Regressor
- XGBoost Regressor
- SVM (Linear SVR)

---

## Evaluation metrics

**Classification**
- Accuracy (overall)
- Precision, Recall, F1-score
- ROC_AUC
- Precision-Recall AUC (PR_AUC/average precision score)
- CV PR_AUC (5 folds)

**Regression**
- MAE (Mean Average Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- RAE (Relative Absolute Error)
- RRSE (Root Relative Squared Error)
- R2 (R-squared)
- Adjusted R2
- Correlation coefficient (Corr)

---

> [!IMPORTANT]
> Key Notes on Model Inference and Experimental Findings:
>
> - The SMOTE technique did not lead to statistically significant improvements (p-value > 0.05) compared to standard class weight balancing (`class_weight='balanced'`) for most algorithms.  
>   This is likely due to the moderate class imbalance (approximately 80% / 20%).  
>   SMOTE tends to be more beneficial in cases of extreme imbalance (e.g., 99% / 1% or more extreme imbalance ratios).  
>   Therefore, standard class weight balancing was retained as the primary imbalance handling strategy.
>
> - The `Distance` feature is generally informative; however, since the analysis focuses exclusively on flights related to SDF airport, the average flight duration is approximately 2 hours with relatively low variance.  
>   As a result, distance did not significantly improve model performance.  
>   Its impact would likely be stronger in a larger, multi-airport dataset with greater route variability.
>
> - Weather data were obtained from a nearby NOAA meteorological station rather than directly from airport runway sensors.  
>   More precise, airport-level weather measurements could potentially improve model performance and provide additional predictive power.

---

## Future Improvements

- Real-time prediction pipeline  
- Deployment as an API  
- More advanced weather feature integration using more reliable data (for example - windspeed on the takeoff and departure runways)
- Deep learning experiments with Neural Networks
