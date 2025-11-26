# 🫀 Heart Disease Prediction – End-to-End Machine Learning Project

A complete, production-style machine learning pipeline for predicting the likelihood of heart disease using clinical features from the UCI Heart Disease dataset.

This project demonstrates **practical ML engineering skills** expected from a junior ML / Data Science role, including:
- Data quality checks
- Feature engineering & preprocessing
- Cross-validation & hyperparameter tuning
- Proper threshold selection
- Model explainability (SHAP)
- Final evaluation on a clean test set
- Production-oriented considerations

---

## 📌 Project Summary

**Objective:**  
Predict whether a patient has heart disease based on 13 clinical features.

**Dataset:**  
UCI Heart Disease (Cleveland) — 302 rows after deduplication.

**Best Model:**  
Tuned **XGBoost** classifier with:
- One-Hot Encoding for categorical features  
- Standardization for numeric features  
- Cross-validated hyperparameter tuning  
- Validation-based threshold optimization  

**Key Metric (Test Set):**
- **ROC-AUC:** ~0.88–0.92  
- **F1 (tuned threshold):** Higher than default  
- **Recall:** Improved at tuned threshold (important for medical use cases)

---


---

## 🧹 1. Data Quality & EDA

Includes:
- Missing value check
- Duplicate detection  
  → Original CSV contained 700+ duplicates due to source issues  
  → Cleaned dataset has **302** rows (correct UCI size)
- Categorical vs numeric identification
- Visual EDA:
  - Numeric feature distributions
  - Categorical features vs target
  - Correlation analysis

---

## 🔧 2. Preprocessing

Using `ColumnTransformer` inside scikit-learn Pipelines:

**Categorical (One-Hot Encoded):**
- sex
- cp
- fbs
- restecg
- exang
- slope
- ca
- thal

**Numeric (Standard Scaled):**
- age
- trestbps
- chol
- thalach
- oldpeak

This ensures:
- No data leakage
- Consistent transformations during CV, tuning, evaluation, and prediction

---

## 🔍 3. Baselines & Model Comparison

Models used:
- Dummy Classifier (baseline)
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Compared using **Stratified 5-fold Cross-Validation** with **F1 score**.

---

## 🎯 4. Hyperparameter Tuning

Performed via **RandomizedSearchCV** on the XGBoost pipeline.

Search space includes:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- Regularization parameters (`reg_lambda`)

Scoring metric: **F1**

---

## ⚖️ 5. Threshold Tuning

Instead of using default threshold 0.5:
- Created a **validation split** from training data
- Computed precision-recall curve
- Selected threshold that maximizes **F1**
- Applied tuned threshold to test set for final evaluation

This demonstrates understanding of:
- Medical domain recall-precision trade-offs
- Proper threshold selection methodology

---

## 📊 6. Final Evaluation

Metrics reported:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion matrix (default vs tuned threshold)
- ROC and PR curves

The tuned threshold improves **recall**, which is important for detecting disease.

---

## 🧠 7. Model Explainability (SHAP)

Provided:
- Global interpretation (SHAP summary & bar plots)
- Local explanation (patient-level contribution)

**Key insights:**
- `thalach` (max heart rate) decreases risk
- `oldpeak` increases risk
- Chest pain type (`cp_*`) strongly influences prediction
- Number of colored vessels (`ca`) is a major risk factor

Interpretation included to demonstrate ability to communicate ML insights.

---

## 🚀 8. Production Sketch

Included:
- Simple prediction function:

```python
def predict_risk(input_df):
    proba = best_model.predict_proba(input_df)[0, 1]
    return proba

