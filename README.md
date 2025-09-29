#  Master's Thesis Project  
**Predicting Mental Health Risks in Video Gamers**  
*Tilburg University, MSc Data Science and Society*  

---

## Overview  
This repository contains the code and data for my Master's Thesis, which investigates the prediction of mental health risks in video gamers using **machine learning, deep learning, and NLP-based feature engineering**.  

The main contribution of this work is the integration of **sentence embeddings** from survey open-text responses (e.g., *Why do you play?*, *Playstyle*, *Earnings*, *League*) into predictive models of mental health.  

---

## Dataset  
- **Participants**: ~13,000 gamers  
- **Features**:  
  - Psychological assessments (e.g., Satisfaction with Life Scale - SWL)  
  - Gaming behavior (hours, motivations, playstyle, earnings)  
  - Open-text survey responses → embedded with `all-MiniLM-L6-v2` + PCA (25 components)  
- **Target variable**: `mental_health_risk` (Low / Moderate / High), derived from SWL thresholds  

---

## Methods  
1. **Preprocessing**  
   - Missing values imputed with MICE  
   - PCA applied to reduce embeddings dimensionality  
   - Countries filtered (≥ 50 responses kept, others grouped as *Other*)  

2. **Modeling**  
   - Logistic Regression  
   - Random Forest  
   - Support Vector Machines (SVM)  
   - XGBoost  
   - LightGBM  
   - CatBoost  
   - TabNet  

3. **Evaluation Metrics**  
   - **Primary**: F1 Macro (class imbalance)  
   - Accuracy, Precision, Recall, AUC (OvR)  

---

## 📈 Results  
- **Top performing models**: Logistic Regression, CatBoost, LightGBM   
- Strong predictions for **Low** and **High** risk groups; **Moderate** risk group remains challenging  
- Interpretability performed via **SHAP** to identify key features  

