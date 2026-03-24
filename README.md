# Customer Churn Prediction and Risk Intelligence System (Machine Learning)

## Overview
This project presents an end-to-end machine learning system designed to predict customer churn and generate actionable retention insights. The system leverages structured customer data to estimate churn probability, classify customers into risk categories, and provide business-oriented recommendations.  

The solution is built using classical machine learning techniques and is deployed as an interactive Streamlit application for real-time inference.

---

## Problem Statement
Customer churn is a critical challenge for subscription-based businesses, directly impacting revenue and growth. The objective of this project is to:

- Predict the likelihood of customer churn  
- Identify high-risk customers  
- Translate model predictions into actionable business strategies  

---

## Dataset
- **Telco Customer Churn Dataset**  
- Contains customer demographics, service usage, and billing information  
- Target variable: **Churn (Yes/No)**  

---

## Project Pipeline

### 1. Data Ingestion
- Loaded dataset from CSV  
- Inspected structure, columns, and data types  

### 2. Data Cleaning and Preprocessing
- Converted `TotalCharges` to numeric format  
- Handled missing values  
- Encoded categorical variables using one-hot encoding  
- Standardized numerical features using scaling  

### 3. Exploratory Data Analysis (EDA)
- Analyzed churn distribution  
- Identified key drivers of churn:  
  - Short tenure  
  - High monthly charges  
  - Month-to-month contracts  
- Visualized feature relationships using count plots, histograms, and boxplots  

### 4. Feature Engineering
- Created model-ready feature set after encoding  
- Ensured consistency between training and inference features  

### 5. Model Development
- Trained a **Logistic Regression classifier**  
- Generated churn probabilities using `predict_proba`  

### 6. Model Evaluation
- Evaluated performance using:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion matrix  
- Analyzed trade-offs between precision and recall  
- Tuned decision thresholds to improve churn detection (recall)  

### 7. Risk Scoring System
- Converted probabilities into business-friendly categories:  
  - **Low Risk:** probability < 0.3  
  - **Medium Risk:** 0.3 – 0.7  
  - **High Risk:** > 0.7  

### 8. Recommendation Layer
- Implemented rule-based logic to generate retention strategies:  
  - **High Risk →** discounts and retention offers  
  - **Medium Risk →** engagement strategies  
  - **Low Risk →** maintain service quality  

### 9. Deployment
- Built an interactive **Streamlit application**  
- Users can input customer attributes and receive:  
  - Churn probability  
  - Risk classification  
  - Recommended action  

---

## Project Structure
.
├── notebook.ipynb        # Full ML pipeline (EDA, preprocessing, modeling)
├── app.py                # Streamlit application
├── model.pkl             # Trained Logistic Regression model
├── scaler.pkl            # Feature scaler
├── features.pkl          # Feature schema for inference consistency
├── README.md             # Project documentation

---

## How to Run the Application

### 1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  

### 2. Install Dependencies
pip install -r requirements.txt  

### 3. Run Streamlit App
streamlit run app.py  

The application will open in your browser at: **http://localhost:8501**

---

## Key Features
- End-to-end machine learning pipeline  
- Real-time churn prediction  
- Interpretable risk segmentation  
- Actionable business recommendations  
- Interactive user interface using Streamlit  

---

## Key Insights
- Customers with shorter tenure are more likely to churn  
- Higher monthly charges are associated with increased churn risk  
- Month-to-month contracts exhibit significantly higher churn compared to long-term contracts  

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  

---

## Future Improvements
- Incorporate additional models (e.g., Random Forest, Gradient Boosting)  
- Improve feature engineering with domain-specific features  
- Integrate model explainability tools (e.g., SHAP)  
- Deploy application to a cloud platform for public access  
- Expand input interface to capture full customer feature set  

---

## Conclusion
This project demonstrates how machine learning can be applied to a real-world business problem by not only predicting outcomes but also translating them into actionable decisions. The system bridges the gap between model outputs and business strategy, making it relevant for both technical and non-technical stakeholders.
