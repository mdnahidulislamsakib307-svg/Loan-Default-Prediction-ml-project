#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


df = pd.read_csv("C:\\Users\\USER-11\\Downloads\\loan_default_dataset (1).csv")


# In[26]:


df.shape


# In[27]:


df.dtypes


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


bins = [300, 500, 600, 700, 800, 900]
labels = ['300-500', '500-600', '600-700', '700-800', '800-900']
df['credit_score_range'] = pd.cut(df['credit_score'], bins=bins, labels=labels)
default_rate_cs = df.groupby('credit_score_range', observed=False)['loan_default'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=default_rate_cs, x='credit_score_range',y ='loan_default',  hue='loan_default')
plt.title('Default Rate by Credit Score Range')
plt.ylabel('Default Rate')
plt.xlabel('Credit Score Range')
plt.savefig('bar_plot_credit_score.png')
plt.show()


# In[8]:


avg_income_default = df.groupby('loan_default')['income'].mean().reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(data=avg_income_default, x='loan_default', y='income',hue='income')
plt.title('Average Income by Loan Default Status')
plt.ylabel('Average Income')
plt.xlabel('Loan Default (0: No, 1: Yes)')
plt.savefig('bar_plot_avg_income.png')
plt.show()


# In[9]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='income', y='loan_amount', hue='loan_default', alpha=0.6)
plt.title('Income vs. Loan Amount')
plt.savefig('scatter_plot_income_loan.png')
plt.show()


# In[10]:


plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('heatmap_correlation.png')
plt.show()


# In[11]:


df['dti_rounded'] = df['debt_to_income_ratio'].round(1)
dti_trend = df.groupby('dti_rounded')['loan_default'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=dti_trend, x='dti_rounded', y='loan_default')
plt.title('Debt-to-Income Ratio vs. Default Probability')
plt.ylabel('Default Probability')
plt.xlabel('Debt-to-Income Ratio')
plt.grid(True)
plt.savefig('line_plot_dti.png')
plt.show()


# In[12]:


x = df.drop(['loan_default'],axis = 1)
y = df['loan_default']


# In[13]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[14]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[15]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


# In[16]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[17]:


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[19]:


model = Pipeline(steps=[
    ('pre',preprocessor),('reg',RandomForestClassifier(n_estimators=200,random_state=42))
])


# In[20]:


model.fit(X_train,y_train)


# In[21]:


y_pred = model.predict(X_test)
print(f'accuracy:{accuracy_score(y_pred,y_test)*100:2f}')
print(f'{classification_report(y_pred,y_test,zero_division=0)}')


# In[92]:


jb.dump(model,'RandomForestClassifier.pkl')


# In[104]:


import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Input Schema
# -----------------------------
class LoanData(BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_term_months: int
    credit_score: int
    employment_years: int
    debt_to_income_ratio: float
    num_credit_lines: int
    past_delinquencies: int
    credit_score_range: str
    dti_rounded: float

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("RandomForestClassifier.pkl")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    description="ML API for predicting loan default risk",
    version="1.0"
)

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}

# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict")
def predict(data: LoanData):

    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "High Risk - Customer may Default"
    else:
        result = "Low Risk - Customer likely to Repay"

    return {
        "prediction": int(prediction),
        "result": result
    }


# In[106]:


load = jb.load('RandomForestClassifier.pkl')

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Loan Default Prediction App")

st.write(
    """
    This app predicts whether a customer is likely to **default on a loan** using a trained Machine Learning model.
    Fill in the customer information and click **Predict**.
    """
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age')
    income = st.number_input('Income')
    loan_amount = st.number_input('Loan Amount')
    credit_score = st.number_input('Credit Score')
    employment_years = st.number_input('Employment Years')

with col2:
    loan_term_months = st.number_input('Loan Term (Months)')
    debt_to_income_ratio = st.number_input('Debt to Income Ratio')
    num_credit_lines = st.number_input('Number of Credit Lines')
    past_delinquencies = st.number_input('Past Delinquencies')
    dti_rounded = st.number_input('DTI Rounded')
    
credit_score_range = st.selectbox(
    "Credit_Score_Range",
    ['300-500','500-600','600-700','700-800','800-900']
)

st.markdown("---")

if st.button("🔍 Predict Loan Status"):
    data = pd.DataFrame({   
        'age':[age],
        'income':[income],
        'loan_amount':[loan_amount],
        'loan_term_months':[loan_term_months],
        'credit_score':[credit_score],
        'employment_years':[employment_years],
        'debt_to_income_ratio':[debt_to_income_ratio],
        'num_credit_lines':[num_credit_lines],
        'past_delinquencies':[past_delinquencies],
        'credit_score_range':[credit_score_range],
        'dti_rounded':[dti_rounded]
    })


    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Customer may Default")
    else:
        st.success("✅ Low Risk: Customer likely to Repay Loan")


st.sidebar.write("""
Model: Random Forest  
Type: Classification  
Use Case: Loan Risk Prediction
""")

st.markdown("---")
st.write("Developed by Nahidul Islam | Data Science & ML")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




