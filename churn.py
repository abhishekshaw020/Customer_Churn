# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the Streamlit app
st.title("Customer Churn Prediction App")

# Load sample data if no file is uploaded
@st.cache_data
def load_sample_data():
    data = pd.read_csv('Churn_Modelling.csv')  # Use your dataset here
    return data

# Preprocessing function
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# Train the model
def train_model(X, y, model_type="logistic"):
    if model_type == "logistic":
        model = LogisticRegression(random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns  # Return the model and the feature names

# Predict churn for new data
def predict(model, input_data, feature_names):
    input_data = input_data[feature_names]  # Ensure correct feature order
    return model.predict(input_data)

# Visualizations
def plot_churn_distribution(data):
    churn_counts = data['Exited'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette='viridis')
    plt.title('Churn Distribution')
    plt.xlabel('Churn (1 = Yes, 0 = No)')
    plt.ylabel('Number of Customers')
    plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn'])
    st.pyplot(plt)

def plot_feature_importance(model, feature_names):
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        st.pyplot(plt)

def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Sidebar for input options
st.sidebar.header("Upload Data or Input Manually")

# File uploader for CSV
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# If file uploaded, load data
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(data.head())

    # Preprocess the data
    data_cleaned = preprocess_data(data)
    
    # Split into features and target
    X = data_cleaned.drop('Exited', axis=1)
    y = data_cleaned['Exited']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train a model and capture feature names
    model_type = st.sidebar.selectbox("Select Model Type", ["logistic", "random_forest"])
    model, feature_names = train_model(X_train, y_train, model_type=model_type)
    
    # Predict on the test set
    y_pred = predict(model, X_test, feature_names)
    
    # Show evaluation results
    st.write("Model Evaluation:")
    st.text(classification_report(y_test, y_pred))
    st.write("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    # Visualizations
    st.subheader("Visualizations")
    plot_churn_distribution(data_cleaned)
    plot_feature_importance(model, feature_names)
    plot_roc_curve(model, X_test, y_test)

else:
    # If no file is uploaded, load the sample data
    st.write("Using sample data for training the model.")
    data = load_sample_data()
    
    # Preprocess the sample data
    data_cleaned = preprocess_data(data)
    
    # Split into features and target
    X = data_cleaned.drop('Exited', axis=1)
    y = data_cleaned['Exited']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train a model and capture feature names
    model_type = st.sidebar.selectbox("Select Model Type", ["logistic", "random_forest"])
    model, feature_names = train_model(X_train, y_train, model_type=model_type)

    # Manual input for customer data
    st.subheader("Enter Customer Data for Prediction:")
    
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    Tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
    Balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0)
    NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
    EstimatedSalary = st.number_input("Estimated Salary", min_value=10000.0, max_value=200000.0, value=50000.0)

    # Create DataFrame for the input
    input_df = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography_Germany': [1 if Geography == "Germany" else 0],
        'Geography_Spain': [1 if Geography == "Spain" else 0],
        'Gender': [1 if Gender == "Male" else 0],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    # Standardize the input data (using the same scaler as training)
    input_df[feature_names] = StandardScaler().fit(X_train).transform(input_df[feature_names])

    # Button for making predictions
    if st.button("Predict Churn"):
        # Make prediction
        prediction = predict(model, input_df, feature_names)
        
        # Show the prediction result
        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
