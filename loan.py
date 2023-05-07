from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
df = pd.read_csv('loan.csv')
df = df.fillna(df.mean())
    
    # Apply LabelEncoder to categorical columns only
le = LabelEncoder()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    
df['z_score']=(df['ApplicantIncome'] - df['ApplicantIncome'].mean())/df['ApplicantIncome'].std()
dfup= df[(df['z_score']>-3) & (df['z_score']<3)]
    
    # Load the trained model
X = dfup.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = dfup['Loan_Status']
lc=LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
lc.fit(X_train, y_train)
@app.route("/")
def home():
    
    
    # Serialize and save the trained model
    with open("model.pkl", "wb") as file:
        pickle.dump(lc, file)

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the form
    gender = request.form["gender"]
    married = request.form["married"]
    dependents = request.form["dependents"]
    education = request.form["education"]
    self_employed = request.form["self_employed"]
    applicant_income = request.form["applicant_income"]
    coapplicant_income = request.form["coapplicant_income"]
    loan_amount = request.form["loan_amount"]
    loan_amount_term = request.form["loan_amount_term"]
    credit_history = request.form["credit_history"]
    property_area = request.form["property_area"]

    # Load the trained model
    with open("model.pkl", "rb") as file:
        lc = pickle.load(file)

    # Convert the input values to a dictionary
    input_data = {"Gender": gender,
                  "Married": married,
                  "Dependents": dependents,
                  "Education": education,
                  "Self_Employed": self_employed,
                  "ApplicantIncome": float(applicant_income),
                  "CoapplicantIncome": float(coapplicant_income),
                  "LoanAmount": float(loan_amount),
                  "Loan_Amount_Term": float(loan_amount_term),
                  "Credit_History": float(credit_history),
                  "Property_Area": property_area}

    # Convert the input data to a DataFrame and apply one-hot encoding
    input_df = pd.DataFrame([input_data])
    columns_to_encode = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    input_df_encoded = pd.get_dummies(input_df, columns=columns_to_encode)
    input_df_encoded = input_df_encoded.reindex(columns=X.columns, fill_value=0)

    # Make a prediction using the trained model
    predicted_status = lc.predict(input_df_encoded)[0]

    # Display the predicted loan status on the result page
    return render_template("result.html", predicted_status=predicted_status)


if __name__ == "__main__":
    app.run(debug=True)
