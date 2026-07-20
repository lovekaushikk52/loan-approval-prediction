
from flask import Flask, render_template,request
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    gender = request.form["Gender"]
    married = request.form["Married"]
    dependents = request.form["Dependents"]
    education = request.form["Education"]
    self_employed = request.form["Self_Employed"]

    applicant_income = float(request.form["ApplicantIncome"])
    coapplicant_income = float(request.form["CoapplicantIncome"])
    loan_amount = float(request.form["LoanAmount"])
    loan_amount_term = float(request.form["Loan_Amount_Term"])
    credit_history = float(request.form["Credit_History"])
    property_area = request.form["Property_Area"]

    # encoding
    gender = 1 if gender == "Male" else 0

    # Married
    married = 1 if married == "Yes" else 0

    # Dependents
    if dependents == "3+":
        dependents = 3
    else:
        dependents = int(dependents)

    # Education
    education = 0 if education == "Graduate" else 1

    # Self Employed
    self_employed = 1 if self_employed == "Yes" else 0

    # Property Area
    if property_area == "Rural":
        property_area = 0
    elif property_area == "Semiurban":
        property_area = 1
    else:
        property_area = 2

    features = [[gender,married,dependents,education,self_employed,applicant_income,coapplicant_income,loan_amount,
                 loan_amount_term,credit_history,property_area]]
    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Congrats your Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)