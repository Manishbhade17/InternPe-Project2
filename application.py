from pathlib import Path

import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS

# --------------------------------------------------
# Flask App
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load Model and Dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

model = pickle.load(open(BASE_DIR / "LinearRegressionModel.pkl", "rb"))
car = pd.read_csv(BASE_DIR / "Cleaned_Car_data.csv")

# --------------------------------------------------
# Prepare Dropdown Data
# --------------------------------------------------

companies = sorted(car["company"].unique())
years = sorted(car["year"].unique(), reverse=True)
fuel_types = sorted(car["fuel_type"].dropna().unique())

# Dictionary: { "Hyundai": ["Hyundai Grand i10", ...], "Ford": [...] }
car_models = {}
for company in companies:
    models = sorted(car.loc[car["company"] == company, "name"].unique())
    car_models[company] = list(models)


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        car_models=car_models
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        company = request.form["company"]
        car_model = request.form["car_models"]
        year = int(request.form["year"])
        kms_driven = int(request.form["kilo_driven"])
        fuel_type = request.form["fuel_type"]

        data = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"]
        )

        prediction = model.predict(data)[0]
        return str(round(float(prediction), 2))

    except Exception as e:
        print(e)
        return "Unable to predict price."


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
