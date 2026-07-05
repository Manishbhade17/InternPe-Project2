# Car Price Predictor

Predicts the resale price of a used car based on company, model, year of purchase, fuel type, and kilometres driven.

## How to run (PyCharm / local)

1. Open this folder as a project in PyCharm.
2. Create a virtual environment (PyCharm will usually prompt you, or: `python -m venv venv`).
3. Activate it and install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run `application.py`.
5. Open the URL shown in the terminal (typically `http://127.0.0.1:5000`).

## Project structure

```
car_price_predictor/
├── application.py              # Flask app (routes + prediction logic)
├── templates/
│   └── index.html              # Front-end form + JS fetch call
├── Cleaned_Car_data.csv        # Cleaned dataset used for dropdowns + training
├── quikr_car.csv               # Raw scraped data (before cleaning)
├── LinearRegressionModel.pkl   # Trained sklearn Pipeline (OneHotEncoder + LinearRegression)
├── train_model.py              # Script used to retrain the model
├── requirements.txt
└── Procfile                    # For deployment (e.g. Render/Heroku-style platforms)
```

## Model details

- Pipeline: `ColumnTransformer(OneHotEncoder on name/company/fuel_type) -> LinearRegression`
- Retrained on current scikit-learn to avoid the "InconsistentVersionWarning" / pickle
  incompatibility that happens when a model trained on an old sklearn version is loaded
  with a newer one.
- R² score on held-out test data: ~0.78 (varies slightly depending on train/test split).

## Common issues this version fixes

- **Old pinned versions in requirements.txt** (Flask 1.1.2, scikit-learn 0.22.2, etc. from
  2020) no longer install cleanly with current Python — replaced with modern minimum-version
  pins.
- **Missing `templates/index.html`** — the original repo referenced it but didn't ship it,
  so `render_template("index.html", ...)` failed with a `TemplateNotFound` error.
- **Pickle/sklearn version mismatch** — the model is retrained fresh so it matches whatever
  scikit-learn version `pip install -r requirements.txt` gives you.
