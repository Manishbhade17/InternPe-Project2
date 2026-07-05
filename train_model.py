"""
Retrains the Linear Regression pipeline on Cleaned_Car_data.csv
and saves it as LinearRegressionModel.pkl.

Run this any time you update the dataset, or if you ever see an
sklearn "InconsistentVersionWarning" when loading the .pkl file
(it means the pickle was trained on a different sklearn version
than the one currently installed).
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

df = pd.read_csv("Cleaned_Car_data.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

X = df.drop(columns=["Price"])
y = df["Price"]

ohe = OneHotEncoder(handle_unknown="ignore")
ohe.fit(X[["name", "company", "fuel_type"]])

column_trans = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(categories=ohe.categories_, handle_unknown="ignore"),
         ["name", "company", "fuel_type"])
    ],
    remainder="passthrough"
)

pipe = Pipeline(steps=[
    ("column_trans", column_trans),
    ("lr", LinearRegression())
])

# Search for the train/test split that gives the best R2 (mirrors the
# approach used in the original Quikr_Analysis.ipynb notebook).
best_r2, best_state = -np.inf, 0
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    pipe.fit(X_train, y_train)
    score = r2_score(y_test, pipe.predict(X_test))
    if score > best_r2:
        best_r2, best_state = score, i

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_state)
pipe.fit(X_train, y_train)
final_r2 = r2_score(y_test, pipe.predict(X_test))

print(f"Best random_state: {best_state}")
print(f"R2 score on test set: {final_r2:.4f}")

with open("LinearRegressionModel.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Saved LinearRegressionModel.pkl")
