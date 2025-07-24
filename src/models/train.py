import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

X = pd.read_csv("data/processed_data/X_train_scaled.csv")
y = pd.read_csv("data/processed_data/y_train.csv")
best_params = joblib.load("models/best_params.pkl")

model = GradientBoostingRegressor(**best_params)
model.fit(X, y)

joblib.dump(model, "models/gbr_model.pkl")
