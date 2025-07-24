import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.1, 0.05, 0.001],
    "subsample": [0.8, 1.0]
}

model = GradientBoostingRegressor()

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

joblib.dump(grid.best_params_, "models/best_params.pkl")

print("Meilleurs paramètres :", grid.best_params_)
