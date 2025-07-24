import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

X_train_scaled_df.to_csv("data/processed_data/X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv("data/processed_data/X_test_scaled.csv", index=False)
