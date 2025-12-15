# preprocess_pm.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(input_file="data/ai4i2020.csv", output_dir="data/"):
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(input_file)

    # Drop IDs
    df = df.drop(columns=["UDI", "Product ID"])

    # Encode categorical feature 'Type'
    le = LabelEncoder()
    df["Type"] = le.fit_transform(df["Type"])

    # Features and Target
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save everything
    joblib.dump((X_train, X_test, y_train, y_test), f"{output_dir}/processed.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(le, f"{output_dir}/encoder.pkl")

    print("✅ Data preprocessed and saved!")

if __name__ == "__main__":
    preprocess_data()
