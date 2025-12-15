import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIGURATION ----------------
TRAIN_CSV = "D:/Desktop/POC/data/synthetic_balanced_data_20000_60_40 (1).csv"   # your training CSV
SCALER_PATH = "D:/Desktop/POC/notebooks/artifacts/models/scaler.pkl"
FEATURE_COLS = [
    "OpSet1","OpSet2","OpSet3",
    "Sensor2","Sensor3","Sensor4",
    "Sensor7","Sensor11","Sensor12",
    "Sensor15","Sensor20","Sensor21"
]

# ---------------- LOAD TRAINING DATA ----------------
train_df = pd.read_csv(TRAIN_CSV)

# Fill missing columns if any (with zeros)
for col in FEATURE_COLS:
    if col not in train_df.columns:
        train_df[col] = 0

X_train = train_df[FEATURE_COLS].values

# ---------------- FIT SCALER ----------------
scaler = StandardScaler()
scaler.fit(X_train)

# ---------------- SAVE SCALER ----------------
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

print(f"Scaler created and saved at: {SCALER_PATH}")
