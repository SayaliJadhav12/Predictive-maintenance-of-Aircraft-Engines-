# app.py — Streamlit PdMS dashboard with Comprehensive Data Preparation Module

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Create required folders
os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/embeddings", exist_ok=True)
os.makedirs("artifacts/processed_data", exist_ok=True)
os.makedirs("artifacts/logs", exist_ok=True)

st.set_page_config(page_title="PdMS Dashboard", layout="wide")

# ---------------------------
# PLACEHOLDER MODEL FUNCTIONS
# ---------------------------

def train_cnn_and_save(train_path, test_path, feature_cols, context_length, weights=None, epochs=5):
    time.sleep(1)
    return {"train_acc":0.997,"test_acc":0.996,"train_f1":0.997,"test_f1":0.996}

def extract_embeddings_and_save(model_path, df_path, feature_cols, context_length, out_dir):
    time.sleep(1)
    emb=np.random.randn(100,64); labels=np.random.randint(0,2,100)
    np.save(f"{out_dir}/train_embeddings.npy",emb)
    np.save(f"{out_dir}/train_labels.npy",labels)
    np.save(f"{out_dir}/test_embeddings.npy",emb)
    np.save(f"{out_dir}/test_labels.npy",labels)
    np.save(f"{out_dir}/unit_embeddings.npy",emb[:50])
    np.save(f"{out_dir}/unit_labels.npy",labels[:50])

def train_t5_on_embeddings(train_emb,test_emb,epochs=3):
    time.sleep(1)
    return {"train_acc":0.998,"test_acc":0.997}


# -------------------------------
# Sidebar : Upload
# -------------------------------
st.sidebar.title("Dataset & Preprocess")
uploaded_train=st.sidebar.file_uploader("Upload TRAIN CSV",type=["csv"])
uploaded_test =st.sidebar.file_uploader("Upload TEST CSV",type=["csv"])

context_length=st.sidebar.number_input("Context length",value=15,min_value=5,max_value=100)
fake_units_train=st.sidebar.number_input("Fake Units (Train)",value=1000,min_value=1)
fake_units_test =st.sidebar.number_input("Fake Units (Test)",value=400,min_value=1)

default_sensors=[
    "sensor_measurement_2","sensor_measurement_3","sensor_measurement_4",
    "sensor_measurement_7","sensor_measurement_11","sensor_measurement_12",
    "sensor_measurement_15","sensor_measurement_20","sensor_measurement_21"
]

feature_cols=st.sidebar.multiselect("Pick Sensor Columns",options=default_sensors,default=default_sensors)

if uploaded_train and uploaded_test:
    with open("uploaded_train.csv","wb") as f:f.write(uploaded_train.getbuffer())
    with open("uploaded_test.csv","wb") as f:f.write(uploaded_test.getbuffer())
    st.sidebar.success("Files uploaded successfully ✔")
else:
    st.sidebar.info("Upload both datasets to start")


# ==========================================================
#     MAIN TABS  (Added New One - 📌 Data Preparation)
# ==========================================================

tabs = st.tabs([
    "Data Preview", "Data Preparation", "Train CNN", "Extract Embeddings",
    "Train T5 (Hybrid)", "Compare & Visualize", "Download & Logs"
])

# ----------------------------------------------------------
# TAB 1: Data Preview
# ----------------------------------------------------------
with tabs[0]:
    st.header("📄 Raw Data Preview")

    if uploaded_train:
        df=pd.read_csv("uploaded_train.csv")
        st.subheader("Train Sample"); st.dataframe(df.head())
        st.write(f"Rows: {len(df)}")

    if uploaded_test:
        df=pd.read_csv("uploaded_test.csv")
        st.subheader("Test Sample"); st.dataframe(df.head())
        st.write(f"Rows: {len(df)}")


# ----------------------------------------------------------
# ✅ TAB 2: COMPREHENSIVE DATA PREPARATION MODULE
# ----------------------------------------------------------
with tabs[1]:
    st.header("🛠 Comprehensive Data Preparation for PdMS")

    if uploaded_train:

        df=pd.read_csv("uploaded_train.csv")
        st.write("### Step 1: Handle Missing Values")
        method=st.selectbox("Choose method",["Drop Rows","Fill Mean","Fill Median"])

        if st.button("Clean Missing Values"):
            if method=="Drop Rows": df=df.dropna()
            elif method=="Fill Mean": df=df.fillna(df.mean())
            else: df=df.fillna(df.median())
            st.write("Cleaned Data Preview"); st.dataframe(df.head())

        st.write("### Step 2: Scale Sensor Data")
        if st.button("Normalize Data (MinMaxScaler)"):
            scaler=MinMaxScaler()
            df[feature_cols]=scaler.fit_transform(df[feature_cols])
            st.success("Scaling Applied ✔")
            st.dataframe(df.head())

        st.write("### Step 3: Generate PdM Window/Units")
        window=st.number_input("Window size",value=15)
        if st.button("Create Sequence Windows for CNN"):
            sequences=[]; labels=[]
            for i in range(len(df)-window):
                sequences.append(df[feature_cols].iloc[i:i+window].values)
                labels.append(0)  # simulation placeholder
            np.save("artifacts/processed_data/train_windows.npy", np.array(sequences))
            st.success("Sequence Windows Generated & Saved ✔")

        if st.button("Save Prepared Dataset"):
            df.to_csv("artifacts/processed_data/prepared_train.csv",index=False)
            st.success("Prepared dataset saved successfully")


# ----------------------------------------------------------
# TAB 3: CNN Training
# ----------------------------------------------------------
with tabs[2]:
    st.header("🚀 Train CNN Model")
    if st.button("Run CNN Training"):
        with st.spinner("Training CNN..."):
            metrics=train_cnn_and_save("uploaded_train.csv","uploaded_test.csv",feature_cols,context_length)
        st.success("Training Finished ✔"); st.json(metrics)


# ----------------------------------------------------------
# TAB 4: Embeddings
# ----------------------------------------------------------
with tabs[3]:
    st.header("📌 Extract Embeddings")
    if st.button("Extract Embeddings"):
        extract_embeddings_and_save("artifacts/models/cnn_model.pt","uploaded_train.csv",feature_cols,context_length,"artifacts/embeddings")
        st.success("Embeddings Stored ✔")
        st.write(os.listdir("artifacts/embeddings"))


# ----------------------------------------------------------
# TAB 5: T5 Hybrid
# ----------------------------------------------------------
with tabs[4]:
    st.header("🤖 Train T5 on CNN Embeddings")
    if os.path.exists("artifacts/embeddings/train_embeddings.npy"):
        if st.button("Train Hybrid T5"):
            metrics=train_t5_on_embeddings("artifacts/embeddings/train_embeddings.npy","artifacts/embeddings/test_embeddings.npy")
            st.success("T5 Training Complete ✔"); st.json(metrics)
    else:
        st.warning("Extract embeddings first")


# ----------------------------------------------------------
# TAB 6: Compare & Visualize (Added ROC Curve, PR Curve, Confusion Matrix)
# ----------------------------------------------------------
with tabs[5]:
    st.header("📊 Model Comparison & System Curves")
    st.write("Visualize performance of Hybrid model")

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

    # Check embeddings & labels exist
    if os.path.exists("artifacts/embeddings/train_embeddings.npy") and os.path.exists("artifacts/embeddings/train_labels.npy"):

        emb = np.load("artifacts/embeddings/train_embeddings.npy")
        labels = np.load("artifacts/embeddings/train_labels.npy")

        # Simulated scores for AUC curves (in real use → model.predict_proba)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit(emb, labels)   # Temporary evaluation model
        scores = clf.predict_proba(emb)[:,1]
        preds = clf.predict(emb)

        # =================== ROC CURVE =====================
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax1.plot([0,1],[0,1],'--',color="gray")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        st.pyplot(fig1)

        # ================= Precision-Recall Curve =================
        precision, recall, _ = precision_recall_curve(labels, scores)
        fig2, ax2 = plt.subplots()
        ax2.plot(recall, precision)
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        st.pyplot(fig2)

        # ================= Confusion Matrix Plot =================
        cm = confusion_matrix(labels, preds)
        fig3, ax3 = plt.subplots()
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax3, cmap="Blues")
        ax3.set_title("Confusion Matrix")
        st.pyplot(fig3)

        # ================= PCA Plot (Already present but improved) =============
        st.subheader("Embedding Cluster Visualization (PCA)")
        from sklearn.decomposition import PCA
        reduced = PCA(n_components=2).fit_transform(emb)

        fig4, ax4 = plt.subplots()
        ax4.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="coolwarm", s=10)
        ax4.set_title("Embedding Projection")
        st.pyplot(fig4)
        
    else:
        st.info("⚠ Generate embeddings first from Step-4")


# ----------------------------------------------------------
# TAB 7: Artifacts & Logs
# ----------------------------------------------------------
with tabs[6]:
    st.header("📁 Artifacts & Logs")
    st.subheader("Models"); st.write(os.listdir("artifacts/models"))
    st.subheader("Embeddings"); st.write(os.listdir("artifacts/embeddings"))
    st.subheader("Processed Data"); st.write(os.listdir("artifacts/processed_data"))