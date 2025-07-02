import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "degolcen/sdg-bert-model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📄 Batch SDG Classifier")
st.write("Unggah file CSV yang berisi kolom `title` dan `abstract`.")

uploaded_file = st.file_uploader("Unggah CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not {"title", "abstract"}.issubset(df.columns):
        st.error("File harus mengandung kolom 'title' dan 'abstract'.")
    else:
        texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
        results = []

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).squeeze().numpy()
            top_sdgs = np.argsort(probs)[-3:][::-1] + 1
            results.append(", ".join([f"SDG {i}" for i in top_sdgs]))

        df["predicted_sdgs"] = results
        st.success("Prediksi selesai! Unduh hasilnya di bawah.")
        st.dataframe(df[["title", "abstract", "predicted_sdgs"]])

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Hasil", data=csv_download, file_name="sdg_predictions.csv", mime="text/csv")
