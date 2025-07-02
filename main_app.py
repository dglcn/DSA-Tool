import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "degolcen/sdg-bert-model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🌍 DSA-Tool: SDG Classifier")

mode = st.radio("Pilih mode input:", ["📝 Input Teks Manual", "📄 Upload File CSV"])

if mode == "📝 Input Teks Manual":
    st.subheader("Masukkan Judul dan/atau Abstrak Jurnal")
    text_input = st.text_area("✏️ Teks:")

    if st.button("🔍 Prediksi"):
        if text_input.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).squeeze().numpy()

            top_indices = np.argsort(probs)[-3:][::-1]
            st.success("Top-3 SDG Prediksi:")
            for idx in top_indices:
                st.write(f"✅ SDG {idx + 1} — Score: {probs[idx]:.2f}")

elif mode == "📄 Upload File CSV":
    st.subheader("Unggah file CSV (kolom: title, abstract)")
    uploaded_file = st.file_uploader("📁 Pilih file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding="utf-8", errors="ignore")

        if not {"title", "abstract"}.issubset(df.columns):
            st.error("File harus memiliki kolom 'title' dan 'abstract'.")
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
            st.success("Prediksi selesai! Hasil ditampilkan di bawah:")
            st.dataframe(df[["title", "abstract", "predicted_sdgs"]])

            csv_output = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil", data=csv_output, file_name="sdg_predictions.csv", mime="text/csv")
