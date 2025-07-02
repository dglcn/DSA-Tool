import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_NAME = "degolcen/sdg-bert-model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🔍 SDG Classifier")
st.write("Masukkan judul dan/atau abstrak jurnal untuk prediksi SDG (Sustainable Development Goals).")

text_input = st.text_area("✏️ Masukkan teks:")

if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze().numpy()

        top_indices = np.argsort(probs)[-3:][::-1]
        st.subheader("Top-3 Prediksi SDG:")
        for idx in top_indices:
            st.write(f"✅ SDG {idx + 1} — Score: {probs[idx]:.2f}")
