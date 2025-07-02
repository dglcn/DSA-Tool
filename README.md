# 🌍 DSA-Tool: Deep SDG Analyzer

DSA-Tool adalah aplikasi berbasis Streamlit untuk mengklasifikasikan teks akademik (judul dan abstrak jurnal) ke dalam kategori Tujuan Pembangunan Berkelanjutan (SDGs). Aplikasi ini menggunakan model BERT yang telah di-*fine-tune* untuk multi-label classification berdasarkan data jurnal ilmiah.

---

## 🚀 Fitur

- ✅ **Input Teks Manual**  
  Masukkan teks (judul atau abstrak jurnal) dan dapatkan prediksi Top-3 SDG secara instan.

- 📁 **Upload File CSV**  
  Unggah file CSV berisi kumpulan teks jurnal (satu kolom saja), dan dapatkan hasil prediksi SDG secara batch.

- 📥 **Unduh Hasil Prediksi**  
  Hasil prediksi dapat diunduh langsung dalam format `.csv` dengan kolom tambahan "predicted_sdgs".

---

## 🧠 Model yang Digunakan

Model BERT telah dilatih khusus untuk klasifikasi multi-label 17 SDG menggunakan dataset akademik.  
Model tersedia di Hugging Face:  
👉 [`degolcen/sdg-bert-model`](https://huggingface.co/degolcen/sdg-bert-model)

## Akses Aplikasi: https://dsa-tool.streamlit.app/
