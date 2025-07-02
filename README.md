# 📊 DSA-Tool: SDG Classifier App

Aplikasi klasifikasi otomatis SDG (Sustainable Development Goals) berbasis model BERT yang telah di-fine-tune.

Model: [`degolcen/sdg-bert-model`](https://huggingface.co/degolcen/sdg-bert-model)

## 🔧 Fitur Aplikasi
- Input teks manual: masukkan judul dan/atau abstrak → muncul prediksi SDG
- Input file CSV: upload file berisi kolom `title` dan `abstract` → dapatkan prediksi SDG dalam bentuk file

## 🚀 Cara Menjalankan Lokal
1. Install dependensi:
   ```bash
   pip install -r requirements.txt
