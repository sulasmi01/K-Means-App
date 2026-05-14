# ☀️ SunCluster System

Aplikasi analisis dan pengelompokan produk sunscreen menggunakan algoritma **K-Means Clustering**.

## Fitur
- Upload dataset CSV/Excel
- Normalisasi data Min-Max Scaling
- Penentuan jumlah cluster optimal (Elbow Method)
- Proses clustering K=4 dengan visualisasi iterasi
- Evaluasi cluster (WCV & BCV)
- Visualisasi sebaran produk

## Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Teknologi
- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Matplotlib
