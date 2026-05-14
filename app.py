import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations

# =====================================
# 1. CONFIG & INITIALIZATION
# =====================================
st.set_page_config(
    page_title="SunCluster System",
    page_icon="☀️",
    layout="wide"
)

for key in ["df_raw", "df_norm", "df_clustered", "kmeans", "iteration_tables"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =====================================
# 2. CORE FUNCTIONS
# =====================================

def hitung_wcv(X, labels, centroids):
    wcv = 0
    for i in range(len(X)):
        cluster_index = labels[i]
        wcv += np.sum((X[i] - centroids[cluster_index]) ** 2)
    return wcv

def hitung_bcv(centroids):
    bcv = 0
    if len(centroids) < 2: return 0
    for i, j in combinations(range(len(centroids)), 2):
        bcv += np.sum((centroids[i] - centroids[j]) ** 2)
    return bcv

def minmax_normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def euclidean(p, c):
    return np.sqrt(np.sum((np.array(p) - np.array(c)) ** 2))

def run_kmeans_manual(X_df, initial_centroids, max_iter=20):
    """
    Jalankan K-Means secara manual dan simpan setiap iterasi
    untuk ditampilkan sebagai tabel seperti di Excel.
    """
    centroids = [c.copy() for c in initial_centroids]
    k = len(centroids)
    iteration_data = []

    for it in range(1, max_iter + 1):
        rows = []
        labels = []

        for idx, row in X_df.iterrows():
            p = [row["norm rating"], row["norm reviewers"]]
            dists = [round(euclidean(p, c), 6) for c in centroids]
            min_dist = min(dists)
            min_dist_sq = round(min_dist ** 2, 6)
            cluster = dists.index(min_dist) + 1  # 1-indexed
            labels.append(cluster)

            # Ambil Kode Produk dari kolom jika ada, fallback ke P{idx+1}
            kode_produk = (
                X_df.loc[idx, "Kode Produk"]
                if "Kode Produk" in X_df.columns
                else f"P{idx + 1}"
            )

            rows.append({
                "Kode Produk": kode_produk,
                "Nama Produk": X_df.loc[idx, "Name"] if "Name" in X_df.columns else f"P{idx+1}",
                "Norm Rating": round(p[0], 6),
                "Norm Reviewer": round(p[1], 6),
                "d(C1)": dists[0],
                "d(C2)": dists[1],
                "d(C3)": dists[2],
                "d(C4)": dists[3],
                "JCT": round(min_dist, 6),
                "JCT²": round(min_dist_sq, 6),
                "Cluster": f"C{cluster}",
            })

        iter_df = pd.DataFrame(rows)
        wcv = round(iter_df["JCT²"].sum(), 6)

        # Hitung centroid baru
        new_centroids = []
        cluster_counts = []
        for c in range(1, k + 1):
            members = [
                [X_df.iloc[i]["norm rating"], X_df.iloc[i]["norm reviewers"]]
                for i, lbl in enumerate(labels) if lbl == c
            ]
            cluster_counts.append(len(members))
            if members:
                new_c = [round(np.mean([m[0] for m in members]), 6),
                         round(np.mean([m[1] for m in members]), 6)]
            else:
                new_c = centroids[c - 1]
            new_centroids.append(new_c)

        iteration_data.append({
            "iterasi": it,
            "df": iter_df,
            "centroids_before": [c.copy() for c in centroids],
            "centroids_after": new_centroids,
            "wcv": wcv,
            "cluster_counts": cluster_counts,
            "labels": labels,
        })

        # Cek konvergensi
        converged = all(
            np.allclose(centroids[i], new_centroids[i], atol=1e-8)
            for i in range(k)
        )
        centroids = new_centroids
        if converged:
            break

    return iteration_data


# =====================================
# 3. SIDEBAR NAVIGATION
# =====================================
st.sidebar.title("☀️ SunCluster Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu Utama",
    [
        "🏠 Dashboard",
        "📂 Upload Dataset",
        "📊 Data Sunscreen",
        "⚙️ Normalisasi Data",
        "📈 Elbow Method",
        "🔍 Proses Clustering",
        "🧮 Evaluasi Cluster 4",
        "📉 Visualisasi"
    ]
)

# =====================================
# 4. IMPLEMENTASI MENU
# =====================================

# --- MENU: DASHBOARD ---
if menu == "🏠 Dashboard":
    st.title("☀️ Sunscreen Cluster Dashboard")
    st.markdown("### 🚀 Analisis Overview")

    col1, col2, col3, col4 = st.columns(4)

    if st.session_state.df_raw is not None:
        total_data = len(st.session_state.df_raw)
        avg_rating = round(st.session_state.df_raw["Rating"].mean(), 2)
        max_rev = st.session_state.df_raw["Total Reviewers"].max()
        col1.metric("Total Produk", f"{total_data} Item")
        col2.metric("Rata-rata Rating", avg_rating)
        col3.metric("Review Tertinggi", f"{max_rev}")
    else:
        col1.metric("Total Produk", "0")
        col2.metric("Rata-rata Rating", "0")
        col3.metric("Review Tertinggi", "0")

    if st.session_state.df_clustered is not None:
        col4.metric("Status Cluster", "4 Clusters ✅")
    else:
        col4.metric("Status Cluster", "Belum Proses")

    st.divider()
    left_co, right_co = st.columns([2, 1])
    with left_co:
        st.info("""
        #### 📌 Tentang Sistem
        Sistem ini dirancang untuk mengelompokkan produk sunscreen berdasarkan 
        **Rating** dan **Total Reviewers** menggunakan algoritma K-Means.
        
        Aplikasi ini memvalidasi hasil perhitungan manual dari Excel ke dalam 
        sistem otomasi berbasis Python (Streamlit).
        """)
        if st.session_state.df_raw is not None:
            st.write("#### 📄 Pratinjau Dataset")
            st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)

    with right_co:
        st.warning("""
        #### 🛠️ Alur Sistem:
        1. **Pre-processing**: Normalisasi Min-Max.
        2. **Optimasi**: Mencari K terbaik (Elbow).
        3. **Clustering**: Inisialisasi Centroid Awal.
        4. **Validasi**: Evaluasi WCV & BCV.
        """)

# --- MENU: UPLOAD ---
elif menu == "📂 Upload Dataset":
    st.header("Upload Dataset")
    file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])
    if file is not None:
        try:
            if file.name.endswith("csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            st.session_state.df_raw = df
            st.success("Data berhasil diunggah!")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

# --- MENU: DATA ---
elif menu == "📊 Data Sunscreen":
    st.header("Data Sunscreen Terunggah")
    if st.session_state.df_raw is not None:
        st.dataframe(st.session_state.df_raw)
    else:
        st.warning("Upload data terlebih dahulu di menu 'Upload Dataset'.")

# --- MENU: NORMALISASI ---
elif menu == "⚙️ Normalisasi Data":
    st.header("Normalisasi Data (Min-Max Scaling)")
    if st.session_state.df_raw is None:
        st.warning("Silakan unggah dataset terlebih dahulu.")
        st.stop()

    df = st.session_state.df_raw.copy()
    df["norm rating"] = minmax_normalize(df["Rating"])
    df["norm reviewers"] = minmax_normalize(df["Total Reviewers"])
    st.session_state.df_norm = df

    st.success("Normalisasi berhasil dilakukan.")

    # Tampilkan kolom Kode Produk jika ada
    cols_to_show = ["Rating", "Total Reviewers", "norm rating", "norm reviewers"]
    if "Kode Produk" in df.columns:
        cols_to_show = ["Kode Produk"] + cols_to_show
    if "Name" in df.columns:
        cols_to_show = ["Name"] + cols_to_show if "Kode Produk" not in cols_to_show else ["Kode Produk", "Name"] + ["Rating", "Total Reviewers", "norm rating", "norm reviewers"]

    st.dataframe(df[cols_to_show], use_container_width=True)

# --- MENU: ELBOW METHOD ---
elif menu == "📈 Elbow Method":
    st.header("Penentuan Jumlah Cluster (Elbow Method)")
    if st.session_state.df_norm is None:
        st.warning("Lakukan normalisasi data terlebih dahulu.")
        st.stop()

    X = st.session_state.df_norm[["norm rating", "norm reviewers"]].values
    results = []

    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", random_state=0, n_init=10)
        labels = km.fit_predict(X)
        wcv = hitung_wcv(X, labels, km.cluster_centers_)
        bcv = hitung_bcv(km.cluster_centers_)
        results.append({"K": k, "WCV": wcv, "BCV": bcv, "Rasio": bcv/wcv if wcv != 0 else 0})

    result_df = pd.DataFrame(results)
    st.subheader("Tabel Nilai WCV & BCV")
    st.table(result_df)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Grafik WCV (Elbow)")
        fig1, ax1 = plt.subplots()
        ax1.plot(result_df["K"], result_df["WCV"], marker='o', color='blue')
        ax1.set_xlabel("K")
        ax1.set_ylabel("WCV")
        st.pyplot(fig1)
    with col_p2:
        st.subheader("Grafik Rasio BCV/WCV")
        fig2, ax2 = plt.subplots()
        ax2.plot(result_df["K"], result_df["Rasio"], marker='o', color='green')
        ax2.set_xlabel("K")
        ax2.set_ylabel("Rasio")
        st.pyplot(fig2)

# --- MENU: PROSES CLUSTERING ---
elif menu == "🔍 Proses Clustering":
    st.header("Proses Clustering K=4")
    if st.session_state.df_norm is None:
        st.warning("Lakukan normalisasi data terlebih dahulu.")
        st.stop()

    df = st.session_state.df_norm.copy().reset_index(drop=True)
    X = df[["norm rating", "norm reviewers"]].values

    # Centroid Awal (sesuai file Excel)
    initial_centroids = [
        [0.75, 0.068882],
        [0.607143, 0.030825],
        [0.678571, 0.688897],
        [0.5, 0.008108],
    ]

    # ── Tombol jalankan clustering ──────────────────────────────────────────
    if st.button("▶️ Jalankan Clustering"):
        with st.spinner("Sedang menghitung iterasi..."):
            iter_data = run_kmeans_manual(df, initial_centroids)
            st.session_state.iteration_tables = iter_data

            # Simpan hasil akhir
            final = iter_data[-1]
            df["Cluster"] = [f"C{l}" for l in final["labels"]]
            st.session_state.df_clustered = df

            # Buat objek KMeans final untuk evaluasi & visualisasi
            final_centroids_arr = np.array(final["centroids_after"])
            km_final = KMeans(n_clusters=4, init=final_centroids_arr, n_init=1, random_state=0)
            km_final.fit(X)
            st.session_state.kmeans = km_final

        st.success(f"✅ Clustering selesai dalam {len(iter_data)} iterasi!")

    # ── Tampilkan tabel iterasi ─────────────────────────────────────────────
    if st.session_state.iteration_tables is not None:
        iter_data = st.session_state.iteration_tables
        total_iter = len(iter_data)

        st.markdown("---")
        st.subheader(f"📋 Tabel Iterasi ({total_iter} iterasi hingga konvergen)")

        # Pilih iterasi yang ingin dilihat
        selected_iter = st.selectbox(
            "Pilih Iterasi:",
            options=list(range(1, total_iter + 1)),
            format_func=lambda x: f"Iterasi {x}" + (" ✅ (Final)" if x == total_iter else ""),
        )

        it = iter_data[selected_iter - 1]
        iter_df = it["df"]

        # Info centroid sebelum & sesudah iterasi ini
        c_before = it["centroids_before"]
        c_after  = it["centroids_after"]

        st.markdown(f"#### 📍 Centroid – Iterasi {selected_iter}")
        cent_cols = st.columns(4)
        for ci in range(4):
            with cent_cols[ci]:
                st.markdown(f"**C{ci+1}**")
                st.markdown(f"Sebelum: `({c_before[ci][0]:.6f}, {c_before[ci][1]:.6f})`")
                st.markdown(f"Sesudah: `({c_after[ci][0]:.6f}, {c_after[ci][1]:.6f})`")

        st.markdown("---")

        # Tabel iterasi utama
        st.markdown(f"#### 📊 Tabel Pengelompokan – Iterasi {selected_iter}")

        # Warna baris per cluster
        CLUSTER_COLORS = {
            "C1": "#d4edda",   # hijau muda
            "C2": "#fff3cd",   # kuning muda
            "C3": "#cce5ff",   # biru muda
            "C4": "#f8d7da",   # merah muda
        }

        def highlight_cluster(row):
            color = CLUSTER_COLORS.get(row["Cluster"], "#ffffff")
            return [f"background-color: {color}"] * len(row)

        # Format angka desimal
        format_dict = {
            "Norm Rating":    "{:.6f}",
            "Norm Reviewer":  "{:.6f}",
            "d(C1)":          "{:.6f}",
            "d(C2)":          "{:.6f}",
            "d(C3)":          "{:.6f}",
            "d(C4)":          "{:.6f}",
            "JCT":            "{:.6f}",
            "JCT²":           "{:.6f}",
        }

        styled = (
            iter_df.style
            .apply(highlight_cluster, axis=1)
            .format(format_dict)
        )

        st.dataframe(styled, use_container_width=True, height=500)

        # ── Download per iterasi ────────────────────────────────────────────
        dl_df = iter_df.copy()
        csv_iter = dl_df.to_csv(index=False).encode("utf-8")
        is_final = selected_iter == total_iter
        label_suffix = " (Final ✅)" if is_final else ""
        st.download_button(
            f"📥 Download Tabel Iterasi {selected_iter}{label_suffix} (CSV)",
            data=csv_iter,
            file_name=f"clustering_iterasi_{selected_iter}.csv",
            mime="text/csv",
            key=f"dl_iter_{selected_iter}",
        )

        # ── Ringkasan iterasi ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"#### 📈 Ringkasan – Iterasi {selected_iter}")

        m1, m2, m3, m4, m5 = st.columns(5)
        counts = it["cluster_counts"]
        m1.metric("C1 (produk berkualitas)", counts[0])
        m2.metric("C2 (kurang diminati)",    counts[1])
        m3.metric("C3 (favorit)",            counts[2])
        m4.metric("C4 (tidak populer)",      counts[3])
        m5.metric("WCV Iterasi Ini",         f"{it['wcv']:.5f}")

        # ── Tabel ringkasan semua iterasi ───────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Ringkasan Semua Iterasi")

        summary_rows = []
        for it_data in iter_data:
            c_aft = it_data["centroids_after"]
            summary_rows.append({
                "Iterasi":  it_data["iterasi"],
                "WCV":      round(it_data["wcv"], 5),
                "C1 (n)":  it_data["cluster_counts"][0],
                "C2 (n)":  it_data["cluster_counts"][1],
                "C3 (n)":  it_data["cluster_counts"][2],
                "C4 (n)":  it_data["cluster_counts"][3],
                "Centroid C1": f"({c_aft[0][0]:.4f}, {c_aft[0][1]:.4f})",
                "Centroid C2": f"({c_aft[1][0]:.4f}, {c_aft[1][1]:.4f})",
                "Centroid C3": f"({c_aft[2][0]:.4f}, {c_aft[2][1]:.4f})",
                "Centroid C4": f"({c_aft[3][0]:.4f}, {c_aft[3][1]:.4f})",
                "Status":   "✅ Konvergen" if it_data["iterasi"] == total_iter else "🔄 Lanjut",
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

# --- MENU: EVALUASI ---
elif menu == "🧮 Evaluasi Cluster 4":
    st.header("Hasil Evaluasi Kualitas Cluster")
    if st.session_state.df_clustered is None:
        st.warning("Lakukan proses clustering terlebih dahulu.")
        st.stop()

    km = st.session_state.kmeans
    X = st.session_state.df_clustered[["norm rating", "norm reviewers"]].values

    WCV = hitung_wcv(X, km.labels_, km.cluster_centers_)
    BCV = hitung_bcv(km.cluster_centers_)
    rasio = BCV / WCV if WCV != 0 else 0

    st.markdown("#### Metrik Akhir (Iterasi Final)")
    ev1, ev2, ev3 = st.columns(3)
    ev1.metric("Within-Cluster Variation (WCV)", round(WCV, 5))
    ev2.metric("Between-Cluster Variation (BCV)", round(BCV, 5))
    ev3.metric("Rasio Evaluasi (BCV/WCV)", round(rasio, 5))

# --- MENU: VISUALISASI ---
elif menu == "📉 Visualisasi":
    st.header("Visualisasi Sebaran Produk")
    if st.session_state.df_clustered is None:
        st.warning("Lakukan proses clustering terlebih dahulu.")
        st.stop()

    df  = st.session_state.df_clustered
    km  = st.session_state.kmeans

    # Map cluster label ke integer (C1→0, C2→1, dst.)
    cluster_map = {"C1": 0, "C2": 1, "C3": 2, "C4": 3}
    df["_cidx"] = df["Cluster"].map(cluster_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8']
    labels_names = ["C1 – Berkualitas", "C2 – Kurang Diminati", "C3 – Favorit", "C4 – Tidak Populer"]

    for i in range(4):
        subset = df[df["_cidx"] == i]
        ax.scatter(
            subset["norm rating"],
            subset["norm reviewers"],
            label=labels_names[i],
            c=colors[i],
            alpha=0.6,
        )

    ax.scatter(
        km.cluster_centers_[:, 0],
        km.cluster_centers_[:, 1],
        c="black", marker="X", s=250, label="Centroids",
    )

    ax.set_xlabel("Rating (Normalized)")
    ax.set_ylabel("Total Reviewers (Normalized)")
    ax.set_title("Pemetaan Cluster Sunscreen")
    ax.legend()
    st.pyplot(fig)