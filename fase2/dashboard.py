import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 🟢 1. Load Dataset
st.title("📊 Dashboard Analisis Sentimen Komentar")

uploaded_file = st.file_uploader("Upload file CSV hasil prediksi sentimen", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pastikan dataset memiliki kolom yang sesuai
    if "comment" not in df.columns or "predicted_label" not in df.columns:
        st.error("Dataset harus memiliki kolom 'comment' dan 'predicted_label'")
    else:
        # 🟢 2. Hitung Distribusi Sentimen
        sentiment_counts = df["predicted_label"].value_counts()

        # 🟢 3. Pie Chart Sentimen
        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", colors=["#FF6B6B", "#FFD93D", "#6BCB77"])
        ax1.set_title("Distribusi Sentimen (%)")
        st.pyplot(fig1)

        # 🟢 4. Bar Chart Sentimen
        fig2, ax2 = plt.subplots()
        ax2.bar(sentiment_counts.index, sentiment_counts.values, color=["#FF6B6B", "#FFD93D", "#6BCB77"])
        ax2.set_title("Jumlah Komentar per Sentimen")
        ax2.set_ylabel("Jumlah Komentar")
        st.pyplot(fig2)

        # 🟢 5. Tampilkan Data
        st.subheader("📋 Tabel Komentar dan Sentimen")
        st.dataframe(df)

else:
    st.warning("Silakan upload file CSV untuk melihat analisis sentimen.")

