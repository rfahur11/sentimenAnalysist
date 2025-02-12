import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords jika belum ada
nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))

# 🟢 Fungsi untuk Membersihkan Teks
def clean_text(text):
    text = str(text).lower()  # Konversi ke huruf kecil
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = " ".join([word for word in text.split() if word not in stop_words])  # Hapus stopwords
    return text

# 🟢 Fungsi untuk Membuat Word Cloud
def generate_wordcloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(" ".join(text_data))
    return wordcloud

# 🟢 Streamlit UI
st.title("📊 Dashboard Analisis Sentimen dengan Word Cloud")

# Upload dataset CSV
uploaded_file = st.file_uploader("Upload file CSV hasil prediksi sentimen", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pastikan dataset memiliki kolom yang sesuai
    if "comment" not in df.columns or "predicted_label" not in df.columns:
        st.error("Dataset harus memiliki kolom 'comment' dan 'predicted_label'")
    else:
        # Bersihkan teks
        df["cleaned_comment"] = df["comment"].apply(clean_text)

        # Pilihan sentimen
        sentiment_options = df["predicted_label"].unique().tolist()
        selected_sentiment = st.selectbox("Pilih Sentimen untuk Word Cloud", sentiment_options)

        # Filter komentar berdasarkan sentimen yang dipilih
        filtered_comments = df[df["predicted_label"] == selected_sentiment]["cleaned_comment"]

        # Tampilkan Word Cloud
        if not filtered_comments.empty:
            st.subheader(f"🔠 Word Cloud untuk Sentimen: {selected_sentiment}")
            wordcloud = generate_wordcloud(filtered_comments)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("Tidak ada data untuk kategori ini.")

else:
    st.warning("Silakan upload file CSV untuk melihat analisis sentimen.")

