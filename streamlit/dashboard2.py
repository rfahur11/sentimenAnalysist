import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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

# 🟢 Fungsi untuk Clustering
def cluster_comments(comments, n_clusters=3):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(comments)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    return clusters, vectorizer, kmeans

# 🟢 Fungsi untuk Membuat Word Cloud dari Cluster
def generate_wordcloud(comments):
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(" ".join(comments))
    return wordcloud

# 🟢 Streamlit UI
st.title("📊 Dashboard Clustering Komentar Berdasarkan Sentimen")

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
        selected_sentiment = st.selectbox("Pilih Sentimen untuk Clustering", sentiment_options)

        # Filter komentar berdasarkan sentimen yang dipilih
        filtered_df = df[df["predicted_label"] == selected_sentiment]

        if not filtered_df.empty:
            # Lakukan Clustering
            n_clusters = st.slider("Pilih jumlah cluster", 2, 5, 3)
            clusters, vectorizer, kmeans = cluster_comments(filtered_df["cleaned_comment"], n_clusters=n_clusters)
            filtered_df["cluster"] = clusters

            # Tampilkan hasil clustering dalam tabel
            st.subheader(f"📋 Tabel Komentar untuk Sentimen '{selected_sentiment}'")
            st.dataframe(filtered_df)

            # Word Cloud untuk setiap cluster
            for i in range(n_clusters):
                cluster_comments = filtered_df[filtered_df["cluster"] == i]["cleaned_comment"]
                if not cluster_comments.empty:
                    st.subheader(f"🔠 Word Cloud untuk Cluster {i}")
                    wordcloud = generate_wordcloud(cluster_comments)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
        else:
            st.warning("Tidak ada data untuk kategori ini.")

else:
    st.warning("Silakan upload file CSV untuk melakukan clustering.")
