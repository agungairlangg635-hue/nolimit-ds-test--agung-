"""Bonus: aplikasi Streamlit untuk demo klasifikasi sentimen + pencarian FAISS.
Jalankan:  streamlit run app.py
"""
import streamlit as st

from preprocess import load_clean, clean_text
from embeddings import load_model
from models import EmbeddingKNNClassifier, EmbeddingLogRegClassifier

st.set_page_config(page_title="Sentimen Bahasa Indonesia", page_icon="💬")

EMOJI = {"positive": "😊", "neutral": "😐", "negative": "😠"}


# --- Muat data, model, & latih classifier SEKALI lalu cache --------------
@st.cache_resource
def setup():
    train = load_clean("data/nusax_train.csv")
    model = load_model()
    knn = EmbeddingKNNClassifier(model, k=5).fit(
        train["text"].tolist(), train["label"].tolist())
    logreg = EmbeddingLogRegClassifier(model).fit(
        train["text"].tolist(), train["label"].tolist())
    return train, knn, logreg


st.title("💬 Analisis Sentimen Bahasa Indonesia")
st.caption("IndoBERT embeddings + FAISS + scikit-learn — NoLimit DS Test")

with st.spinner("Memuat model (sekali saja, mohon tunggu)..."):
    train, knn, logreg = setup()

metode = st.radio("Metode:", ["Logistic Regression", "FAISS k-NN"], horizontal=True)
teks = st.text_area("Masukkan teks:",
                    "Pelayanannya ramah dan makanannya enak sekali!")

col1, col2 = st.columns(2)

with col1:
    if st.button("Prediksi", type="primary") and teks.strip():
        bersih = clean_text(teks)
        clf = logreg if metode == "Logistic Regression" else knn
        pred = clf.predict([bersih])[0]

        st.subheader(f"Hasil: {EMOJI.get(pred, '')} {pred.upper()}")

        # Pamerkan FAISS: 5 kalimat latih paling mirip + teksnya
        st.markdown("##### 🔍 5 kalimat latih paling mirip (via FAISS):")
        for skor, label, posisi in knn.neighbours(bersih, k=5):
            st.write(f"{EMOJI.get(label, '')} **{label}** · kemiripan {skor:.3f}")
            st.caption(train["text"].iloc[posisi][:120])

with col2:
    if st.button("Contoh acak dari data"):
        sample = train.sample(1).iloc[0]
        st.info(f"**{sample['label']}**\n\n{sample['text']}")