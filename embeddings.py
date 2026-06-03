"""Ubah teks jadi embedding (IndoBERT) lalu bangun index FAISS untuk pencarian."""
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = "firqaaa/indo-sentence-bert-base"

def load_model():
    print(f"Memuat model embedding: {MODEL_NAME} (unduh otomatis saat pertama) ...")
    return SentenceTransformer(MODEL_NAME)

def encode(model, texts):
    """Ubah daftar teks jadi matriks vektor ternormalisasi (cosine-ready)."""
    emb = model.encode(
        list(texts),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # panjang vektor = 1 -> inner product = cosine
    )
    return emb.astype("float32")

def build_faiss_index(embeddings):
    """Bangun index FAISS berbasis cosine similarity."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # IP = inner product
    index.add(embeddings)
    return index

# Uji cepat
if __name__ == "__main__":
    model = load_model()
    contoh = [
        "Pelayanannya ramah dan makanannya enak sekali",
        "Kualitas produknya buruk, saya kecewa",
        "Tempatnya bagus, suasananya nyaman",
    ]
    emb = encode(model, contoh)
    print("\nBentuk matriks embedding:", emb.shape, "(jumlah kalimat x dimensi)")

    index = build_faiss_index(emb)
    print("Jumlah vektor dalam index FAISS:", index.ntotal)

    # Cari kalimat paling mirip dengan query
    query = encode(model, ["restoran ini menyenangkan, recommended"])
    skor, posisi = index.search(query, k=3)
    print("\nQuery: 'restoran ini menyenangkan, recommended'")
    print("Kemiripan ke-3 kalimat contoh (makin tinggi makin mirip):")
    for s, p in zip(skor[0], posisi[0]):
        print(f"  skor {s:.3f}  <-  {contoh[p]}")