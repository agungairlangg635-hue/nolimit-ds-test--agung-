# NoLimit DS Test — Analisis Sentimen Bahasa Indonesia

Solusi untuk **Task A (Classification)**: klasifikasi sentimen teks berbahasa
Indonesia ke dalam tiga kelas — **positive / neutral / negative** — menggunakan
model Hugging Face dan embeddings dengan pencarian FAISS.

## Ringkasan Pendekatan

Tiga metode dibangun dan dibandingkan, semuanya berbasis komponen Hugging Face:

1. **Embedding + FAISS k-NN** — kalimat di-embed dengan IndoBERT, lalu diklasifikasi
   lewat voting *k* tetangga terdekat pada index FAISS (cosine similarity).
2. **Embedding + Logistic Regression** — vektor IndoBERT yang sama, dengan classifier
   Logistic Regression (scikit-learn) di atasnya.
3. **Transformer** — model sentimen Indonesia pretrained via Transformers `pipeline`.

## Hasil (pada 400 data uji)

| Metode | Accuracy | Macro-F1 |
|---|---|---|
| Embedding + FAISS k-NN | 0.788 | 0.760 |
| Embedding + Logistic Regression | 0.875 | 0.868 |
| **Transformer (pretrained)** | **0.908** | **0.894** |

Tren ini sesuai ekspektasi: classifier yang mempelajari batas antar-kelas (LogReg)
mengungguli voting tetangga (k-NN), dan model transformer yang fine-tuned untuk
sentimen memberi hasil terbaik.

**Catatan analisis:** kelas `neutral` konsisten menjadi yang paling sulit di ketiga
metode (mis. recall k-NN hanya 0.50). Ini sejalan dengan distribusi data yang timpang
(neutral ±24% vs negative/positive ±38% masing-masing). Karena itu **macro-F1**
digunakan sebagai metrik utama, sebab accuracy dapat menyembunyikan kelemahan pada
kelas minoritas.

## Dataset

- **Sumber:** NusaX-senti, subset Indonesia (`ind`), via Hugging Face
  [`mteb/NusaX-senti`](https://huggingface.co/datasets/mteb/NusaX-senti).
- **Asal:** diturunkan dari dataset SmSA (IndoNLU); diterjemahkan/dikurasi manusia.
- **Ukuran:** 500 train / 100 validation / 400 test (label: positive/neutral/negative).
- **Lisensi:** CC-BY-SA 4.0.
- **Sample lokal:** `data/sample_sentiment.csv` (60 baris) disertakan untuk verifikasi cepat.

Catatan label: pada sumber, label berupa angka `0/1/2` yang dipetakan menjadi
`negative/neutral/positive` (lihat `get_data.py`).

## Model yang Digunakan

- **Embedding:** `firqaaa/indo-sentence-bert-base` (sentence-transformers, IndoBERT).
- **Transformer classifier:** `w11wo/indonesian-roberta-base-sentiment-classifier`.
- **Index pencarian:** FAISS `IndexFlatIP` (inner product = cosine, vektor dinormalisasi).

## Pipeline

Lihat `flowchart.png` / `flowchart.pdf`.