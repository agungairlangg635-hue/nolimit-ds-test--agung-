"""Tiga classifier sentimen, semuanya berbasis komponen Hugging Face.

1. EmbeddingKNNClassifier    -> embedding + FAISS k-NN (tanpa latih; index = model)
2. EmbeddingLogRegClassifier -> embedding + Logistic Regression (sklearn)
3. TransformerClassifier     -> model sentimen Indonesia pretrained (HF pipeline)
"""
from collections import Counter
from embeddings import encode, build_faiss_index

# Pemetaan label model transformer ke bentuk kanonik kita
CANON = {
    "positive": "positive", "positif": "positive", "pos": "positive",
    "negative": "negative", "negatif": "negative", "neg": "negative",
    "neutral": "neutral", "netral": "neutral",
}
LABEL_ORDER = ["positive", "neutral", "negative"]  # untuk model yg keluarannya LABEL_0/1/2

def _canon(raw):
    k = str(raw).strip().lower()
    if k in CANON:
        return CANON[k]
    if k.startswith("label_"):
        try:
            return LABEL_ORDER[int(k.split("_")[1])]
        except (ValueError, IndexError):
            pass
    return k


class EmbeddingKNNClassifier:
    """Klasifikasi via voting k tetangga terdekat di index FAISS."""
    def __init__(self, model, k=5):
        self.model = model
        self.k = k
        self.index = None
        self.train_labels = []

    def fit(self, texts, labels):
        emb = encode(self.model, texts)
        self.index = build_faiss_index(emb)
        self.train_labels = list(labels)
        self.k = min(self.k, len(self.train_labels))
        return self

    def predict(self, texts):
        emb = encode(self.model, texts)
        _, idx = self.index.search(emb, self.k)
        preds = []
        for row in idx:
            votes = [self.train_labels[i] for i in row if i != -1]
            preds.append(Counter(votes).most_common(1)[0][0])
        return preds

    def neighbours(self, text, k=None):
        """[(skor, label, posisi)] tetangga terdekat — untuk demo/penjelasan."""
        k = k or self.k
        emb = encode(self.model, [text])
        scores, idx = self.index.search(emb, k)
        return [(float(s), self.train_labels[int(i)], int(i))
                for s, i in zip(scores[0], idx[0]) if i != -1]


class EmbeddingLogRegClassifier:
    """Logistic Regression dilatih di atas embedding kalimat."""
    def __init__(self, model):
        from sklearn.linear_model import LogisticRegression
        self.model = model
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, texts, labels):
        self.clf.fit(encode(self.model, texts), labels)
        return self

    def predict(self, texts):
        return list(self.clf.predict(encode(self.model, texts)))


class TransformerClassifier:
    """Model sentimen Indonesia pretrained lewat pipeline Transformers."""
    def __init__(self, model_name="w11wo/indonesian-roberta-base-sentiment-classifier"):
        from transformers import pipeline
        self.model_name = model_name
        self.pipe = pipeline("text-classification", model=model_name, truncation=True)

    def fit(self, texts=None, labels=None):
        return self  # pretrained — tak perlu dilatih

    def predict(self, texts):
        return [_canon(o["label"]) for o in self.pipe(list(texts))]