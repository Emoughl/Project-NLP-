from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def sentence_similarity(sentence1, sentence2):
    # Tính độ tương đồng giữa hai câu bằng TF-IDF + Cosine Similarity.

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform([sentence1, sentence2])

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])

    return similarity[0][0]