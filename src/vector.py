from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_tfidf(sentences):
    vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    norm="l2"
    )
    tfidf = vectorizer.fit_transform(sentences)
    return tfidf, vectorizer


def build_similarity_matrix(tfidf):

    similarity = cosine_similarity(tfidf)

    np.fill_diagonal(similarity, 0)

    return similarity