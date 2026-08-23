import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_matrix(sentences):
    vectorizer = TfidfVectorizer(
        stop_words="english",       
        ngram_range=(1, 2),    
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer


def calculate_similarity_matrix(tfidf_matrix):
    #Tính ma trận Cosine Similarity giữa tất cả cặp câu, xóa đường chéo."""
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


