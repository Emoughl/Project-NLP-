from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_tfidf(sentences):
    # Đặc trưng 1: TF-IDF
    vectorizer = TfidfVectorizer(
        # Đặc trưng 2: Loại bỏ stop words    
        stop_words="english",
        lowercase=True,
        # Đặc trưng 3: Sử dụng unigram và bigram
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        norm="l2"
    )
    tfidf = vectorizer.fit_transform(sentences)
    return tfidf, vectorizer


def build_similarity_matrix(tfidf):
    # Xây dựng ma trận độ tương đồng giữa các câu bằng Cosine Similarity
    similarity = cosine_similarity(tfidf)

    # Tiêu chí 8:
    # Cải tiến - Bổ sung trọng số vị trí vào đồ thị
    # Các câu ở đầu văn bản được ưu tiên hơn

    n = similarity.shape[0]

    for i in range(n):
        for j in range(n):

            position_weight = (
                (1 - i / n) +
                (1 - j / n)
            ) / 2

            similarity[i][j] *= position_weight
            
    np.fill_diagonal(similarity, 0)

    return similarity