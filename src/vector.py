import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf_matrix(sentences):
    #Biến danh sách câu thành ma trận TF-IDF (unigram + bigram, loại stop-word)
    vectorizer = TfidfVectorizer(
        stop_words="english",       # Đặc trưng 1: Stop-word Removal (loại từ dừng tiếng Anh)
        ngram_range=(1, 2),         # Đặc trưng 2: TF-IDF Unigram + Đặc trưng 3: TF-IDF Bigram
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer


def calculate_similarity_matrix(tfidf_matrix):
    #Tính ma trận Cosine Similarity giữa tất cả cặp câu, xóa đường chéo."""
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix


def extract_additional_features(sentences):
    # Trích xuất 4 đặc trưng bổ sung cho mỗi câu (giá trị 0-1)."""
    num_sentences = len(sentences)
    features = np.zeros((num_sentences, 4))
    max_len = max(len(s.split()) for s in sentences) if sentences else 1

    for i, s in enumerate(sentences):
        words = s.split()
        sentence_len = len(words)

        # Đặc trưng 4: Sentence Length — độ dài câu (chuẩn hóa theo câu dài nhất)
        len_score = sentence_len / max_len

        # Đặc trưng 5: Sentence Position — vị trí câu (câu đầu = 1.0, càng cuối càng thấp)
        position_score = 1.0 - (i / num_sentences)

        # Đặc trưng 6: Numeric Content — có chứa số liệu hay không
        has_number = 1.0 if re.search(r"\d+", s) else 0.0

        # Đặc trưng 7: Proper Nouns / Capitalization — tỷ lệ từ viết hoa / tên riêng
        capital_count = sum(1 for w in words if w.isupper() or w.istitle())
        capital_ratio = capital_count / sentence_len if sentence_len > 0 else 0

        features[i] = [len_score, position_score, has_number, capital_ratio]

    return features
