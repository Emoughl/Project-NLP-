import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_matrix(sentences):
    #Biến danh sách các câu thành ma trận TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english", # 1.Stop-word Removal (đặc trưng lọc từ dừng)
        ngram_range=(1, 2), # 2. TF-IDF Unigram (đặc trưng từ đơn), 3.TF-IDF Bigram (đặc trưng cụm 2 từ)
        min_df=1
    )
    
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer

def calculate_similarity_matrix(tfidf_matrix):
    #Tính ma trận độ tương đồng Cosine giữa tất cả các cặp câu
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    #Xóa đường chéo chính (độ tương đồng của 1 câu với chính nó = 1.0)
    np.fill_diagonal(sim_matrix, 0)
    
    return sim_matrix

def extract_additional_features(sentences):
    num_sentences = len(sentences)
    features = np.zeros((num_sentences, 4))
    
    # Tìm độ dài câu dài nhất để chuẩn hóa
    max_len = max([len(s.split()) for s in sentences]) if sentences else 1
    
    for i, s in enumerate(sentences):
        words = s.split()
        sentence_len = len(words)
        
        # 4.Sentence Length (đặc trưng độ dài câu)
        len_score = sentence_len / max_len
        
        # 5.Sentence Position (đặc trưng vị trí câu)
        position_score = 1.0 - (i / num_sentences)
        
        # 6.Numeric Content (đặc trưng chứa con số/dữ liệu)
        has_number = 1.0 if re.search(r"\d+", s) else 0.0
        
       # 7.Proper Nouns / Capitalization (Tên riêng, địa danh, chữ viết hoa)
        capital_count = sum(1 for w in words if w.isupper() or w.istitle())
        capital_ratio = capital_count / max_len if max_len > 0 else 0
        
        features[i] = [len_score, position_score, has_number, capital_ratio]
        
    return features