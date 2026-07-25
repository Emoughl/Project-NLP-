from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf(sentences):
    #Chuyển danh sách câu thành ma trận TF-IDF'
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(sentences)

    return tfidf_matrix, vectorizer

#Mục 4 :Biểu diễn văn bản thành đồ thị'
def build_similarity_matrix(tfidf_matrix):
    #Tính ma trận độ tương đồng giữa các câu'
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix