from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(sentences):
    """
    Chuyển danh sách câu thành ma trận TF-IDF.
    """
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(sentences)

    return tfidf_matrix, vectorizer