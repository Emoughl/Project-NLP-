import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def sentence_similarity(sentence1, sentence2):
    #Tính Cosine Similarity giữa 2 câu/đoạn văn qua vector TF-IDF
    #Cùng config TfidfVectorizer với vector.py (stop_words, ngram 1-2)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    tfidf = vectorizer.fit_transform([sentence1, sentence2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]


_STOPWORDS_VI_EN = {
    # Bộ lọc nhỏ cho từ nối/hư từ khi liệt kê từ khóa trùng nhau
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are",
    "was", "were", "for", "with", "by", "at", "this", "that", "it", "its",
    "be", "as", "from", "has", "have", "had", "will", "would", "can",
    "là", "và", "của", "các", "những", "một", "cho", "được", "có", "trong",
    "này", "đó", "khi", "với", "để", "từ", "như", "sẽ", "đã", "bị", "bởi",
    "nên", "thì", "mà", "cũng", "rất", "còn", "vì", "nếu", "hay",
}


def _keywords(text):
    words = re.findall(r"[\wÀ-ỹ]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS_VI_EN]


def keyword_overlap(reference_text, candidate_text):
    """Liệt kê từ khóa xuất hiện ở cả bản tham chiếu và bản tóm tắt."""
    ref_set = set(_keywords(reference_text))
    cand_set = set(_keywords(candidate_text))
    shared = sorted(ref_set & cand_set)

    return {
        "shared_words": shared,
        "shared_total": len(shared),
        "reference_word_count": len(ref_set),
        "candidate_word_count": len(cand_set),
    }
