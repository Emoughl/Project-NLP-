import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def sentence_similarity(sentence1, sentence2):
    #Calculate the similarity between two sentences using TF-IDF and Cosine Similarity.
    #stop_words="english": loai cac tu noi (the, a, of, was, said...) de
    #tranh diem bi "bom" ao khi 2 van ban dai deu dung nhieu tu noi giong
    #nhau du noi dung hoan toan khac nhau.

    vectorizer = TfidfVectorizer(
            stop_words="english", # 1.Stop-word Removal (đặc trưng lọc từ dừng)
            ngram_range=(1, 2), # 2. TF-IDF Unigram (đặc trưng từ đơn), 3.TF-IDF Bigram (đặc trưng cụm 2 từ)
            min_df=1
        )

    tfidf = vectorizer.fit_transform([sentence1, sentence2])

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])

    return similarity[0][0]


_STOPWORDS_VI_EN = {
    # Danh sách rất nhỏ, chỉ để lọc bớt từ nối/hư từ khi liệt kê từ khoá
    # trùng nhau — không phải bộ stopword đầy đủ.
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
    """Từ khoá xuất hiện ở CẢ bản tham chiếu và bản tóm tắt hệ thống —
    dùng để minh hoạ trực quan "vì sao" 2 bản được coi là giống nhau khi
    tính Cosine Similarity (thay cho phần liệt kê n-gram trùng khớp mà
    ROUGE từng cung cấp)."""
    ref_set = set(_keywords(reference_text))
    cand_set = set(_keywords(candidate_text))
    shared = sorted(ref_set & cand_set)

    return {
        "shared_words": shared,
        "shared_total": len(shared),
        "reference_word_count": len(ref_set),
        "candidate_word_count": len(cand_set),
    }
