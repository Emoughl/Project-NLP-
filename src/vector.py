from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


#Feature 4: Sentence length
def sentence_length_scores(sentences, ideal_length=20):
    scores = []
    for s in sentences:
        word_count = len(s.split())
        score = 1.0 - min(1.0, abs(word_count - ideal_length) / ideal_length)
        scores.append(score)
    return np.array(scores)


#Feature 5: Contains numeric data
def numeric_content_scores(sentences):
    scores = []
    for s in sentences:
        has_number = bool(re.search(r"\d", s))
        scores.append(1.0 if has_number else 0.0)
    return np.array(scores)


def build_tfidf(sentences):
    #Feature 1: TF-IDF (Term Frequency–Inverse Document Frequency)
    vectorizer = TfidfVectorizer(
        #Feature 2: Remove stop words
        stop_words="english",
        lowercase=True, #Convert all words to lowercase
        #Feature 3: Use unigram and bigram
        ngram_range=(1, 2), #Get 1 word or phrase
        min_df=2, #If word appears in less than 2 sentences, it will be removed
        max_df=0.9, #If word appears in more than 90% of sentences, it will be removed
        norm="l2"# Normalize the vector to have unit norm
    )
    tfidf = vectorizer.fit_transform(sentences) #
    return tfidf, vectorizer


def build_similarity_matrix(tfidf, edge_threshold=0.02, return_raw=False):
    #Build similarity matrix between sentences using Cosine Similarity
    raw_similarity = cosine_similarity(tfidf)

    # Feature 8:
    # Improvement - Add position weights to the graph
    # Sentences at the beginning of the text are given higher priority
    similarity = raw_similarity.copy()
    n = similarity.shape[0]

    for i in range(n):
        for j in range(n):

            position_weight = (
                (1 - i / n) +
                (1 - j / n)
            ) / 2

            similarity[i][j] *= position_weight
            
    np.fill_diagonal(similarity, 0)

    similarity[similarity < edge_threshold] = 0

    #raw_similarity: used later for duplicate check (no position weighting)
    if return_raw:
        return similarity, raw_similarity

    return similarity