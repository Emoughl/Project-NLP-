from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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


def build_similarity_matrix(tfidf , edge_threshold=0.02):
    #Build similarity matrix between sentences using Cosine Similarity
    similarity = cosine_similarity(tfidf)

    # Feature 8:
    # Improvement - Add position weights to the graph
    # Sentences at the beginning of the text are given higher priority
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

    return similarity