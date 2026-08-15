import numpy as np


def _pagerank(similarity_matrix, alpha=0.85, max_iter=500, tol=1.0e-6):

    W = np.array(similarity_matrix, dtype=float)
    n = W.shape[0]

    if n == 0:
        return {}

    #Make sure there are no self-loops (a sentence linking to itself)
    np.fill_diagonal(W, 0)

    row_sums = W.sum(axis=1)

    #Build the transition matrix: M[i][j] = probability of moving from i to j
    M = np.zeros_like(W)
    dangling = row_sums == 0

    for i in range(n):
        if dangling[i]:
            #Dangling node (no outgoing edges): distribute uniformly
            M[i] = 1.0 / n
        else:
            M[i] = W[i] / row_sums[i]

    scores = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        new_scores = alpha * (scores @ M) + (1 - alpha) * teleport

        #Convergence check (L1 norm), same stopping rule used by networkx
        if np.abs(new_scores - scores).sum() < n * tol:
            scores = new_scores
            break

        scores = new_scores

    return {i: float(scores[i]) for i in range(n)}


def rank_sentences(similarity_matrix):
    #Feature 5: Represent text as a graph and rank sentences with PageRank
    scores = _pagerank(
        similarity_matrix,
        alpha=0.85,
        max_iter=500
    )

    return scores

#Choose the top sentences to create a summary
def generate_summary(sentences, scores, similarity_matrix, top_n=18):

    if not sentences:
        return []

    top_n = min(top_n, len(sentences))

    #Feature 4: Penalize very short sentences
    adjusted_scores = {}
    for idx, sc in scores.items():
        word_count = len(sentences[idx].split())
        length_factor = min(1.0, word_count / 6)
        adjusted_scores[idx] = sc * length_factor

    ranked = sorted(
        adjusted_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    summary = []

    for index , _ in ranked:

        sentence = sentences[index].strip()

        duplicate = False
        for selected_index, _ in summary:
            sim = similarity_matrix[index][selected_index]
            if sim > 0.9:
                duplicate = True
                break

        if duplicate:
            continue

        summary.append((index, sentence))

        if len(summary) == top_n:
            break

    #Sort by original position so the summary reads in order
    summary.sort(key=lambda item: item[0])

    return [sentence for _, sentence in summary]
