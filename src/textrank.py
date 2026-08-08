import networkx as nx

def rank_sentences(similarity_matrix):
    #Feature 5: Represent text as a graph
    graph = nx.from_numpy_array(similarity_matrix)

    #Rank the importance of sentences using PageRank
    scores = nx.pagerank(
        graph,
        alpha=0.85,
        weight="weight",
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