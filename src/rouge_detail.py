from rouge_score import rouge_scorer as _rs
from rouge_score.tokenizers import DefaultTokenizer

_tokenizer = DefaultTokenizer(use_stemmer=True)

MAX_EXAMPLES = 40


def _matched_ngrams(target_ngrams, prediction_ngrams):
    """Danh sách (n-gram, số lần trùng) sắp theo số lần trùng giảm dần."""
    matched = []
    for ngram, count in target_ngrams.items():
        overlap = min(count, prediction_ngrams.get(ngram, 0))
        if overlap > 0:
            matched.append((" ".join(ngram), overlap))
    matched.sort(key=lambda item: (-item[1], item[0]))
    return matched


def _ngram_detail(target_tokens, prediction_tokens, n):
    target_ngrams = _rs._create_ngrams(target_tokens, n)
    prediction_ngrams = _rs._create_ngrams(prediction_tokens, n)
    score = _rs._score_ngrams(target_ngrams, prediction_ngrams)

    matched = _matched_ngrams(target_ngrams, prediction_ngrams)
    match_count = sum(count for _, count in matched)

    return {
        "precision": round(score.precision, 4),
        "recall": round(score.recall, 4),
        "fmeasure": round(score.fmeasure, 4),
        "match_count": match_count,
        "candidate_count": sum(prediction_ngrams.values()),
        "reference_count": sum(target_ngrams.values()),
        "matched_examples": [w for w, _ in matched[:MAX_EXAMPLES]],
        "matched_total_unique": len(matched),
    }


def _lcs_detail(target_tokens, prediction_tokens):
    if not target_tokens or not prediction_tokens:
        return {
            "precision": 0, "recall": 0, "fmeasure": 0,
            "match_count": 0,
            "candidate_count": len(prediction_tokens),
            "reference_count": len(target_tokens),
            "matched_examples": [],
            "matched_total_unique": 0,
        }

    lcs_table = _rs._lcs_table(target_tokens, prediction_tokens)
    lcs_length = lcs_table[-1][-1]
    lcs_indices = _rs._backtrack_norec(lcs_table, target_tokens, prediction_tokens)
    lcs_words = [target_tokens[i] for i in lcs_indices]

    score = _rs._score_lcs(target_tokens, prediction_tokens)

    return {
        "precision": round(score.precision, 4),
        "recall": round(score.recall, 4),
        "fmeasure": round(score.fmeasure, 4),
        "match_count": lcs_length,
        "candidate_count": len(prediction_tokens),
        "reference_count": len(target_tokens),
        "matched_examples": lcs_words[:MAX_EXAMPLES],
        "matched_total_unique": len(lcs_words),
    }


def rouge_breakdown(reference_text, summary_sentences):
    """Trả về chi tiết rouge1 / rouge2 / rougeL để giải thích cho người dùng."""
    prediction_text = " ".join(summary_sentences)

    target_tokens = _tokenizer.tokenize(reference_text)
    prediction_tokens = _tokenizer.tokenize(prediction_text)

    return {
        "rouge1": _ngram_detail(target_tokens, prediction_tokens, 1),
        "rouge2": _ngram_detail(target_tokens, prediction_tokens, 2),
        "rougeL": _lcs_detail(target_tokens, prediction_tokens),
    }
