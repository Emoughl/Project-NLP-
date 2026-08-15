"""
Flask API nhỏ bọc lại pipeline TextRank hiện có (preprocess/vector/textrank)
để phục vụ giao diện web (static/index.html).

Chạy:
    python api.py
    truy cập http://127.0.0.1:5000
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from flask import Flask, request, jsonify, send_from_directory

try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

from vector import (
    build_tfidf,
    build_similarity_matrix,
    sentence_length_scores,
    numeric_content_scores,
)
from textrank import rank_sentences, generate_summary
from web_text import split_sentences_generic, clean_reference_text

try:
    from rouge_score import rouge_scorer
    from rouge_detail import rouge_breakdown
    _rouge_scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    _HAS_ROUGE = True
except ImportError:
    _rouge_scorer = None
    rouge_breakdown = None
    _HAS_ROUGE = False

app = Flask(__name__, static_folder="static", static_url_path="")
if _HAS_CORS:
    CORS(app)

MAX_CHARS = 50_000


def _compute_rouge(reference_text, summary_sentences):
    """Tính ROUGE-1 / ROUGE-2 / ROUGE-L giữa tóm tắt hệ thống và bản tham chiếu
    do người dùng cung cấp — cùng cách tính với src/evaluate.py."""
    if not _HAS_ROUGE:
        return None, "Thiếu thư viện rouge_score (pip install rouge_score)."

    ref_clean = clean_reference_text(reference_text)
    if not ref_clean:
        return None, None

    system_summary = " ".join(summary_sentences)
    scores = _rouge_scorer.score(ref_clean, system_summary)

    result = {
        key: {
            "precision": round(val.precision, 4),
            "recall": round(val.recall, 4),
            "fmeasure": round(val.fmeasure, 4),
        }
        for key, val in scores.items()
    }
    return result, None


def _compute_rouge_detail(reference_text, summary_sentences):
    """Số liệu chi tiết (số n-gram/từ trùng khớp, danh sách ví dụ...) phục vụ
    phần 'Xem chi tiết' trên giao diện."""
    if not _HAS_ROUGE or rouge_breakdown is None:
        return None

    ref_clean = clean_reference_text(reference_text)
    if not ref_clean:
        return None

    try:
        return rouge_breakdown(ref_clean, summary_sentences)
    except Exception:
        return None


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    ratio = data.get("ratio")
    top_n = data.get("top_n")
    reference_text = (data.get("reference") or "").strip()

    if not text:
        return jsonify({"error": "Vui lòng nhập văn bản cần tóm tắt."}), 400

    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Văn bản quá dài (tối đa {MAX_CHARS:,} ký tự)."}), 400

    sentences = split_sentences_generic(text)

    if len(sentences) < 2:
        return jsonify({"error": "Văn bản cần có ít nhất 2 câu để có thể tóm tắt."}), 400

    if not top_n:
        ratio = float(ratio) if ratio else 0.3
        ratio = min(max(ratio, 0.05), 1.0)
        top_n = max(1, round(len(sentences) * ratio))

    top_n = min(int(top_n), len(sentences))

    try:
        tfidf_matrix, _ = build_tfidf(sentences)
        similarity_matrix, raw_similarity_matrix = build_similarity_matrix(
            tfidf_matrix, return_raw=True
        )
        scores = rank_sentences(similarity_matrix)
    except ValueError:
        # Xảy ra khi TF-IDF (min_df=2) không tìm được từ nào lặp lại
        # giữa các câu -> văn bản quá ngắn / các câu quá khác biệt nhau.
        return jsonify({
            "error": "Không đủ từ lặp lại giữa các câu để tính TF-IDF. "
                     "Hãy thử dán một đoạn văn bản dài hơn."
        }), 400

    len_scores = sentence_length_scores(sentences)
    num_scores = numeric_content_scores(sentences)

    combined_scores = {
        idx: (0.8 * sc + 0.1 * len_scores[idx] + 0.1 * num_scores[idx])
        for idx, sc in scores.items()
    }

    summary_sentences = generate_summary(
        sentences, combined_scores, raw_similarity_matrix, top_n=top_n
    )
    summary_order = {s: i for i, s in enumerate(summary_sentences)}

    detail = [
        {
            "index": i,
            "text": s,
            "score": round(float(combined_scores[i]), 4),
            "in_summary": s in summary_order,
        }
        for i, s in enumerate(sentences)
    ]

    rouge_result, rouge_warning, rouge_detail = (None, None, None)
    if reference_text:
        rouge_result, rouge_warning = _compute_rouge(reference_text, summary_sentences)
        if rouge_result:
            rouge_detail = _compute_rouge_detail(reference_text, summary_sentences)

    return jsonify({
        "sentence_count": len(sentences),
        "summary_count": len(summary_sentences),
        "compression": round(len(summary_sentences) / len(sentences), 3),
        "summary": summary_sentences,
        "sentences": detail,
        "rouge": rouge_result,
        "rouge_warning": rouge_warning,
        "rouge_detail": rouge_detail,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
