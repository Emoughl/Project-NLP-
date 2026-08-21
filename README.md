# 📝 Natural Language Processing Project

## 📌 Project Information

| Item | Description |
|------|-------------|
| **Course** | Natural Language Processing |
| **Project** | Build a Text Summarization System using the TextRank Algorithm |
| **Programming Language** | Python |
| **Algorithm** | TextRank (Graph-based Extractive Summarization) |

---

# 🎯 Project Objective

Develop an **extractive text summarization system** that automatically identifies and extracts the most important sentences from a document using the **TextRank** algorithm.

### Input
- DUC_TEXT dataset

### Output
- `summary.txt` (Generated summary)

---

# 📖 Methodology

The project adopts a **Graph-based Text Summarization** approach.

Each sentence is represented as a node in a graph.

The similarity between sentences is calculated using **TF-IDF** and **Cosine Similarity**.

The **TextRank (PageRank)** algorithm is then applied to rank sentence importance, and the highest-ranked sentences are selected to generate the final summary.

---

# ⚙️ Project Workflow (main.py)

The program follows six main steps:

1. 📂 Read data from the **DUC_TEXT** dataset.
2. 🧹 Preprocess the text and split it into sentences.
3. 📊 Build TF-IDF vectors.
4. 🔗 Compute the sentence similarity matrix.
5. ⭐ Rank sentences using the **TextRank** algorithm.
6. 📝 Generate and save the summary.

---

# 🔍 TextRank Features (7 features)

The implementation uses a total of **7 features**:

1. **Stop-word Removal** — loại từ dừng tiếng Anh (`stop_words="english"`) → `vector.py`
2. **TF-IDF Unigram** — trọng số từ đơn (`ngram_range=(1, 2)`) → `vector.py`
3. **TF-IDF Bigram** — trọng số cụm 2 từ (`ngram_range=(1, 2)`) → `vector.py`
4. **Sentence Length** — độ dài câu, chuẩn hóa theo câu dài nhất → `vector.py`
5. **Sentence Position** — vị trí câu, câu đầu được ưu tiên → `vector.py`
6. **Numeric Content** — câu có chứa số liệu hay không → `vector.py`
7. **Proper Nouns / Capitalization** — tỷ lệ từ viết hoa / tên riêng → `vector.py`

---

# 🏆 Sentence Ranking

Sentence importance is calculated using the **TextRank (PageRank)** algorithm with Power Iteration.

The similarity matrix is enhanced by blending Cosine Similarity with additional feature weights:

```
enhanced_sim = α × cosine_sim + (1 − α) × feature_matrix   (α = 0.7)
```

Higher-ranked sentences are selected to produce the final extractive summary.

---

# 📄 Output

```
output/
└── <docname>_summary.txt
```

Contains the generated summary of each input document.

---

# 📈 Evaluation

The generated summaries are evaluated using **Cosine TF-IDF Similarity** — measuring the cosine of the angle between the TF-IDF vectors of the system summary and the reference summary (DUC_SUM).

A score closer to **1.0** indicates higher lexical overlap with the reference.

---

# 📂 Project Structure

```
Project(NLP)/
│
├── data/
│   ├── DUC_TEXT/       # Input documents (tagged <s>...</s>)
│   └── DUC_SUM/        # Reference summaries
│
├── output/             # Generated summaries
│
├── src/
│   ├── preprocess.py   # Read & split DUC_TEXT sentences
│   ├── vector.py       # TF-IDF, Cosine Similarity, additional features
│   ├── textrank.py     # Stochastic Matrix, PageRank, feature blending
│   ├── main.py         # Pipeline orchestrator
│   ├── evaluate.py     # Evaluation (Cosine TF-IDF) + commentary
│   ├── similarity.py   # Sentence similarity helpers (web UI)
│   ├── api.py          # Flask API for web interface
│   ├── web_text.py     # Sentence splitter for free-form text
│   └── static/
│       └── index.html  # Web interface
│
├── library.txt         # Python dependencies
├── README.md
└── WEB_UI_README.md
```

---

# 🚀 Technologies

- Python
- scikit-learn (TF-IDF, Cosine Similarity)
- NumPy (custom PageRank implementation, no NetworkX)
- Flask (web interface)
- NLTK (sentence tokenization for web input)
