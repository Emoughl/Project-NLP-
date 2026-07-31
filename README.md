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

# 🔍 TextRank Features

The implementation includes the following features:

- TF-IDF vectorization
- Stop-word removal
- Unigram & Bigram representation
- Sentence similarity using Cosine Similarity
- Sentence position weighting

---

# 🏆 Sentence Ranking

Sentence importance is calculated using the **TextRank (PageRank)** algorithm.

Higher-ranked sentences are selected to produce the final extractive summary.

---

# 📄 Output

```
summary.txt
```

Contains the generated summary of the input document.

---

# 📈 Evaluation

The generated summaries are evaluated using **ROUGE** metrics.

Evaluation includes:

- ROUGE-1
- ROUGE-2
- ROUGE-L

These metrics compare the generated summary with the reference summary provided in the DUC dataset.

---

# 📂 Project Structure

```
Project/
│
├── data/
│   ├── DUC_TEXT/
│   └── DUC_SUM/
│
├── output/
│   └── summary.txt
│
├── src/
│   ├── evaluate.py
│   ├── main.py
│   ├── preprocess.py
│   ├── similarity.py
│   ├── textrank.py
│   └── vector.py
│
└── README.md
```

---

# 🚀 Technologies

- Python
- scikit-learn
- NetworkX
- NumPy
- ROUGE