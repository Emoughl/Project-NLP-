from pathlib import Path
import re
from rouge_score import rouge_scorer

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"

REFERENCE_DIR = (
    BASE_DIR
    / "data"
    / "DUC_SUM"
)

def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_reference(text):
    sentences = re.findall(
        r"<s[^>]*>(.*?)</s>",
        text,
        flags=re.DOTALL
    )

    return " ".join(sentences)


scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)

scores_all = []

for summary_file in OUTPUT_DIR.glob("*_summary.txt"):

    doc_name = summary_file.stem.replace("_summary", "")

    reference_file = REFERENCE_DIR / f"{doc_name}"

    if not reference_file.exists():
        print(f": {reference_file.name}")
        continue

    system_summary = read_file(summary_file)

    reference_summary = clean_reference(
        read_file(reference_file)
    )

    score = scorer.score(
        reference_summary,
        system_summary
    )

    scores_all.append(score)

    print("=" * 60)
    print(doc_name)

    print(
        f"ROUGE-1 : {score['rouge1'].fmeasure:.4f}"
    )

    print(
        f"ROUGE-2 : {score['rouge2'].fmeasure:.4f}"
    )

    print(
        f"ROUGE-L : {score['rougeL'].fmeasure:.4f}"
    )

if scores_all:

    avg_r1 = sum(s["rouge1"].fmeasure for s in scores_all) / len(scores_all)

    avg_r2 = sum(s["rouge2"].fmeasure for s in scores_all) / len(scores_all)

    avg_rl = sum(s["rougeL"].fmeasure for s in scores_all) / len(scores_all)

    print("\n---KẾT QUẢ TRUNG BÌNH -----")

    print(f"ROUGE-1 : {avg_r1:.4f}")

    print(f"ROUGE-2 : {avg_r2:.4f}")

    print(f"ROUGE-L : {avg_rl:.4f}")