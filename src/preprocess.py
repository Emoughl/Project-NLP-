from pathlib import Path
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "demo"
    / "DUC_TEXT"
    / "train"
)

def read_text(filename):
    file_path = DATA_DIR / filename

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_all_files():
    files = sorted(DATA_DIR.glob("*"))

    return [
        file.name
        for file in files
        if file.is_file()
    ]
    

def split_sentences(text):

    pattern = r"<s[^>]*>(.*?)</s>"

    sentences = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    sentences = [
        re.sub(r"\s+", " ", s).strip()
        for s in sentences
    ]

    return sentences