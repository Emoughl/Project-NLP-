from underthesea import sent_tokenize
from pathlib import Path


def read_text(filename):
    """
    Đọc nội dung file txt
    """
    file_path = Path(__file__).parent.parent / "data" / "raw" / filename

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def split_sentences(text):
    """
    Tách văn bản thành các câu
    """
    sentences = sent_tokenize(text)

    return sentences