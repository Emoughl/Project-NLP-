from underthesea import sent_tokenize
from pathlib import Path
import re


def read_text(filename):
    #Đọc nội dung file txt
    file_path = Path(__file__).parent.parent / "data" / "raw" / filename

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        # Xóa các tham chiếu trong văn bản, ví dụ: [1], [2], [3], ...
        text = re.sub(r"\[\d+\]", "", text)

    return text


def split_sentences(text):
    # Tách văn bản thành các câu, gộp lại các câu bị tách nhầm do chứa dấu ngoặc kép chưa cân bằng
    raw_sentences = sent_tokenize(text)

    sentences = []
    buffer = ""

    for s in raw_sentences:
        buffer = f"{buffer} {s}".strip() if buffer else s

        # Nếu số dấu " trong buffer là chẵn -> câu đã trọn vẹn
        if buffer.count('"') % 2 == 0:
            sentences.append(buffer)
            buffer = ""

    # Phòng trường hợp cuối văn bản vẫn còn buffer dở dang
    if buffer:
        sentences.append(buffer)

    return sentences
