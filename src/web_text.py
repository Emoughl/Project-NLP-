"""
Sentence splitter dùng riêng cho giao diện web (nhận văn bản tự do do
người dùng dán vào), khác với preprocess.split_sentences() vốn chỉ đọc
được định dạng gắn thẻ <s>...</s> của dataset DUC_TEXT.

Không đụng tới preprocess.py / main.py hiện có — module này độc lập,
chỉ được api.py gọi tới.
"""

import re

_HAS_NLTK = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize

    def _ensure_punkt():
        for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
                return True
            except LookupError:
                continue
        # Chưa có dữ liệu punkt -> thử tải (cần mạng, chỉ chạy 1 lần).
        for pkg in ("punkt_tab", "punkt"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
                return True
            except LookupError:
                continue
        return False

    _HAS_NLTK = _ensure_punkt()
except Exception:
    _HAS_NLTK = False


_DUC_TAG_PATTERN = re.compile(r"<s[^>]*>(.*?)</s>", re.DOTALL | re.IGNORECASE)


def _split_duc_tagged(text):
    """
    Nếu người dùng lỡ dán nguyên văn bản có định dạng thẻ của dataset DUC_TEXT
    (<s docid="..." num="..." wdcount="...">...</s>), bóc câu ra từ trong thẻ
    thay vì tách theo dấu câu — tránh việc các thuộc tính docid/num/wdcount
    (vốn gần như không lặp lại giữa các câu) làm nhiễu TF-IDF.
    """
    matches = _DUC_TAG_PATTERN.findall(text)
    if not matches:
        return None
    sentences = [re.sub(r"\s+", " ", s).strip() for s in matches]
    sentences = [s for s in sentences if s]
    return sentences or None


def _regex_split(text):
    """Fallback: tách câu bằng regex khi không có nltk/punkt."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def clean_reference_text(text):
    """
    Chuẩn hoá văn bản tóm tắt tham chiếu (dùng để tính Cosine TF-IDF):
    nếu là file dạng DUC_SUM (có thẻ <s>...</s>) thì bóc câu ra và nối lại,
    ngược lại chỉ chuẩn hoá khoảng trắng.
    """
    text = (text or "").strip()
    if not text:
        return ""

    duc_sentences = _split_duc_tagged(text)
    if duc_sentences:
        return " ".join(duc_sentences)

    return re.sub(r"\s+", " ", text).strip()


def split_sentences_generic(text):
    """Tách một đoạn văn bản tự do thành danh sách câu."""
    text = (text or "").strip()
    if not text:
        return []

    # Ưu tiên nhận diện định dạng thẻ <s>...</s> của dataset DUC_TEXT trước.
    duc_sentences = _split_duc_tagged(text)
    if duc_sentences:
        return duc_sentences

    if _HAS_NLTK:
        try:
            sentences = sent_tokenize(text)
            sentences = [re.sub(r"\s+", " ", s).strip() for s in sentences if s.strip()]
            if sentences:
                return sentences
        except Exception:
            pass

    return _regex_split(text)
