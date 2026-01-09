import re

from .models.coconut import COCONUTGPT2
from .models.codi import CODIGPT2
from .models.colar import ColarLlama


def extract_answer_number(sentence: str) -> float:
    try:
        numbers = re.findall(r"-?\d+\.?\d*", sentence.replace(",", ""))
    except:
        return float("inf")
    return float(numbers[-1]) if numbers else float("inf")


def coconut_extract_answer_number(sentence: str) -> float:
    ans = sentence.split("#")[-1].replace(",", "").strip()
    try:
        return float(ans)
    except ValueError:
        return float("inf")


def colar_extract_answer_number(sentence: str) -> float:
    try:
        answer_template = "Answer:"
        ans = sentence.strip("#").split(answer_template)[-1]
        return float(ans)
    except ValueError:
        return float("inf")


MODELS = {
    "coconut": {
        "class": COCONUTGPT2,
        "id": "checkpoints/coconut",
        "answer_extractor": coconut_extract_answer_number,
    },
    "codi": {
        "class": CODIGPT2,
        "id": "checkpoints/codi",
        "answer_extractor": extract_answer_number,
    },
    "colar": {
        "class": ColarLlama,
        "id": "checkpoints/colar",
        "answer_extractor": colar_extract_answer_number,
    },
}
