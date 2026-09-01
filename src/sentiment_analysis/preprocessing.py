"""Pre-processamento leve que preserva sinais uteis de sentimento."""

import html
import re
import unicodedata

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_USER_RE = re.compile(r"(?<!\w)@[\w_]+")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: object) -> str:
    """Normaliza texto sem remover acentos, negacoes, emojis ou pontuacao."""
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", html.unescape(str(text)))
    value = _URL_RE.sub(" URL ", value)
    value = _USER_RE.sub(" USUARIO ", value)
    return _SPACE_RE.sub(" ", value).strip().lower()


def combine_review_text(title: object, body: object) -> str:
    """Combina titulo e corpo sem produzir os literais 'nan' ou 'None'."""
    parts = [normalize_text(part) for part in (title, body)]
    return " — ".join(part for part in parts if part and part not in {"nan", "none"})
