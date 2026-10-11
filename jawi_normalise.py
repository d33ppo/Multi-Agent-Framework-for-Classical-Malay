"""
Jawi text normalisation and tokenisation shared by Method B (dictionary lookup)
and the Inter-Method Agreement Gate.

Why this exists:
- The same Jawi letter can be typed with different Unicode codepoints
  (e.g. keheh U+06A9 vs kaf U+0643, or presentation forms such as U+FB7A for ca).
- Majalah Qalam (1950-1969) uses pre-1986 spelling, e.g. fa (ف) for p where
  modern Jawi uses pa (ڤ), and teh marbuta (ة) for a final -t. The reference
  dictionary is built from modern Jawi, so lookups retry these known variants.

Normalisation never touches the stored original OCR text; it is only applied
to working copies used for lookup and scoring.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Codepoints that should be treated as the same letter for lookup purposes.
LETTER_MAP = {
    "ک": "ك",  # keheh ک -> kaf ك
    "ی": "ي",  # farsi yeh ی -> yeh ي
    "ى": "ي",  # alef maksura ى -> yeh ي
    "أ": "ا",  # alef with hamza above أ -> alef ا
    "إ": "ا",  # alef with hamza below إ -> alef ا
    "آ": "ا",  # alef with madda آ -> alef ا
}

# Diacritics (harakat), tatweel and zero-width joiners carry no lookup value.
_STRIP_RE = re.compile("[ً-ٰٟـ‌‍‏‎]")

PUNCTUATION_MAP = {
    "،": ",",  # Arabic comma
    "؛": ";",  # Arabic semicolon
    "؟": "?",  # Arabic question mark
    "﴿": "(",  # ornate parenthesis (opening in logical order in Qalam text)
    "﴾": ")",
    "–": "-",
    "—": "-",
}

REDUPLICATION_MARK = "٢"  # Arabic-Indic digit two, e.g. ايبو٢ = ibu-ibu
_ARABIC_DIGITS = {chr(0x0660 + i): str(i) for i in range(10)}

_LETTERS = "ء-غف-يٱ-ۓ"
# A word is a run of Jawi letters, optionally hyphen-joined, optionally followed by ٢.
WORD_RE = re.compile(f"([{_LETTERS}]+(?:-[{_LETTERS}]+)*)({REDUPLICATION_MARK})?")

# Pre-1986 (Za'ba-era) spelling -> modern spelling rewrites, tried in order.
OLD_SPELLING_RULES = (
    ("fa_to_pa", "ف", "ڤ"),  # ف -> ڤ  (hidup: هيدوف -> هيدوڤ)
    ("teh_marbuta_to_teh", "ة", "ت"),  # ة -> ت  (semangat: سماڠة -> سماڠت)
    ("yeh_hamza_to_alef", "ئ", "ا"),  # ئ -> ا  (seorang: سئورڠ -> ساورڠ)
)

# Vowel letters that pre-1986 spelling often leaves out (boleh: بوله vs modern بوليه).
_VOWEL_LETTERS = "اويءئ"


def normalise_jawi(text: str) -> str:
    """Return a lookup-ready copy of Jawi text with consistent codepoints."""
    text = unicodedata.normalize("NFKC", text)
    text = _STRIP_RE.sub("", text)
    return "".join(LETTER_MAP.get(ch, ch) for ch in text)


def normalise_separator(text: str) -> str:
    """Map Jawi punctuation and digits in non-word text to their Rumi equivalents."""
    return "".join(PUNCTUATION_MAP.get(ch) or _ARABIC_DIGITS.get(ch) or ch for ch in text)


@dataclass
class JawiToken:
    """One piece of Jawi text: either a word to romanise or separator text to copy."""

    text: str
    is_word: bool
    reduplicated: bool = False


def tokenise_jawi(text: str) -> list[JawiToken]:
    """Split normalised Jawi text into word and separator tokens, preserving order."""
    tokens: list[JawiToken] = []
    position = 0
    for match in WORD_RE.finditer(text):
        if match.start() > position:
            tokens.append(JawiToken(text[position : match.start()], is_word=False))
        tokens.append(JawiToken(match.group(1), is_word=True, reduplicated=bool(match.group(2))))
        position = match.end()
    if position < len(text):
        tokens.append(JawiToken(text[position:], is_word=False))
    return tokens


def spelling_variants(word: str) -> list[tuple[str, str]]:
    """
    Return (variant, rule_name) pairs to try when a word is not found as written.

    The original word is always first with rule name "exact". Each old-spelling
    rule is tried alone, then all applicable rules together.
    """
    variants = [(word, "exact")]
    seen = {word}
    applicable = [rule for rule in OLD_SPELLING_RULES if rule[1] in word]

    for name, old, new in applicable:
        candidate = word.replace(old, new)
        if candidate not in seen:
            variants.append((candidate, name))
            seen.add(candidate)

    if len(applicable) > 1:
        combined = word
        for _, old, new in applicable:
            combined = combined.replace(old, new)
        if combined not in seen:
            variants.append((combined, "+".join(rule[0] for rule in applicable)))

    return variants


def consonant_skeleton(word: str) -> str:
    """
    Reduce a Jawi word to its consonants so old and modern spellings can meet.

    Applies every old-spelling rule, then drops vowel letters and hamza after
    the first letter (the first letter is kept because it often carries the
    initial vowel, e.g. ايبو ibu). Matches on this key are low-confidence.
    """
    for _, old, new in OLD_SPELLING_RULES:
        word = word.replace(old, new)
    if not word:
        return word
    return word[0] + "".join(ch for ch in word[1:] if ch not in _VOWEL_LETTERS)


_RUMI_PUNCT_RE = re.compile(r"[^\w\s-]")
_SPACE_RE = re.compile(r"\s+")


def normalise_rumi_for_comparison(text: str) -> str:
    """Lowercase Rumi text and drop punctuation so two romanisations can be compared fairly."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _RUMI_PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()
