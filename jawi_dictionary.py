"""
Method B romanisation: rule-based Jawi -> Rumi reverse-lookup dictionary.

Source data (mesolitica/jawi-ms-en on HuggingFace, gathered by the notebooks in
github.com/huseinzol05/malay-dataset/tree/master/normalization/rumi-jawi):
  - wikipedia-1word.json : [[rumi, jawi], ...] single-word pairs
  - rumi-set.json / jawi-set.json : index-aligned 1-6 word phrase windows
    ({"train": [...], "test": [...]}); only "train" is used here.
No license is declared on the HuggingFace dataset card; the gathering repo is
Apache-2.0 and the text is derived from Wikipedia. Record this in the thesis.

The source pairs are Rumi -> Jawi. This module inverts them into a
Jawi -> {Rumi: count} table. Because Jawi does not write short vowels, one Jawi
spelling can map to several Rumi words; lookups return the most frequent
candidate and flag the word as ambiguous when a runner-up is common enough.
Words not in the table are flagged as unknown and kept visible in the output.

Build the table (one-off):
    python jawi_dictionary.py build
Measure coverage on local ground-truth Jawi text:
    python jawi_dictionary.py coverage data/ground_truth/*.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jawi_normalise import (
    consonant_skeleton,
    normalise_jawi,
    normalise_separator,
    spelling_variants,
    tokenise_jawi,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data" / "method_b"
DEFAULT_LOOKUP_PATH = DATA_DIR / "jawi_rumi_lookup.json"

# A runner-up candidate with at least this share of the counts makes a word ambiguous.
AMBIGUITY_MIN_SHARE = 0.2

# Qalam writes the prefixes di- and ke- as detached single letters (د اندونيسيا).
# The dictionary has no reliable entry for a lone letter, so these are fixed.
CLITIC_OVERRIDES = {"د": "di", "ك": "ke"}

# Qalam also often joins these words to the next one (يڠتركنل = yang terkenal).
JOINED_PREFIXES = (("يڠ", "yang"), ("د", "di"), ("ك", "ke"))

UNKNOWN_OPEN, UNKNOWN_CLOSE = "⟦", "⟧"  # ⟦ ⟧ wrap unknown Jawi words in output

_RUMI_WORD_RE = re.compile(r"^[a-z]+(?:-[a-z]+)*$")
_JAWI_WORD_RE = re.compile("^[ء-غف-يٱ-ۓ]+(?:-[ء-غف-يٱ-ۓ]+)*$")
_QUOTES = "'\""
_EDGE_PUNCT = ".,;:!?()[]{}" + "،؛؟"


# ── Building the lookup table ────────────────────────────────────────────────


def clean_pair(rumi: str, jawi: str) -> tuple[str, str] | str:
    """
    Clean one Rumi/Jawi word pair.

    Returns (rumi, jawi) on success, or a rejection reason string.
    Wiki markup quotes ('' / ''') were transliterated into hamza (ء) on the
    Jawi side, so when quotes are stripped from Rumi they are stripped from
    the Jawi edges too.
    """
    rumi, jawi = rumi.strip(), jawi.strip()
    unquoted = rumi.strip(_QUOTES)
    if unquoted != rumi:
        rumi = unquoted
        jawi = jawi.strip(_QUOTES + "ء")
    rumi = rumi.strip(_EDGE_PUNCT).lower()
    jawi = normalise_jawi(jawi).strip(_EDGE_PUNCT)

    if not rumi or not jawi:
        return "empty"
    if not _RUMI_WORD_RE.match(rumi):
        return "rumi_not_alphabetic"
    if not _JAWI_WORD_RE.match(jawi):
        return "jawi_not_letters"
    if rumi.count("-") != jawi.count("-"):
        return "hyphen_mismatch"
    return rumi, jawi


def build_lookup(
    word_pairs: list[list[str]],
    phrase_pairs: list[tuple[str, str]] = (),
) -> tuple[dict[str, dict[str, int]], dict]:
    """Invert Rumi->Jawi pairs into a Jawi->{Rumi: count} table plus a build report."""
    table: dict[str, Counter] = defaultdict(Counter)
    report = {
        "word_pairs_read": len(word_pairs),
        "word_pairs_kept": 0,
        "phrase_pairs_read": len(phrase_pairs),
        "phrase_pairs_aligned": 0,
        "phrase_tokens_kept": 0,
        "rejected": Counter(),
    }

    for pair in word_pairs:
        if len(pair) != 2:
            report["rejected"]["malformed_row"] += 1
            continue
        cleaned = clean_pair(pair[0], pair[1])
        if isinstance(cleaned, str):
            report["rejected"][cleaned] += 1
            continue
        table[cleaned[1]][cleaned[0]] += 1
        report["word_pairs_kept"] += 1

    for rumi_phrase, jawi_phrase in phrase_pairs:
        rumi_words, jawi_words = rumi_phrase.split(), jawi_phrase.split()
        if len(rumi_words) != len(jawi_words):
            report["rejected"]["phrase_token_count_mismatch"] += 1
            continue
        report["phrase_pairs_aligned"] += 1
        for rumi, jawi in zip(rumi_words, jawi_words):
            cleaned = clean_pair(rumi, jawi)
            if isinstance(cleaned, str):
                report["rejected"][f"phrase_{cleaned}"] += 1
                continue
            table[cleaned[1]][cleaned[0]] += 1
            report["phrase_tokens_kept"] += 1

    entries = {jawi: dict(counts.most_common()) for jawi, counts in sorted(table.items())}
    report["rejected"] = dict(report["rejected"].most_common())
    report["jawi_entries"] = len(entries)
    report["ambiguous_entries"] = sum(1 for c in entries.values() if _is_ambiguous(list(c.items())))
    return entries, report


def build_from_files(data_dir: Path = DATA_DIR, include_phrases: bool = True) -> tuple[dict, dict]:
    word_pairs = json.loads((data_dir / "wikipedia-1word.json").read_text(encoding="utf-8"))
    phrase_pairs: list[tuple[str, str]] = []
    if include_phrases:
        rumi_set = json.loads((data_dir / "rumi-set.json").read_text(encoding="utf-8"))
        jawi_set = json.loads((data_dir / "jawi-set.json").read_text(encoding="utf-8"))
        if len(rumi_set["train"]) != len(jawi_set["train"]):
            raise ValueError("rumi-set.json and jawi-set.json train splits are not index-aligned")
        phrase_pairs = list(zip(rumi_set["train"], jawi_set["train"]))
    return build_lookup(word_pairs, phrase_pairs)


# ── Looking words up ─────────────────────────────────────────────────────────


def _is_ambiguous(candidates: list[tuple[str, int]]) -> bool:
    if len(candidates) < 2:
        return False
    total = sum(count for _, count in candidates)
    return candidates[1][1] / total >= AMBIGUITY_MIN_SHARE


@dataclass
class WordLookup:
    """Lookup outcome for one Jawi word."""

    jawi: str
    rumi: str | None
    status: str  # "known" | "ambiguous" | "unknown"
    rule: str = "exact"  # which spelling variant matched, or "clitic_override"
    candidates: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class DictionaryRomanisation:
    """Method B output: romanised text plus per-word evidence."""

    text: str
    words: list[WordLookup]

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def unknown_words(self) -> list[WordLookup]:
        return [w for w in self.words if w.status == "unknown"]

    @property
    def ambiguous_words(self) -> list[WordLookup]:
        return [w for w in self.words if w.status == "ambiguous"]

    @property
    def coverage(self) -> float:
        """Share of words the dictionary could romanise (known or ambiguous)."""
        if not self.words:
            return 0.0
        return 1 - len(self.unknown_words) / len(self.words)

    def case_log(self) -> list[dict]:
        """Ambiguous and unknown words, for the thesis limitations discussion."""
        return [asdict(w) for w in self.words if w.status != "known"]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "word_count": self.word_count,
            "coverage": round(self.coverage, 4),
            "unknown_count": len(self.unknown_words),
            "ambiguous_count": len(self.ambiguous_words),
            "cases": self.case_log(),
        }


class JawiRumiDictionary:
    def __init__(self, entries: dict[str, dict[str, int]]):
        self.entries = entries
        self._skeletons: dict[str, Counter] | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LOOKUP_PATH) -> "JawiRumiDictionary":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Method B lookup table not found at {path}. Build it with: python jawi_dictionary.py build"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["entries"])

    def lookup_word(self, jawi_word: str) -> WordLookup:
        """
        Look up one word, trying in order:
        1. the word and its old-spelling variants (counts pooled, so a rare noisy
           exact entry cannot beat a common modern spelling),
        2. a detached-prefix split (يڠ / د / ك joined to the next word),
        3. the consonant skeleton (always flagged ambiguous).
        """
        if jawi_word in CLITIC_OVERRIDES:
            return WordLookup(jawi_word, CLITIC_OVERRIDES[jawi_word], "known", "clitic_override")

        exact = self._lookup_variants(jawi_word)
        if exact:
            return exact

        for prefix, prefix_rumi in JOINED_PREFIXES:
            rest = jawi_word[len(prefix) :]
            if jawi_word.startswith(prefix) and len(rest) >= 3:
                rest_result = self._lookup_variants(rest)
                if rest_result:
                    return WordLookup(
                        jawi_word,
                        f"{prefix_rumi} {rest_result.rumi}",
                        rest_result.status,
                        f"prefix_split+{rest_result.rule}",
                        rest_result.candidates,
                    )

        skeleton_counts = self._skeleton_index().get(consonant_skeleton(jawi_word))
        if skeleton_counts:
            candidates = skeleton_counts.most_common(5)
            return WordLookup(jawi_word, candidates[0][0], "ambiguous", "consonant_skeleton", candidates)

        return WordLookup(jawi_word, None, "unknown")

    def _lookup_variants(self, jawi_word: str) -> WordLookup | None:
        pooled: Counter = Counter()
        rule_for: dict[str, str] = {}
        for variant, rule in spelling_variants(jawi_word):
            for rumi, count in self.entries.get(variant, {}).items():
                pooled[rumi] += count
                rule_for.setdefault(rumi, rule)
        if not pooled:
            return None
        candidates = pooled.most_common(5)
        status = "ambiguous" if _is_ambiguous(pooled.most_common()) else "known"
        return WordLookup(jawi_word, candidates[0][0], status, rule_for[candidates[0][0]], candidates)

    def _skeleton_index(self) -> dict[str, Counter]:
        """Consonant-skeleton -> {Rumi: count}, built on first use."""
        if self._skeletons is None:
            self._skeletons = defaultdict(Counter)
            for jawi, counts in self.entries.items():
                self._skeletons[consonant_skeleton(jawi)].update(counts)
        return self._skeletons

    def romanise(self, jawi_text: str) -> DictionaryRomanisation:
        """Romanise Jawi text word by word. Never raises on unknown words."""
        parts: list[str] = []
        words: list[WordLookup] = []

        for token in tokenise_jawi(normalise_jawi(jawi_text)):
            if not token.is_word:
                parts.append(normalise_separator(token.text))
                continue

            result = self.lookup_word(token.text)
            if result.status == "unknown" and "-" in token.text:
                result = self._lookup_hyphenated(token.text)
            words.append(result)

            if result.rumi is None:
                parts.append(f"{UNKNOWN_OPEN}{token.text}{UNKNOWN_CLOSE}")
            elif token.reduplicated:
                parts.append(f"{result.rumi}-{result.rumi}")
            else:
                parts.append(result.rumi)

        output = DictionaryRomanisation(text="".join(parts), words=words)
        for case in output.case_log():
            logger.debug("Method B %s word: %s", case["status"], case["jawi"])
        return output

    def _lookup_hyphenated(self, jawi_word: str) -> WordLookup:
        """Look up each half of a hyphenated word that is missing as a whole."""
        halves = [self.lookup_word(part) for part in jawi_word.split("-")]
        if any(h.rumi is None for h in halves):
            return WordLookup(jawi_word, None, "unknown")
        status = "ambiguous" if any(h.status == "ambiguous" for h in halves) else "known"
        return WordLookup(jawi_word, "-".join(h.rumi for h in halves), status, "hyphen_split")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cmd_build(args: argparse.Namespace) -> None:
    entries, report = build_from_files(Path(args.data_dir), include_phrases=not args.words_only)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"meta": report, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Lookup table written to {output}")


def _cmd_coverage(args: argparse.Namespace) -> None:
    dictionary = JawiRumiDictionary.load(args.lookup)
    words, cases = [], []
    for name in args.files:
        result = dictionary.romanise(Path(name).read_text(encoding="utf-8"))
        words.extend(result.words)
        cases.extend({"file": name, **case} for case in result.case_log())

    rules = Counter(w.rule for w in words if w.status != "unknown")
    summary = {
        "files": len(args.files),
        "words": len(words),
        "known": sum(w.status == "known" for w in words),
        "ambiguous": sum(w.status == "ambiguous" for w in words),
        "unknown": sum(w.status == "unknown" for w in words),
        "matched_by_rule": dict(rules.most_common()),
    }
    summary["coverage"] = round(1 - summary["unknown"] / max(1, len(words)), 4)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.cases_out:
        Path(args.cases_out).write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8"
        )
        print(f"Ambiguous/unknown cases written to {args.cases_out}")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Method B Jawi->Rumi reverse-lookup dictionary.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build the lookup table from the downloaded source files.")
    build.add_argument("--data-dir", default=str(DATA_DIR))
    build.add_argument("--output", default=str(DEFAULT_LOOKUP_PATH))
    build.add_argument("--words-only", action="store_true", help="Skip the rumi-set/jawi-set phrase windows.")
    build.set_defaults(func=_cmd_build)

    coverage = sub.add_parser("coverage", help="Report dictionary coverage on Jawi text files.")
    coverage.add_argument("files", nargs="+")
    coverage.add_argument("--lookup", default=str(DEFAULT_LOOKUP_PATH))
    coverage.add_argument("--cases-out", default=None, help="Write ambiguous/unknown cases as JSONL.")
    coverage.set_defaults(func=_cmd_coverage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
