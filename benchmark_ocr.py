"""
Benchmarking framework for Jawi OCR methods.

Compares Tesseract, Claude, and ILMU on test manuscript images.

Metrics:
  - Processing time (wall-clock)
  - Text length / word count
  - OCR confidence (Tesseract) or N/A (LLM)
  - Pairwise character-level similarity (Levenshtein ratio)
  - Cost estimate (LLM only)

Outputs:
  - Console summary table
  - output/benchmark_results.json   (structured results)
  - output/benchmark/               (per-image text files)
  - output/benchmark_report.md      (Markdown report)

Usage:
    python benchmark_ocr.py                         # all images, all methods
    python benchmark_ocr.py --image <path>           # single image
    python benchmark_ocr.py --skip-missing           # skip providers w/o keys
    python benchmark_ocr.py --providers tesseract claude
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Imports — with graceful handling for optional deps
# ---------------------------------------------------------------------------

try:
    from Levenshtein import ratio as levenshtein_ratio
except ImportError:
    # Pure-Python fallback (slower, but works without python-Levenshtein)
    from difflib import SequenceMatcher

    def levenshtein_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # will fall back to simple formatting

from ocr_jawi import extract_jawi_text                     # Tesseract
from ocr_llm import extract_jawi_text_llm, PROVIDERS       # LLM providers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).resolve().parent / "test-data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BENCHMARK_DIR = OUTPUT_DIR / "benchmark"

ALL_IMAGES = sorted(TEST_DATA_DIR.glob("jawi-manuscript*.*"))

# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _run_tesseract(image_path: str) -> dict | None:
    """Run Tesseract OCR and return result dict (or None on failure)."""
    try:
        result = extract_jawi_text(image_path)
        result["provider"] = "tesseract"
        result["model"] = "Tesseract LSTM (ara)"
        result["cost_estimate_usd"] = 0.0
        result["usage"] = {}
        return result
    except Exception as e:
        print(f"  [tesseract] ERROR: {e}")
        return None


def _run_llm(image_path: str, provider: str) -> dict | None:
    """Run an LLM provider and return result dict (or None on failure)."""
    try:
        return extract_jawi_text_llm(image_path, provider)
    except EnvironmentError as e:
        # Missing API key
        print(f"  [{provider}] SKIPPED — {e}")
        return None
    except Exception as e:
        print(f"  [{provider}] ERROR: {e}")
        return None


def _text_stats(text: str) -> dict:
    """Compute basic text statistics."""
    chars = len(text)
    words = len(text.split()) if text else 0
    lines = text.count("\n") + 1 if text else 0
    return {"char_count": chars, "word_count": words, "line_count": lines}


def _pairwise_similarity(results: dict[str, dict]) -> list[dict]:
    """Compute pairwise Levenshtein similarity between all methods."""
    providers = list(results.keys())
    pairs = []
    for a, b in combinations(providers, 2):
        text_a = results[a]["raw_text"]
        text_b = results[b]["raw_text"]
        sim = levenshtein_ratio(text_a, text_b) if (text_a and text_b) else 0.0
        pairs.append({
            "pair": f"{a} ↔ {b}",
            "similarity": round(sim * 100, 2),
        })
    return pairs


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def benchmark_image(
    image_path: str,
    providers: list[str],
    skip_missing: bool = False,
) -> dict:
    """Benchmark all requested providers on a single image."""
    image_name = Path(image_path).name
    print(f"\n{'='*60}")
    print(f"  Image: {image_name}")
    print(f"{'='*60}")

    results: dict[str, dict] = {}

    # --- Tesseract ---
    if "tesseract" in providers:
        print("  Running Tesseract ...", end=" ", flush=True)
        res = _run_tesseract(image_path)
        if res:
            results["tesseract"] = res
            print(f"done ({res['processing_time']}s)")

    # --- LLM providers ---
    for prov in providers:
        if prov == "tesseract":
            continue
        print(f"  Running {prov.upper()} ...", end=" ", flush=True)
        res = _run_llm(image_path, prov)
        if res:
            results[prov] = res
            print(f"done ({res['processing_time']}s)")
        elif skip_missing:
            continue
        # if res is None and not skip_missing, error already printed

    # --- Compute stats & similarity ---
    stats = {}
    for prov, res in results.items():
        stats[prov] = _text_stats(res["raw_text"])

    similarity = _pairwise_similarity(results)

    return {
        "image": image_name,
        "image_path": str(image_path),
        "results": results,
        "stats": stats,
        "similarity": similarity,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_summary_table(benchmarks: list[dict]) -> str:
    """Print and return a formatted summary table."""
    rows = []
    for bm in benchmarks:
        for prov, res in bm["results"].items():
            st = bm["stats"].get(prov, {})
            rows.append({
                "Image": bm["image"],
                "Provider": prov.upper(),
                "Model": res.get("model", "—"),
                "Time (s)": res["processing_time"],
                "Chars": st.get("char_count", "—"),
                "Words": st.get("word_count", "—"),
                "Confidence": res.get("confidence", "—"),
                "Cost ($)": res.get("cost_estimate_usd") or "—",
            })

    if tabulate:
        table = tabulate(rows, headers="keys", tablefmt="github")
    else:
        # simple fallback
        if not rows:
            table = "(no results)"
        else:
            headers = list(rows[0].keys())
            col_widths = {h: max(len(str(h)), max(len(str(r[h])) for r in rows)) for h in headers}
            header_row = " | ".join(h.ljust(col_widths[h]) for h in headers)
            sep_row = "-|-".join("-" * col_widths[h] for h in headers)
            data_rows = "\n".join(
                " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers)
                for r in rows
            )
            table = f"{header_row}\n{sep_row}\n{data_rows}"

    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}\n")
    print(table)

    # Similarity
    all_sims = []
    for bm in benchmarks:
        if bm["similarity"]:
            all_sims.append((bm["image"], bm["similarity"]))
    if all_sims:
        print(f"\n{'='*60}")
        print("  PAIRWISE SIMILARITY (character-level)")
        print(f"{'='*60}")
        for img, sims in all_sims:
            for s in sims:
                print(f"  {img}  {s['pair']}  →  {s['similarity']}%")

    return table


def _save_text_outputs(benchmarks: list[dict]) -> None:
    """Save each method's extracted text to output/benchmark/."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    for bm in benchmarks:
        stem = Path(bm["image"]).stem
        for prov, res in bm["results"].items():
            fname = f"{stem}_{prov}.txt"
            path = BENCHMARK_DIR / fname
            with open(path, "w", encoding="utf-8") as f:
                f.write(res["raw_text"])


def _save_json(benchmarks: list[dict], path: Path) -> None:
    """Save structured benchmark results to JSON."""

    def _serialise(obj):
        """Make non-serialisable objects JSON-friendly."""
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "benchmarks": benchmarks,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=_serialise,
        )
    print(f"\nJSON results  → {path}")


def _save_markdown_report(benchmarks: list[dict], table_str: str, path: Path) -> None:
    """Save a Markdown benchmark report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Jawi OCR Benchmark Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Summary",
        "",
        table_str,
        "",
    ]

    # Similarity section
    has_sim = any(bm["similarity"] for bm in benchmarks)
    if has_sim:
        lines.append("## Pairwise Similarity (character-level Levenshtein ratio)")
        lines.append("")
        lines.append("| Image | Pair | Similarity |")
        lines.append("|---|---|---|")
        for bm in benchmarks:
            for s in bm["similarity"]:
                lines.append(f"| {bm['image']} | {s['pair']} | {s['similarity']}% |")
        lines.append("")

    # Per-image details
    lines.append("## Per-Image Details")
    lines.append("")
    for bm in benchmarks:
        lines.append(f"### {bm['image']}")
        lines.append("")
        for prov, res in bm["results"].items():
            st = bm["stats"].get(prov, {})
            lines.append(f"**{prov.upper()}** — {res.get('model', '—')}")
            lines.append(f"- Time: {res['processing_time']}s")
            lines.append(f"- Characters: {st.get('char_count', '—')}  |  Words: {st.get('word_count', '—')}  |  Lines: {st.get('line_count', '—')}")
            lines.append(f"- Confidence: {res.get('confidence', '—')}")
            cost = res.get("cost_estimate_usd")
            if cost is not None:
                lines.append(f"- Est. Cost: ${cost}")
            lines.append("")
            # Preview first 5 lines of extracted text
            preview = "\n".join(res["raw_text"].splitlines()[:5])
            lines.append(f"<details><summary>Text preview (first 5 lines)</summary>\n\n```\n{preview}\n```\n\n</details>")
            lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Jawi OCR methods (Tesseract vs. LLM providers)."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image to benchmark. Default: all test images."
    )
    parser.add_argument(
        "--providers", nargs="+", default=["tesseract", "claude", "ilmu", "zai"],
        help="OCR providers to include. Default: tesseract claude ilmu zai"
    )
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="Silently skip providers whose API keys are not configured."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory."
    )

    args = parser.parse_args()

    if args.output_dir:
        global OUTPUT_DIR, BENCHMARK_DIR
        OUTPUT_DIR = Path(args.output_dir)
        BENCHMARK_DIR = OUTPUT_DIR / "benchmark"

    # Determine image list
    if args.image:
        images = [Path(args.image)]
    else:
        images = ALL_IMAGES
        if not images:
            print(f"No manuscript images found in {TEST_DATA_DIR}")
            sys.exit(1)

    print(f"Benchmark started — {len(images)} image(s), providers: {args.providers}")

    benchmarks = []
    for img in images:
        bm = benchmark_image(
            str(img),
            providers=args.providers,
            skip_missing=args.skip_missing,
        )
        benchmarks.append(bm)

    # Save outputs
    _save_text_outputs(benchmarks)
    table_str = _print_summary_table(benchmarks)
    _save_json(benchmarks, OUTPUT_DIR / "benchmark_results.json")
    _save_markdown_report(benchmarks, table_str, OUTPUT_DIR / "benchmark_report.md")

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()
