"""
Fix for: only 1950_12_012/016/017/018/020 reaching the dataset folder.

Cause: the old discovery matched Drive file titles to article ids with
exact equality (`title in articles`). Most 1950_12 sheets are titled
"1950_12_005_DONE" or "1950_12_005_ DONE", so they never matched.

This module normalises titles with a regex, only considers real Google
Sheets, handles pagination and duplicates, and adds retry/backoff for
the Sheets/Drive 503 errors.
"""
import re
import time

ID_RE = re.compile(r"^\s*(\d{4}_\d{2}_\d{3})")
SHEET_MIME = "application/vnd.google-apps.spreadsheet"


def with_retry(fn, *args, tries=6, base_delay=2.0, **kwargs):
    """Call fn, retrying on 429/5xx style errors with exponential backoff."""
    last = None
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # gspread APIError or googleapiclient HttpError
            msg = str(e)
            transient = any(code in msg for code in ("503", "500", "502", "504", "429", "rateLimit"))
            if not transient or attempt == tries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  transient error ({msg[:60]}...), retry {attempt+1}/{tries-1} in {delay:.0f}s")
            time.sleep(delay)
            last = e
    raise last


def _rank(f):
    """Higher is better when several files share one article id."""
    name = f["name"].upper()
    return (
        f.get("mimeType") == SHEET_MIME,   # real sheet beats csv
        "DONE" in name,                    # corrected beats uncorrected
        f.get("modifiedTime", ""),         # newest wins last
    )


def attach_sheet_ids(articles, drive_service, csv_folder_ids):
    """
    articles: existing dict keyed by article id (e.g. "1950_12_005"),
              built from the PDFs. Each value is a dict; "sheet_id" is added.
    csv_folder_ids: {"1950_12": "<folder id>", ...}
    Returns dict {article_id: [matching file names]} for reporting.
    """
    seen = {}
    for issue, folder_id in csv_folder_ids.items():
        page_token = None
        while True:
            resp = with_retry(
                drive_service.files().list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageSize=200,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ).execute
            )
            for f in resp.get("files", []):
                if f.get("mimeType") != SHEET_MIME:
                    continue
                m = ID_RE.match(f["name"])
                if not m:
                    continue
                seen.setdefault(m.group(1), []).append(f)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    report = {}
    for article_id, files in seen.items():
        if article_id not in articles:
            continue
        best = max(files, key=_rank)
        articles[article_id]["sheet_id"] = best["id"]
        articles[article_id]["sheet_name"] = best["name"]
        report[article_id] = [f["name"] for f in files]
        if len(files) > 1:
            print(f"WARNING {article_id}: {len(files)} sheets {[f['name'] for f in files]}, using '{best['name']}'")

    # NOTE: the old discovery pre-creates "sheet_id": None, so test the VALUE, not the key.
    missing = sorted(a for a, v in articles.items() if not v.get("sheet_id"))
    if missing:
        print(f"No sheet found for: {missing}")
    return report


def sanity_check_lines(article_id, lines, min_lines=3):
    """Flag articles whose picked ground-truth column looks empty/wrong."""
    if len(lines) < min_lines:
        print(f"CHECK {article_id}: only {len(lines)} GT line(s). Corrected column may be empty.")
        return False
    return True


def needs_generation(out_json, max_mb=20):
    """
    True if ls_tasks.json is missing OR suspiciously large (made by the old
    generate_tasks.py that embedded full PNGs, which Label Studio cannot open).
    """
    from pathlib import Path
    p = Path(out_json)
    if not p.exists():
        return True
    size_mb = p.stat().st_size / 1e6
    if size_mb > max_mb:
        print(f"  {p.parent.name}: existing ls_tasks.json is {size_mb:.0f}MB (old format), will regenerate")
        return True
    return False
