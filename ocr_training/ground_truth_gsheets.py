"""
Ground truth loading for Jawi Qalam data stored as Google Sheets
(one Sheet per article, under Al_Qalam_Rumi_by_batches/<issue>/csv/).

This replaces the openpyxl-based ground_truth.py for use in Colab, where
the ground truth lives as native Google Sheets rather than downloaded
.xlsx files. Requires running inside Colab with Drive mounted AND
Google auth available (Colab provides this automatically via
google.colab.auth).

Column identification still uses the CONTENT-based approach (majority
Arabic-script text), not header name matching, since real files were
confirmed to disagree on header naming and column count -- some sheets
have an unnamed 4th "jawi_corrected" column with no header cell at all.
"""

import re

ARABIC_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]


def _is_arabic_char(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ARABIC_RANGES)


def _arabic_fraction(text):
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic_count = sum(1 for c in letters if _is_arabic_char(c))
    return arabic_count / len(letters)


def _find_jawi_column(rows, header, sample_size=8):
    sample_rows = rows[1:1 + sample_size]
    n_cols = max([len(header)] + [len(r) for r in sample_rows])

    jawi_like_cols = []
    for col_idx in range(n_cols):
        fractions = []
        for row in sample_rows:
            if col_idx >= len(row):
                continue
            val = row[col_idx]
            if isinstance(val, str) and val.strip():
                fractions.append(_arabic_fraction(val))
        min_occurrences = max(2, len(sample_rows) // 2)
        if len(fractions) >= min_occurrences and (sum(fractions) / len(fractions)) > 0.5:
            jawi_like_cols.append(col_idx)

    if not jawi_like_cols:
        raise ValueError(
            f"No column with predominantly Arabic/Jawi script content found "
            f"consistently across rows. Header was: {header}"
        )
    return jawi_like_cols[-1]


def load_article_lines_from_sheet(spreadsheet_id, gspread_client=None):
    """
    Reads a Google Sheet by its file ID and returns a flat list of
    ground-truth line strings in reading order.

    In Colab:
        import gspread
        from google.colab import auth
        from google.auth import default

        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)

        lines = load_article_lines_from_sheet(spreadsheet_id, gspread_client=gc)

    spreadsheet_id is the file ID from the Drive URL, e.g. for
    https://docs.google.com/spreadsheets/d/1K3ClmxLOjVncI3vl28aQpGspDMaVXx0uNTD3Pi98vIU/edit
    the id is 1K3ClmxLOjVncI3vl28aQpGspDMaVXx0uNTD3Pi98vIU
    """
    if gspread_client is None:
        raise RuntimeError(
            "gspread_client is required. In Colab, authenticate first:\n"
            "  from google.colab import auth\n"
            "  auth.authenticate_user()\n"
            "  import gspread\n"
            "  from google.auth import default\n"
            "  creds, _ = default()\n"
            "  gc = gspread.authorize(creds)\n"
            "Then pass gc as gspread_client."
        )

    sheet = gspread_client.open_by_key(spreadsheet_id)
    ws = sheet.sheet1
    rows = ws.get_all_values()  # list of lists of strings
    if not rows:
        return []

    header = rows[0]
    jawi_col = _find_jawi_column(rows, header)

    lines = []
    for row in rows[1:]:
        if jawi_col >= len(row):
            continue
        cell = row[jawi_col]
        if not cell:
            continue
        for raw_line in cell.split("\n"):
            clean = raw_line.strip()
            if clean:
                lines.append(clean)
    return lines


def extract_spreadsheet_id(url_or_id):
    """Accepts either a raw file ID or a full Drive/Sheets URL and returns the ID."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id
