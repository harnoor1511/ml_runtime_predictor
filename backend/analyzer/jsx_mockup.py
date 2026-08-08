"""Lightweight, LLM-free mockup generator.

Parses JSX/TSX/Vue/Svelte source with regex heuristics to pull out headings,
button labels, input placeholders, labels, and static text, then assembles a
templated HTML mockup WITH synthesized realistic sample data (not just the
literal labels found in source). No model inference involved, so it runs in
milliseconds instead of minutes — trades closeness-to-real-UI for speed.
"""

import re
import html as html_lib
import random

HEADING_TAG_RE = re.compile(r"<(h[1-6]|H[1-6])[^>]*>(.*?)</\1>", re.DOTALL)
CUSTOM_TITLE_RE = re.compile(
    r"<(Title|Heading|CardTitle|PageTitle|SectionTitle)[^>]*>(.*?)</\1>", re.DOTALL
)
BUTTON_RE = re.compile(r"<(button|Button)[^>]*>(.*?)</\1>", re.DOTALL)
PLACEHOLDER_RE = re.compile(r'placeholder=["\']([^"\']+)["\']')
LABEL_RE = re.compile(r"<(label|Label)[^>]*>(.*?)</\1>", re.DOTALL)
PARAGRAPH_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL)
TH_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.DOTALL)
JSX_EXPR_RE = re.compile(r"\{[^{}]*\}")  # strip simple {expr} interpolations
TAG_RE = re.compile(r"<[^>]+>")

# Section-type accent colors (cycled by hash of filename) and icons chosen by keyword.
ACCENTS = ["#3C6E71", "#8E5B3F", "#3E5C8A", "#7A4E8C"]
ICON_RULES = [
    (("table", "list", "grid", "results"), "\U0001F4CA"),   # bar chart
    (("upload", "import", "dataset", "file"), "\U0001F4E4"),  # outbox
    (("chat", "reply", "message", "conversation"), "\U0001F4AC"),  # speech balloon
    (("card", "result", "summary"), "\U0001F4C4"),  # page
    (("panel", "settings", "config"), "\u2699\uFE0F"),  # gear
    (("app", "layout", "main"), "\U0001F5A5\uFE0F"),  # desktop
]
DEFAULT_ICON = "\U0001F9E9"  # puzzle piece

# Keyword -> generators for synthesizing a plausible example value for a short
# label/heading (used both for standalone "field" headings and table columns).
SAMPLE_GENERATORS = {
    "id": lambda i: f"#{1040 + i}",
    "ticket": lambda i: f"TCK-{1040 + i}",
    "name": lambda i: random.choice(["Alicia Renner", "Marcus Cole", "Priya Nair", "Devon Wu"]),
    "customer": lambda i: random.choice(["Alicia Renner", "Marcus Cole", "Priya Nair", "Devon Wu"]),
    "email": lambda i: random.choice(["a.renner@mail.com", "m.cole@mail.com", "p.nair@mail.com"]),
    "date": lambda i: random.choice(["Jul 24, 2026", "Jul 25, 2026", "2 hours ago", "Yesterday"]),
    "time": lambda i: random.choice(["09:41", "14:12", "22:03"]),
    "created": lambda i: random.choice(["2 hours ago", "Yesterday", "3 days ago"]),
    "updated": lambda i: random.choice(["Just now", "10 min ago", "1 hour ago"]),
    "status": lambda i: random.choice(["Open", "Resolved", "Pending", "Escalated"]),
    "priority": lambda i: random.choice(["High", "Medium", "Low"]),
    "severity": lambda i: random.choice(["High", "Medium", "Low"]),
    "score": lambda i: f"{random.randint(72, 98)}%",
    "confidence": lambda i: f"{random.randint(72, 98)}%",
    "sentiment": lambda i: random.choice(["Positive", "Neutral", "Frustrated"]),
    "category": lambda i: random.choice(["Billing", "Shipping", "Account access", "Refund"]),
    "type": lambda i: random.choice(["Billing", "Shipping", "Account access", "Refund"]),
    "tag": lambda i: random.choice(["billing", "urgent", "refund", "follow-up"]),
    "reply": lambda i: "Hi Alicia — I've refunded the duplicate charge; it'll post to your card within 3-5 business days.",
    "response": lambda i: "Hi Alicia — I've refunded the duplicate charge; it'll post to your card within 3-5 business days.",
    "message": lambda i: "Hi Alicia — I've refunded the duplicate charge; it'll post to your card within 3-5 business days.",
    "internal": lambda i: "Customer was double-charged due to a retry bug on our end. Refund issued, no follow-up needed.",
    "summary": lambda i: "Customer was double-charged due to a retry bug on our end. Refund issued, no follow-up needed.",
    "note": lambda i: "Customer was double-charged due to a retry bug on our end. Refund issued, no follow-up needed.",
    "row": lambda i: f"{random.randint(120, 600)} rows",
    "count": lambda i: str(random.randint(3, 42)),
    "file": lambda i: random.choice(["support_tickets_q3.csv", "chat_logs_export.json"]),
}
DEFAULT_SAMPLE_WORDS = ["Sample value", "Example entry", "Placeholder data", "Demo record"]


def guess_sample(label, i=0):
    low = label.lower()
    for key, gen in SAMPLE_GENERATORS.items():
        if key in low:
            return gen(i)
    return DEFAULT_SAMPLE_WORDS[i % len(DEFAULT_SAMPLE_WORDS)]


def is_badge_field(label):
    low = label.lower()
    return any(k in low for k in ("status", "priority", "severity", "sentiment", "tag"))


def badge_class(value):
    v = value.lower()
    if v in ("high", "escalated", "urgent", "frustrated"):
        return "mock-badge mock-badge-red"
    if v in ("medium", "pending"):
        return "mock-badge mock-badge-amber"
    return "mock-badge mock-badge-green"


def clean_text(raw):
    if not raw:
        return ""
    text = JSX_EXPR_RE.sub("", raw)
    text = TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_component_elements(content):
    headings = []
    for pattern in (HEADING_TAG_RE, CUSTOM_TITLE_RE):
        for m in pattern.finditer(content):
            text = clean_text(m.group(2))
            if text and len(text) < 120:
                headings.append(text)

    buttons = []
    for m in BUTTON_RE.finditer(content):
        text = clean_text(m.group(2))
        if text and len(text) < 60:
            buttons.append(text)

    inputs = [p for p in PLACEHOLDER_RE.findall(content) if len(p) < 80]

    labels = []
    for m in LABEL_RE.finditer(content):
        text = clean_text(m.group(2))
        if text and len(text) < 60:
            labels.append(text)

    paragraphs = []
    for m in PARAGRAPH_RE.finditer(content):
        text = clean_text(m.group(1))
        if text and 5 < len(text) < 200:
            paragraphs.append(text)

    table_headers = []
    for m in TH_RE.finditer(content):
        text = clean_text(m.group(1))
        if text and len(text) < 40:
            table_headers.append(text)

    return {
        "headings": dedupe(headings)[:6],
        "buttons": dedupe(buttons)[:8],
        "inputs": dedupe(inputs)[:8],
        "labels": dedupe(labels)[:8],
        "paragraphs": dedupe(paragraphs)[:4],
        "table_headers": dedupe(table_headers)[:8],
    }


def dedupe(items):
    seen = set()
    out = []
    for i in items:
        if i.lower() not in seen:
            seen.add(i.lower())
            out.append(i)
    return out


def pick_icon(file_label):
    low = file_label.lower()
    for keywords, icon in ICON_RULES:
        if any(k in low for k in keywords):
            return icon
    return DEFAULT_ICON


def render_table(columns, n_rows=3):
    if not columns:
        columns = ["ID", "Item", "Category", "Status", "Priority"]
    head = "".join(f"<th>{html_lib.escape(c)}</th>" for c in columns)
    rows = []
    for r in range(n_rows):
        cells = []
        for c in columns:
            val = str(guess_sample(c, r))
            if is_badge_field(c):
                cells.append(f'<td><span class="{badge_class(val)}">{html_lib.escape(val)}</span></td>')
            else:
                cells.append(f"<td>{html_lib.escape(val)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="mock-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def looks_like_field_label(text):
    # Short, no trailing punctuation, no spaces-heavy sentence -> treat as a
    # field/section label that should get a synthesized example value under it,
    # rather than a already-complete sentence.
    return bool(text) and len(text) < 45 and not text.rstrip().endswith((".", "!", "?"))


def build_fast_mockup_html(info):
    """Assemble a templated HTML mockup from parsed component elements, with
    synthesized realistic sample data. No LLM call — pure static analysis +
    templating."""
    repo = info.get("repo", "Repository")
    description = info.get("description") or ""
    fs = info.get("frontend_source", {})
    files = fs.get("files", [])

    sections_html = []
    any_content_found = False

    for idx, f in enumerate(files):
        el = extract_component_elements(f["content"])
        if not any(el.values()):
            continue
        any_content_found = True

        file_label = f["path"].split("/")[-1]
        is_table_component = (
            "table" in file_label.lower()
            or el["table_headers"]
            or "list" in file_label.lower()
        )
        is_upload_component = "upload" in file_label.lower() or any(
            "csv" in p.lower() or "json" in p.lower() or "file" in p.lower() for p in el["inputs"]
        )

        accent = ACCENTS[idx % len(ACCENTS)]
        icon = pick_icon(file_label)
        parts = [f'<div class="mock-section" style="border-left-color:{accent}">']
        parts.append(
            f'<div class="mock-section-label"><span class="mock-icon">{icon}</span>'
            f'{html_lib.escape(file_label)}</div>'
        )

        # Table-like component -> render an actual synthesized table instead of raw labels
        if is_table_component:
            columns = el["table_headers"] or el["labels"] or el["headings"] or []
            for h in el["headings"]:
                parts.append(f'<h2 class="mock-heading">{html_lib.escape(h)}</h2>')
            parts.append(render_table(columns[:6]))
            sections_html.append("\n".join(parts) + "</div>")
            continue

        # Upload/dropzone-like component -> render a dropzone with a sample file already "in" it
        if is_upload_component:
            for h in el["headings"]:
                parts.append(f'<h2 class="mock-heading">{html_lib.escape(h)}</h2>')
            for p in el["paragraphs"]:
                parts.append(f'<p class="mock-text">{html_lib.escape(p)}</p>')
            sample_file = guess_sample("file")
            sample_rows = guess_sample("row")
            parts.append(
                '<div class="mock-dropzone">'
                f'<div class="mock-dropzone-file">\U0001F4C4 {html_lib.escape(sample_file)} '
                f'<span class="mock-muted">&middot; {html_lib.escape(sample_rows)}</span></div>'
                '<div class="mock-muted">drop another file to replace</div>'
                '</div>'
            )
            if el["buttons"]:
                parts.append('<div class="mock-buttons">')
                for i, b in enumerate(el["buttons"]):
                    btn_class = "mock-btn mock-btn-primary" if i == 0 else "mock-btn mock-btn-secondary"
                    parts.append(f'<button class="{btn_class}" disabled>{html_lib.escape(b)}</button>')
                parts.append('</div>')
            sections_html.append("\n".join(parts) + "</div>")
            continue

        # Generic component: headings/paragraphs, but synthesize example content
        # under short "field label" style headings instead of leaving them bare.
        for h in el["headings"]:
            parts.append(f'<h2 class="mock-heading">{html_lib.escape(h)}</h2>')
            if looks_like_field_label(h) and not el["paragraphs"]:
                example = guess_sample(h)
                parts.append(f'<p class="mock-example">{html_lib.escape(example)}</p>')

        for p in el["paragraphs"]:
            parts.append(f'<p class="mock-text">{html_lib.escape(p)}</p>')

        if el["labels"] or el["inputs"]:
            parts.append('<div class="mock-form">')
            for lbl in el["labels"]:
                parts.append(f'<label class="mock-label">{html_lib.escape(lbl)}</label>')
            for ph in el["inputs"]:
                sample = guess_sample(ph)
                parts.append(
                    f'<input class="mock-input" placeholder="{html_lib.escape(ph)}" '
                    f'value="{html_lib.escape(sample)}" disabled />'
                )
            parts.append('</div>')

        if el["buttons"]:
            parts.append('<div class="mock-buttons">')
            for i, b in enumerate(el["buttons"]):
                btn_class = "mock-btn mock-btn-primary" if i == 0 else "mock-btn mock-btn-secondary"
                parts.append(f'<button class="{btn_class}" disabled>{html_lib.escape(b)}</button>')
            parts.append('</div>')

        parts.append('</div>')
        sections_html.append("\n".join(parts))

    if not any_content_found:
        file_list = "".join(f'<li>{html_lib.escape(f["path"])}</li>' for f in files)
        sections_html.append(f"""
        <div class="mock-section" style="border-left-color:{ACCENTS[0]}">
          <div class="mock-section-label"><span class="mock-icon">{DEFAULT_ICON}</span>Detected components</div>
          <p class="mock-text">Couldn't extract specific text/labels from the source
          (likely heavy use of dynamic/computed JSX). Components found:</p>
          <ul class="mock-file-list">{file_list}</ul>
        </div>
        """)

    body = "\n".join(sections_html)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, "Segoe UI", sans-serif;
    background: #F0F1EC;
    color: #1B1E23;
    padding: 32px;
  }}
  .mock-banner {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #C6862B;
    background: #FBF1DE;
    border: 1px solid #E9D19E;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 20px;
    display: inline-block;
  }}
  h1.mock-title {{ font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  p.mock-desc {{ color: #565C63; margin: 0 0 24px; max-width: 620px; }}
  .mock-section {{
    background: #fff;
    border: 1px solid #DCDDD4;
    border-left: 4px solid #3C6E71;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 1px 2px rgba(20,20,20,0.04);
  }}
  .mock-section-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8A8F86;
    margin-bottom: 12px;
    font-family: monospace;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .mock-icon {{ font-size: 14px; }}
  h2.mock-heading {{ font-size: 18px; margin: 10px 0 4px; }}
  p.mock-text {{ color: #444; line-height: 1.5; margin: 6px 0; }}
  p.mock-example {{
    color: #2E2E2E; line-height: 1.5; margin: 2px 0 10px;
    background: #F6F5F1; border-left: 3px solid #C9CAC2;
    padding: 8px 12px; border-radius: 4px; font-size: 14px;
  }}
  .mock-muted {{ color: #9A9E96; font-size: 12px; }}
  .mock-form {{ display: flex; flex-direction: column; gap: 8px; margin: 12px 0; max-width: 380px; }}
  .mock-label {{ font-size: 13px; color: #565C63; }}
  .mock-input {{
    border: 1px solid #DCDDD4; border-radius: 6px; padding: 8px 12px;
    font-size: 14px; background: #FAFAF8; color: #333;
  }}
  .mock-buttons {{ display: flex; gap: 10px; margin-top: 12px; }}
  .mock-btn {{
    border-radius: 6px; padding: 9px 18px; font-size: 14px; cursor: default; border: none;
  }}
  .mock-btn-primary {{ background: #1B1E23; color: #F4F5F2; }}
  .mock-btn-secondary {{ background: #fff; color: #1B1E23; border: 1px solid #DCDDD4; }}
  .mock-file-list {{ font-family: monospace; font-size: 13px; color: #565C63; }}
  .mock-dropzone {{
    border: 1.5px dashed #C9CAC2; border-radius: 8px; padding: 16px 18px;
    background: #FAFAF8; margin: 10px 0;
  }}
  .mock-dropzone-file {{ font-size: 14px; margin-bottom: 4px; }}
  table.mock-table {{
    width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13.5px;
  }}
  .mock-table th {{
    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
    color: #8A8F86; padding: 8px 10px; border-bottom: 1px solid #DCDDD4;
  }}
  .mock-table td {{ padding: 9px 10px; border-bottom: 1px solid #ECEDE7; }}
  .mock-table tbody tr:nth-child(odd) {{ background: #FAFAF8; }}
  .mock-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
  }}
  .mock-badge-red {{ background: #FBE4E1; color: #A23B2E; }}
  .mock-badge-amber {{ background: #FBF1DE; color: #A26B1E; }}
  .mock-badge-green {{ background: #E3EEE4; color: #2E6B3E; }}
</style>
</head>
<body>
  <span class="mock-banner">quick mockup — parsed from source, not LLM-generated</span>
  <h1 class="mock-title">{html_lib.escape(repo)}</h1>
  <p class="mock-desc">{html_lib.escape(description)}</p>
  {body}
</body>
</html>"""
