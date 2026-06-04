"""Convert reports/final_report.md to reports/final_report.pdf using xhtml2pdf."""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

import markdown
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from xhtml2pdf import pisa

REPO = Path(__file__).resolve().parents[1]
MD_PATH = REPO / "reports" / "final_report.md"
PDF_PATH = REPO / "reports" / "final_report.pdf"


def pisa_link_callback(uri: str, rel: str) -> str:
    """Resolve URIs in CSS (font, image) to local file paths.

    xhtml2pdf does not natively understand file:// URIs on Windows. This callback
    converts them, and also resolves bare absolute paths and relative paths.
    """
    if uri.startswith("file:///"):
        return uri[8:].replace("/", "\\")
    if uri.startswith("file://"):
        return uri[7:].replace("/", "\\")
    if re.match(r"^[a-zA-Z]:/", uri):
        return uri.replace("/", "\\")
    return uri


def register_unicode_fonts() -> None:
    """Register Times New Roman TTFs and reroute xhtml2pdf's `times` alias.

    Default ReportLab Times-Roman is a PostScript Type 1 font with only
    Latin-1 glyphs. Turkish characters (ğ, ş, ç, ı, ö, ü) render as boxes.
    We register the Windows TrueType files AND patch xhtml2pdf's internal
    DEFAULT_FONT map so any CSS `font-family: serif | Times-Roman | Times`
    resolves to the Unicode TTF.
    """
    fonts_dir = Path("C:/Windows/Fonts")
    candidates = {
        "TimesUni": "times.ttf",
        "TimesUni-Bold": "timesbd.ttf",
        "TimesUni-Italic": "timesi.ttf",
        "TimesUni-BoldItalic": "timesbi.ttf",
    }
    for name, fname in candidates.items():
        path = fonts_dir / fname
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
    addMapping("TimesUni", 0, 0, "TimesUni")
    addMapping("TimesUni", 1, 0, "TimesUni-Bold")
    addMapping("TimesUni", 0, 1, "TimesUni-Italic")
    addMapping("TimesUni", 1, 1, "TimesUni-BoldItalic")

    # Reroute xhtml2pdf's built-in font aliases. xhtml2pdf does NOT inspect
    # ReportLab's registered-font list when resolving CSS font-family; it uses
    # its own DEFAULT_FONT dict. Patching it here is the documented hook.
    from xhtml2pdf.default import DEFAULT_FONT

    DEFAULT_FONT["times"] = "TimesUni"
    DEFAULT_FONT["times-roman"] = "TimesUni"
    DEFAULT_FONT["times new roman"] = "TimesUni"
    DEFAULT_FONT["serif"] = "TimesUni"
    DEFAULT_FONT["timesuni"] = "TimesUni"


register_unicode_fonts()


def parse_yaml_meta(text: str) -> tuple[dict, str]:
    """Extract title/author/date/abstract from YAML front matter and return body."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    yaml_block = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []
    for line in yaml_block.split("\n"):
        if re.match(r"^\w[\w_]*:\s*\|", line):
            if current_key is not None:
                meta[current_key] = "\n".join(buffer).strip()
            current_key = line.split(":", 1)[0].strip()
            buffer = []
        elif re.match(r"^\w[\w_]*:\s*", line) and not line.startswith(" "):
            if current_key is not None:
                meta[current_key] = "\n".join(buffer).strip()
            k, v = line.split(":", 1)
            current_key = k.strip()
            v = v.strip().strip('"').strip("'")
            if v:
                meta[current_key] = v
                current_key = None
                buffer = []
            else:
                buffer = []
        else:
            if line.strip():
                buffer.append(line.strip())
    if current_key is not None:
        meta[current_key] = "\n".join(buffer).strip()
    return meta, body


def build_toc(html: str) -> str:
    """Build a simple Table of Contents from h1/h2 anchors."""
    pattern = re.compile(r"<h([12])>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
    items: list[tuple[int, str, str]] = []
    for match in pattern.finditer(html):
        level = int(match.group(1))
        title = re.sub(r"<.*?>", "", match.group(2)).strip()
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        items.append((level, title, slug))
    lines = ['<h2>Contents</h2><ul class="toc">']
    for level, title, slug in items:
        klass = f"toc-lvl{level}"
        lines.append(f'<li class="{klass}"><a href="#{slug}">{title}</a></li>')
    lines.append("</ul>")
    return "\n".join(lines)


def add_anchors(html: str) -> str:
    """Add id="slug" anchors to h1 and h2 tags."""

    def _replace(match: re.Match) -> str:
        level = match.group(1)
        title = match.group(2)
        plain = re.sub(r"<.*?>", "", title).strip()
        slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
        return f'<h{level} id="{slug}">{title}</h{level}>'

    return re.sub(r"<h([12])>(.*?)</h\1>", _replace, html, flags=re.IGNORECASE | re.DOTALL)


def mark_appendix_a_landscape(html: str) -> str:
    """Wrap Appendix A in a wide-table block and shorten long cell labels.

    The 12-column table cannot fit on A4 portrait at a readable font while
    keeping verbose names like "GPT-OSS 120B (reasoning=medium)" on one line.
    xhtml2pdf 0.2.17 does not honour named landscape pages reliably, so we keep
    portrait, shrink the font, and substitute compact aliases inside this block
    only. A short legend goes above the table.
    """
    aliases = {
        "GPT-OSS 120B (reasoning=medium)": "GPT-OSS-120B",
        "Qwen3 32B (reasoning=default)": "Qwen3-default",
        "Qwen3 32B (reasoning=none)": "Qwen3-none",
        "Llama 3.3 70B": "Llama-3.3-70B",
        "Llama 3.1 8B": "Llama-3.1-8B",
        "Naive Bayes (TF-IDF)": "NB",
        "LinearSVC (TF-IDF)": "SVM",
        "Random Forest (TF-IDF)": "RF",
        "XGBoost (TF-IDF)": "XGB",
        "RoBERTa-base (fine-tuned)": "RoBERTa",
        "employee_reviews": "ER",
        "fakenewsnet": "FNN",
        "classical_ml": "ML",
        "FS_CoT_RP_NA": "FS",
        "ZS_CoT": "ZS-CoT",
    }
    pattern = re.compile(
        r'(<h1 id="appendix-a[^"]*"[^>]*>.*?</h1>)(.*?)(?=<h1 id="appendix-b)',
        re.DOTALL,
    )
    legend = (
        '<p style="font-size: 8pt; color: #444; margin-top: 4pt;">'
        "Aliases used in this table only: "
        "NB = Naive Bayes (TF-IDF); SVM = LinearSVC (TF-IDF); "
        "RF = Random Forest (TF-IDF); XGB = XGBoost (TF-IDF); "
        "RoBERTa = RoBERTa-base (fine-tuned); "
        "ER = employee_reviews; FNN = fakenewsnet; ML = classical_ml; "
        "FS = FS_CoT_RP_NA; ZS-CoT = ZS_CoT."
        "</p>"
    )

    def _shorten(match: re.Match) -> str:
        head = match.group(1)
        body = match.group(2)
        for k, v in aliases.items():
            body = body.replace(k, v)
        return (
            '<div class="pagebreak"></div><div class="wide-table">'
            + head
            + legend
            + body
            + "</div>"
            + '<div class="pagebreak"></div>'
        )

    return pattern.sub(_shorten, html, count=1)


def rewrite_image_paths(html: str, base: Path) -> str:
    """Embed images as base64 data URIs so xhtml2pdf does not need to read them off disk."""

    def _replace(match: re.Match) -> str:
        path = match.group(1)
        if path.startswith(("http", "data:")):
            return match.group(0)
        abs_path = (base / path).resolve()
        if not abs_path.exists():
            return match.group(0)
        mime = mimetypes.guess_type(str(abs_path))[0] or "image/png"
        data = base64.b64encode(abs_path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', _replace, html)


def main() -> int:
    raw = MD_PATH.read_text(encoding="utf-8")
    meta, body = parse_yaml_meta(raw)
    title = meta.get("title", "Final Report")
    author = meta.get("author", "")
    date = meta.get("date", "")
    abstract = meta.get("abstract", "")

    md_engine = markdown.Markdown(extensions=["tables", "fenced_code"])
    body_html = md_engine.convert(body)
    body_html = add_anchors(body_html)
    body_html = mark_appendix_a_landscape(body_html)
    body_html = rewrite_image_paths(body_html, REPO / "reports")
    abstract_html = (
        f'<div class="abstract"><h2>Abstract</h2>{md_engine.convert(abstract)}</div>'
        if abstract
        else ""
    )
    toc_html = build_toc(body_html)

    css = """
    @page {
        size: A4 portrait;
        margin: 2.5cm 2.5cm 2.5cm 2.5cm;
        @frame footer {
            -pdf-frame-content: footer_content;
            left: 50pt; width: 500pt; top: 800pt; height: 30pt;
        }
    }
    body, p, h1, h2, h3, h4, h5, h6, td, th, li, a, span, div {
        font-family: "TimesUni", "Times New Roman", serif;
    }
    body {
        font-size: 11pt;
        line-height: 1.45;
        color: #111;
    }
    h1.title {
        font-size: 22pt;
        margin-top: 200pt;
        text-align: center;
        line-height: 1.25;
    }
    .author, .date, .course {
        text-align: center;
        font-size: 12pt;
        margin-top: 14pt;
    }
    .pagebreak { page-break-after: always; }
    h1 {
        font-size: 16pt;
        margin-top: 18pt;
        margin-bottom: 6pt;
        border-bottom: 1px solid #aaa;
        padding-bottom: 2px;
    }
    h2 { font-size: 13pt; margin-top: 14pt; }
    h3 { font-size: 12pt; margin-top: 10pt; }
    p { margin: 4pt 0; text-align: justify; }
    table {
        border-collapse: collapse;
        margin: 6pt 0;
        font-size: 9pt;
        width: 100%;
    }
    th, td {
        border: 1px solid #999;
        padding: 2pt 4pt;
    }
    th { background-color: #e8e8e8; }
    /* Appendix A's 12-column table needs aggressive shrinkage. All long model
       names are hyphenated (no internal whitespace) so they cannot wrap, and
       nowrap forces each cell onto one line. */
    .wide-table table { font-size: 5pt; }
    .wide-table th, .wide-table td {
        padding: 1pt 2pt;
        font-size: 5pt;
        white-space: nowrap;
    }
    .wide-table h1 { font-size: 14pt; margin-top: 6pt; }
    code, pre {
        font-family: "Courier", "Courier New", monospace;
        font-size: 9.5pt;
        background-color: #f3f3f3;
    }
    pre { padding: 6pt; }
    img { max-width: 16cm; }
    .abstract {
        margin-top: 10pt;
        font-size: 10.5pt;
        background-color: #f7f7f7;
        padding: 8pt 12pt;
        border-left: 2pt solid #999;
    }
    .abstract h2 { margin-top: 0; font-size: 12pt; }
    ul.toc { list-style: none; padding-left: 0; }
    .toc-lvl1 { font-weight: bold; margin-top: 4pt; }
    .toc-lvl2 { margin-left: 16pt; }
    a { color: #111; text-decoration: none; }
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>

<div id="footer_content" style="text-align:center; font-size:9pt; color:#555;">
<pdf:pagenumber />
</div>

<h1 class="title">{title}</h1>
<p class="author">{author}</p>
<p class="date">{date}</p>
<p class="course">Course: Data Mining &mdash; Final Project</p>

<div class="pagebreak"></div>

{abstract_html}

<div class="pagebreak"></div>

{toc_html}

<div class="pagebreak"></div>

{body_html}

</body>
</html>
"""

    tmp_path = PDF_PATH.with_suffix(".pdf.tmp")
    with tmp_path.open("wb") as fh:
        result = pisa.CreatePDF(
            html,
            dest=fh,
            encoding="utf-8",
            link_callback=pisa_link_callback,
        )
    if result.err:
        print(f"ERROR: xhtml2pdf reported {result.err} errors", file=sys.stderr)
        return 1
    try:
        tmp_path.replace(PDF_PATH)
    except PermissionError:
        print(
            f"WARN: {PDF_PATH.name} is locked (open in a viewer?). "
            f"Wrote {tmp_path.name} instead; rename manually.",
            file=sys.stderr,
        )
        size = tmp_path.stat().st_size
        print(f"Wrote {tmp_path} ({size:,} bytes)")
        return 0
    size = PDF_PATH.stat().st_size
    print(f"Wrote {PDF_PATH} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
