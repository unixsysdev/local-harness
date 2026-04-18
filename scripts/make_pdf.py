"""Convert README.md to README.pdf using Python markdown + weasyprint.

Produces a clean research-paper-style PDF with a small CSS for tables,
code blocks, and headings.
"""
from __future__ import annotations
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "README.md"
PDF_PATH = ROOT / "README.pdf"

CSS_STR = """
@page {
  size: A4;
  margin: 22mm 20mm 22mm 20mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: 'Source Sans Pro', Arial, sans-serif;
    font-size: 9pt;
    color: #666;
  }
}
body {
  font-family: 'Source Sans Pro', 'Helvetica Neue', Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #222;
}
h1 { font-size: 22pt; margin-top: 0; border-bottom: 2px solid #222; padding-bottom: 6pt; }
h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #bbb; padding-bottom: 3pt; }
h3 { font-size: 12pt; margin-top: 12pt; color: #333; }
h4 { font-size: 11pt; margin-top: 10pt; }
p, li { text-align: justify; }
hr { border: none; border-top: 1px solid #ccc; margin: 18pt 0; }
code {
  font-family: 'JetBrains Mono', 'Menlo', 'Consolas', monospace;
  font-size: 9pt;
  background: #f3f3f3;
  padding: 1px 4px;
  border-radius: 3px;
}
pre {
  background: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 8pt 10pt;
  overflow-x: auto;
  font-size: 8.5pt;
  line-height: 1.3;
}
pre code {
  background: none;
  padding: 0;
  font-size: inherit;
}
table {
  border-collapse: collapse;
  margin: 8pt 0;
  width: 100%;
  font-size: 9.5pt;
}
th, td {
  border: 1px solid #ccc;
  padding: 4pt 8pt;
  text-align: left;
  vertical-align: top;
}
th { background: #eee; font-weight: 600; }
blockquote {
  border-left: 3px solid #888;
  margin: 10pt 0;
  padding: 2pt 12pt;
  color: #444;
  background: #fafafa;
  font-style: italic;
}
a { color: #0a58ca; text-decoration: none; }
strong { color: #000; }
"""


def main() -> int:
    md_text = MD_PATH.read_text()
    html_body = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>local-harness — research notes</title></head>
<body>{html_body}</body>
</html>"""

    HTML(string=html_doc, base_url=str(ROOT)).write_pdf(
        str(PDF_PATH),
        stylesheets=[CSS(string=CSS_STR)],
    )
    print(f"wrote {PDF_PATH} ({PDF_PATH.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
