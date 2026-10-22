"""Build the report and executive summary as Word documents and PDFs from their Markdown sources.

python3 build_docx.py            -> report.docx/.pdf, executive-summary.docx/.pdf

Handles the Markdown subset the sources use: YAML front matter (cover page), # and ## headings, paragraphs,
- and 1. lists, pipe tables, images with captions (optional {width=12cm}), **bold**, *italic* and `code`.
Format follows the CDE "Guidelines for Preparation of UREx Report": A4, 2.5 cm margins, Times New Roman 12 pt,
1.5 line spacing, justified text, the sample cover page, abstract, preface and acknowledgements and table of
contents before the main report, main report numbered from 1, main headings bold upper case, appendices single
spaced in a smaller font. The table of contents needs page numbers, so the report is built twice: once to lay it
out (LibreOffice, soffice), and again with the page numbers read from that layout.
"""
import os
import re
import subprocess
import tempfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Times New Roman"
BODY = 12
APPX = 10  # appendices: smaller font, single spacing


def set_font(run, size=None, bold=None, italic=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_inline(par, text, size=BODY, base_italic=False):
    """Add text with **bold**, *italic* and `code` spans."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = par.add_run(tok[2:-2]); set_font(r, size, bold=True, italic=base_italic)
        elif tok.startswith("*"):
            r = par.add_run(tok[1:-1]); set_font(r, size, italic=not base_italic)
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1]); set_font(r, size - 1, italic=base_italic); r.font.name = "Consolas"
        else:
            r = par.add_run(tok); set_font(r, size, italic=base_italic)


def page_number_footer(section):
    p = section.footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(); set_font(r, 10)
    for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        r._r.append(el)


def page_numbering(section, fmt, start=None):
    pg = section._sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType"); section._sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))


def setup(doc, full):
    st = doc.styles["Normal"]; st.font.name = FONT; st.font.size = Pt(BODY if full else 11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format; pf.line_spacing = 1.5 if full else 1.15; pf.space_after = Pt(6)
    for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Title", 18)):
        h = doc.styles[name]; h.font.name = FONT; h.font.size = Pt(size); h.font.bold = True
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.element.rPr.rFonts.set(qn("w:eastAsia"), FONT); h.element.rPr.rFonts.set(qn("w:ascii"), FONT)
        h.element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
        h.paragraph_format.space_before = Pt(10 if name == "Heading 2" else 12); h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.line_spacing = 1.15
    if full:
        doc.styles["Heading 1"].font.all_caps = True  # "A main heading should be in bold, upper case"
    s = doc.sections[0]
    s.page_width, s.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(s, side, Cm(2.5))
    page_number_footer(s)
    if full:  # front matter in roman numerals, no number on the cover
        s.different_first_page_header_footer = True
        page_numbering(s, "lowerRoman", 1)


def new_section(doc, landscape, restart=False):
    from docx.enum.section import WD_ORIENT
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    sec.orientation = WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    w, h = (Cm(29.7), Cm(21.0)) if landscape else (Cm(21.0), Cm(29.7))
    sec.page_width, sec.page_height = w, h
    sec.different_first_page_header_footer = False
    pg = sec._sectPr.find(qn("w:pgNumType"))
    if restart:
        page_numbering(sec, "decimal", 1)
    elif pg is not None and pg.get(qn("w:start")) is not None:  # add_section copies the previous sectPr
        del pg.attrib[qn("w:start")]
    return sec


def front_matter(md):
    m = re.match(r"---\n(.*?)\n---\n", md, re.S)
    if not m:
        return {}, md
    meta = dict(re.findall(r'^(\w+):\s*"?(.*?)"?\s*$', m.group(1), re.M))
    return meta, md[m.end():]


def centred(doc, text, size, bold=False, before=0, italic=False):
    q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    q.paragraph_format.space_before = Pt(before); q.paragraph_format.line_spacing = 1.15
    r = q.add_run(text); set_font(r, size, bold=bold, italic=italic)


def title_page(doc, meta, full):
    if not full:
        centred(doc, meta.get("title", ""), 15, bold=True)
        if meta.get("author"):
            centred(doc, meta["author"], 11, italic=True)
        return
    # the sample cover page in the CDE guidelines
    centred(doc, "UNDERGRADUATE RESEARCH EXPERIENCE", 16, bold=True, before=60)
    centred(doc, "(UREx)", 16, bold=True)
    centred(doc, "PROJECT REPORT", 16, bold=True, before=60)
    centred(doc, meta.get("title", ""), 16, bold=True, before=60)
    centred(doc, f'{meta.get("name", "")} ({meta.get("student_number", "")})', 12, before=72)
    centred(doc, "Supervised by:", 12, before=48)
    centred(doc, meta.get("supervisor", ""), 12)
    centred(doc, meta.get("department", ""), 12, before=120)
    centred(doc, "National University of Singapore", 12)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_); tcPr.append(sh)


def add_table(doc, lines, width_cm=16.0):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|[\s\-|:]+\|$", l.strip())]
    n = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=n); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    small = 7.5 if n > 9 else (8 if n > 7 else 9.5)
    if n > 2:  # full width with a broader first column, so row labels wrap less
        t.autofit = False
        longest = max(len(r[0].replace("*", "")) for r in rows)  # first column sized to its labels
        f = min(5.0, max(2.0, 0.17 * longest)) if n <= 7 else 3.6
        first = Cm(f); rest = Cm((width_cm - f) / (n - 1))
        for row in t.rows:
            for j, cell in enumerate(row.cells):
                cell.width = first if j == 0 else rest
        for j, col in enumerate(t.columns):
            col.width = first if j == 0 else rest
    for i, r in enumerate(rows):
        for j in range(n):
            cell = t.cell(i, j); cell.text = ""
            p = cell.paragraphs[0]; p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(0)
            txt = r[j] if j < len(r) else ""
            if i == 0:
                rr = p.add_run(txt.replace("**", "")); set_font(rr, small, bold=True); shade(cell, "E8E8E8")
            else:
                add_inline(p, txt, small)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_image(doc, path, caption, width_cm=16.0):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True; p.paragraph_format.line_spacing = 1.0
    if os.path.exists(os.path.join(HERE, path)):
        p.add_run().add_picture(os.path.join(HERE, path), width=Cm(width_cm))
    else:
        r = p.add_run(f"[missing figure: {path}]"); set_font(r, 11, bold=True)
    c = doc.add_paragraph(); c.paragraph_format.line_spacing = 1.15
    m = re.match(r"(Figure [A-Z]?\d+\.)(.*)", caption)
    if m:
        r = c.add_run(m.group(1)); set_font(r, 10, bold=True); add_inline(c, m.group(2), 10)
    else:
        add_inline(c, caption, 10)


def toc_entries(body):
    """Headings from the Introduction on: every # heading and the ## headings of the main report."""
    out, started, appx = [], False, False
    for l in body.split("\n"):
        if l.startswith("# "):
            t = l[2:].strip(); started = started or t.startswith("1. ")
            appx = appx or t == "References"
            if started:
                out.append((1, t))
        elif l.startswith("## ") and started and not appx:
            out.append((2, l[3:].strip()))
    return out


def add_toc(doc, entries, pages):
    doc.add_heading("Table of Contents", level=1)
    for lvl, t in entries:
        p = doc.add_paragraph(); pf = p.paragraph_format
        pf.line_spacing = 1.1; pf.space_after = Pt(1); pf.space_before = Pt(0 if lvl == 2 else 3)
        pf.left_indent = Cm(0 if lvl == 1 else 0.9)
        pf.tab_stops.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        r = p.add_run(f"{t}\t{pages.get(t, '')}"); set_font(r, 11, bold=(lvl == 1))


def convert(md_name, docx_name, full, pages=None):
    md = open(os.path.join(HERE, md_name), encoding="utf-8").read()
    meta, body = front_matter(md)
    doc = Document(); setup(doc, full); title_page(doc, meta, full)
    lines = body.split("\n"); i = 0; para = []
    st = {"size": BODY if full else 11, "appx": False}  # the executive summary has no set format; 1 to 2 pages

    def flush():
        if para:
            text = " ".join(s.strip() for s in para)
            p = doc.add_paragraph()
            if st["appx"]:
                p.paragraph_format.line_spacing = 1.0
            if re.match(r"^\*(Table [A-Z]?\d+\..*)\*$", text):  # table caption
                p.paragraph_format.keep_with_next = True; p.paragraph_format.line_spacing = 1.15
                m = re.match(r"^\*(Table [A-Z]?\d+\.)(.*)\*$", text)
                r = p.add_run(m.group(1)); set_font(r, 10, bold=True); add_inline(p, m.group(2), 10)
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                add_inline(p, text, st["size"])
            para.clear()

    in_refs = False; first_h1 = True
    while i < len(lines):
        l = lines[i]
        if l.startswith("# "):
            flush(); title = l[2:].strip()
            if full and title.startswith("1. "):
                doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
                add_toc(doc, toc_entries(body), pages or {})
                new_section(doc, landscape=False, restart=True)
            elif full and title.startswith("Appendix A"):
                new_section(doc, landscape=True); st["appx"] = True; st["size"] = APPX
            elif full and title.startswith("Appendix B"):
                new_section(doc, landscape=False)
            brk = full and not first_h1 and (title in ("References", "Preface and Acknowledgements")
                                             or re.match(r"Appendix [C-Z]\.", title) is not None)
            first_h1 = False
            h = doc.add_heading(title, level=1); in_refs = title == "References"
            h.paragraph_format.page_break_before = bool(brk)  # not a break paragraph, which can leave a blank page
        elif l.startswith("## "):
            flush(); doc.add_heading(l[3:].strip(), level=2)
        elif l.startswith("!["):
            flush(); m = re.match(r"!\[(.*)\]\((.*?)\)(\{width=([\d.]+)cm\})?", l)  # optional {width=12cm}
            add_image(doc, m.group(2), m.group(1), float(m.group(4) or 16.0))
        elif l.startswith("|"):
            flush(); tl = []
            while i < len(lines) and lines[i].startswith("|"):
                tl.append(lines[i]); i += 1
            add_table(doc, tl, 24.7 if (st["appx"] and doc.sections[-1].page_width > Cm(25)) else 16.0); continue
        elif re.match(r"^(\s*)(- |\d+\. )", l):
            flush(); m = re.match(r"^(\s*)(- |\d+\. )(.*)", l)
            item = [m.group(3)]; i += 1
            while i < len(lines) and lines[i].startswith("  ") and not re.match(r"^\s*(- |\d+\. )", lines[i]):
                item.append(lines[i].strip()); i += 1
            if m.group(2) == "- ":
                p = doc.add_paragraph(style="List Bullet")
            else:  # numbered items as plain paragraphs, so numbering restarts in every list
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.6)
                r = p.add_run(m.group(2)); set_font(r, st["size"])
            p.paragraph_format.space_after = Pt(2)
            if st["appx"]:
                p.paragraph_format.line_spacing = 1.0
            add_inline(p, " ".join(item), st["size"]); continue
        elif not l.strip():
            flush()
        else:
            para.append(l)
            if in_refs:  # references: one paragraph each, hanging indent
                flush(); p = doc.paragraphs[-1]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.left_indent = Cm(1.27); p.paragraph_format.first_line_indent = Cm(-1.27)
                p.paragraph_format.line_spacing = 1.15
        i += 1
    flush()
    doc.save(os.path.join(HERE, docx_name)); return toc_entries(body)


def to_pdf(docx_name, outdir):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", outdir, os.path.join(HERE, docx_name)],
                   check=True, capture_output=True)
    return os.path.join(outdir, docx_name.replace(".docx", ".pdf"))


def heading_pages(pdf, entries):
    """Printed page number of each heading: main report pages count from the page holding '1. Introduction'."""
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout.split("\f")
    found = {}
    for lvl, t in entries:
        key = re.sub(r"\s+", " ", t.lower())[:35]
        for k, pg in enumerate(text):
            if any(re.sub(r"\s+", " ", ln.strip().lower()).startswith(key) and "...." not in ln for ln in pg.split("\n")):
                found[t] = k; break
    start = found.get(entries[0][1], 0)
    return {t: str(k - start + 1) for t, k in found.items()}, start, len(text)


def main_report_pages(pdf, start):
    """Pages from the Introduction up to, not including, the References."""
    text = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout.split("\f")
    for k in range(start, len(text)):
        if re.search(r"^\s*REFERENCES\s*$", text[k], re.M):
            return k - start
    return None


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    entries = convert("report.md", "report.docx", full=True)
    pages, start, _ = heading_pages(to_pdf("report.docx", tmp), entries)
    convert("report.md", "report.docx", full=True, pages=pages)
    pdf = to_pdf("report.docx", HERE)
    pages2, start2, total = heading_pages(pdf, entries)
    if pages2 != pages:  # the numbers are the same width, so one more pass settles any shift
        convert("report.md", "report.docx", full=True, pages=pages2); pdf = to_pdf("report.docx", HERE)
    print(f"wrote report.docx and report.pdf: {total} pages, main report {main_report_pages(pdf, start2)} pages, "
          f"{os.path.getsize(pdf) / 1e6:.1f} MB")
    convert("executive-summary.md", "executive-summary.docx", full=False)
    es = to_pdf("executive-summary.docx", HERE)
    n = len(subprocess.run(["pdftotext", es, "-"], capture_output=True, text=True).stdout.split("\f")) - 1
    print(f"wrote executive-summary.docx and .pdf: {n} pages, {os.path.getsize(es) / 1e6:.2f} MB")
