#!/usr/bin/env python3
"""Convert SYSTEM_TECHNICAL_REPORT.md into a formatted Word document."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "SYSTEM_TECHNICAL_REPORT.md"
DOCX_PATH = ROOT / "Pose系統與技術報告.docx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x1F, 0x6F, 0x6A)
GRAY = RGBColor(0x44, 0x44, 0x44)
HEADER_BG = "1F3A5F"
ALT_ROW = "F3F6F8"
CODE_BG = "F4F4F4"


def set_run_font(run, name="Calibri", east="微軟正黑體", size=11, bold=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def shade_cell(cell, hex_color: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def set_cell_border(cell) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "C5CDD6")
        tcBorders.append(el)
    tcPr.append(tcBorders)


def add_formatted_runs(paragraph, text: str, size=11, color=GRAY) -> None:
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_run_font(run, name="Consolas", east="微軟正黑體", size=size - 1, color=RGBColor(0x9A, 0x34, 0x16))
        elif part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_run_font(run, size=size, bold=True, color=color)
        else:
            run = paragraph.add_run(part)
            set_run_font(run, size=size, color=color)


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    if level == 1:
        set_run_font(run, size=16, bold=True, color=NAVY)
    else:
        set_run_font(run, size=13, bold=True, color=TEAL)


def add_body(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    add_formatted_runs(p, text)


def add_list_item(doc: Document, text: str, ordered: bool, index: int | None = None) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.space_after = Pt(4)
    prefix = f"{index}. " if ordered else "• "
    run = p.add_run(prefix)
    set_run_font(run, size=11, bold=ordered, color=NAVY)
    add_formatted_runs(p, text)


def add_code_block(doc: Document, lines: list[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    shade_cell(cell, CODE_BG)
    set_cell_border(cell)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("\n".join(lines))
    set_run_font(run, name="Consolas", east="Consolas", size=9, color=RGBColor(0x22, 0x22, 0x22))
    doc.add_paragraph()


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for r_i, row in enumerate(rows):
        for c_i, value in enumerate(row):
            cell = table.cell(r_i, c_i)
            cell.text = ""
            p = cell.paragraphs[0]
            add_formatted_runs(
                p,
                value,
                size=10,
                color=RGBColor(0xFF, 0xFF, 0xFF) if r_i == 0 else GRAY,
            )
            if r_i == 0:
                shade_cell(cell, HEADER_BG)
                for run in p.runs:
                    run.bold = True
                    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            elif r_i % 2 == 0:
                shade_cell(cell, ALT_ROW)
            set_cell_border(cell)
    doc.add_paragraph()


def parse_table(block: list[str]) -> list[list[str]]:
    rows = []
    for line in block:
        if re.match(r"^\|?\s*-{3,}", line.replace("|", " | ")):
            continue
        if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def add_cover(doc: Document) -> None:
    for _ in range(3):
        doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Pose 姿勢偵測系統")
    set_run_font(run, size=28, bold=True, color=NAVY)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("系統報告　／　技術報告")
    set_run_font(run, size=18, bold=True, color=TEAL)

    info = [
        "平台：iOS 17+（SwiftUI）＋ FastAPI 後端",
        "骨架：QuickPose SDK（MediaPipe BlazePose Full）",
        "資料：本機 Realm　／　雲端 MongoDB Atlas",
        "版本：cursor/add-runpose-backend-dd14",
    ]
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(24)
    for line in info:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        set_run_font(run, size=12, color=GRAY)

    doc.add_page_break()


def convert() -> Path:
    markdown = MD_PATH.read_text(encoding="utf-8")
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)

    add_cover(doc)

    lines = markdown.splitlines()
    i = 0
    # skip the first H1; cover already has the title
    if lines and lines[0].startswith("# "):
        i = 1

    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip() == "---":
            i += 1
            continue

        if line.startswith("## "):
            add_heading(doc, line[3:].strip(), 1)
            i += 1
            continue
        if line.startswith("### "):
            add_heading(doc, line[4:].strip(), 2)
            i += 1
            continue

        if line.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            add_code_block(doc, block)
            continue

        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, parse_table(block))
            continue

        numbered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if numbered:
            add_list_item(doc, numbered.group(2).rstrip(), ordered=True, index=int(numbered.group(1)))
            i += 1
            continue
        if line.startswith("- "):
            add_list_item(doc, line[2:].rstrip(), ordered=False)
            i += 1
            continue

        add_body(doc, line.strip())
        i += 1

    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("Pose 姿勢偵測系統 — 系統與技術報告")
    set_run_font(run, size=9, color=RGBColor(0x88, 0x88, 0x88))

    DOCX_PATH.write_bytes(b"")  # ensure parent exists / truncate
    doc.save(str(DOCX_PATH))
    return DOCX_PATH


if __name__ == "__main__":
    path = convert()
    print(path)
    print(path.stat().st_size)
