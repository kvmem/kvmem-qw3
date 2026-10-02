"""Create neutral local documents for P1 screen capture, without model inference.

Use the existing OSWorld test environment (docx, pptx, openpyxl, Pillow). These
files are inputs, not screenshots or reviewed answers. Capture and annotate the
actual Windows application state before constructing a diagnostic corpus.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build(output):
    from docx import Document
    from docx.shared import Pt
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.worksheet.views import Selection
    from pptx import Presentation
    from pptx.util import Inches, Pt as SlidePt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from PIL import Image, ImageDraw, ImageFont

    output.mkdir(parents=True, exist_ok=False)
    (output / "intake.txt").write_text(
        "DEVICE INTAKE\n\nBatch: A47-Blue\nWindow: 09:35 - 11:10\nOperator: Morgan Lee\nStatus: Pending review\n\n"
        "Bay 1: Keyboard\nBay 2: Display\nBay 3: Network adapter\n", encoding="utf-8")

    book = Workbook(); sheet = book.active; sheet.title = "Inventory"
    for row in [["Item", "Unit cost", "Quantity", "Status"], ["Cable", 18.75, 4, "Ready"],
                ["Adapter", 29.50, 2, "Waiting"], ["Stand", 42.25, 3, "Ready"],
                ["Hub", 37.80, 5, "Review"], ["Mouse", 23.60, 6, "Waiting"], ["Keyboard", 68.90, 1, "Ready"]]:
        sheet.append(row)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="244A76")
    for name in ("A", "B", "C", "D"): sheet.column_dimensions[name].width = 21
    for row in range(2, 8): sheet.cell(row, 2).number_format = "0.00"
    sheet.sheet_view.selection = [Selection(activeCell="B4", sqref="B4:D6")]
    book.save(output / "inventory.xlsx")

    doc = Document(); doc.styles["Normal"].font.name = "Arial"; doc.styles["Normal"].font.size = Pt(11)
    doc.add_heading("Team Atlas / Draft 07", 0)
    doc.add_paragraph("Review owner: Morgan Lee\nNext checkpoint: 17:40\nApproval: Pending")
    doc.add_heading("Inspection queue", 1)
    table = doc.add_table(rows=1, cols=3); table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, ("Location", "Record", "State")): cell.text = value
    for row in (("North", "Q17", "Ready"), ("East", "Q23", "Hold"), ("South", "Q41", "Review")):
        for cell, value in zip(table.add_row().cells, row): cell.text = value
    doc.add_paragraph("Keep this draft until the final checklist is reviewed.")
    doc.save(output / "review.docx")

    deck = Presentation(); deck.slide_width = Inches(10); deck.slide_height = Inches(5.625)
    first = deck.slides.add_slide(deck.slide_layouts[6])
    heading = first.shapes.add_textbox(Inches(.5), Inches(.25), Inches(9), Inches(.7))
    heading.text_frame.text = "REVIEW SCHEDULE / R4"
    heading.text_frame.paragraphs[0].font.size = SlidePt(28)
    for text, x, y, color in (("ALPHA\n17:40", 1, 1.5, (34, 86, 146)), ("BETA\n19:15", 5.5, 3, (211, 108, 35))):
        shape = first.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(3), Inches(1.2))
        shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor(*color)
        shape.text_frame.text = text
        for paragraph in shape.text_frame.paragraphs:
            paragraph.font.size = SlidePt(24); paragraph.font.color.rgb = RGBColor(255, 255, 255)
    second = deck.slides.add_slide(deck.slide_layouts[6])
    second.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1)).text_frame.text = "APPENDIX / HOLD"
    deck.save(output / "schedule.pptx")

    canvas = Image.new("RGB", (900, 600), "#f3f5f8"); draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 34)
    draw.text((55, 35), "PANEL P6", font=font, fill="#18283e")
    draw.ellipse((95, 150, 285, 340), fill="#235d9f")
    draw.rectangle((430, 310, 790, 495), fill="#de9233")
    draw.text((105, 365), "BLUE", font=font, fill="#18283e")
    draw.text((465, 510), "ORANGE", font=font, fill="#18283e")
    canvas.save(output / "panel.png")

    (output / "dashboard.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Atlas review desk</title><style>body{font:18px Arial;margin:40px;color:#163047;background:#f3f6fa}main{background:white;padding:28px;max-width:900px}h1{margin-top:0}label{display:block;margin:18px 0}input,select,button{font:inherit;padding:9px}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{border:1px solid #c5d0de;padding:12px;text-align:left}button{margin-right:18px}.badge{background:#fff0c2;padding:9px}</style>
<main><h1>Atlas review desk</h1><p class="badge">Awaiting approval</p>
<label>Account <input value="Atlas-Delta" aria-label="Account"></label>
<label>Priority <select aria-label="Priority"><option selected>Normal</option><option>High</option></select></label>
<table><tr><th>Record</th><th>Amount</th><th>Owner</th></tr><tr><td>R-17</td><td>19.75</td><td>Morgan Lee</td></tr><tr><td>R-23</td><td>46.20</td><td>Sam Rivera</td></tr></table>
<button>Save draft</button><button disabled>Publish</button></main></html>''', encoding="utf-8")

    manifest = {"kind": "P1_neutral_materials_only", "model_used": False, "screens_captured": False,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}}
    (output / "materials.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
