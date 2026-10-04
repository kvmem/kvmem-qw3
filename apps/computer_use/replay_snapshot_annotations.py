"""Annotate an exported replay when both panes select the same attempt."""
import argparse
from pathlib import Path


def annotate(path):
    path = Path(path)
    text = path.read_text()
    if 'id="pairNotice"' in text:
        raise ValueError("Snapshot already annotated")
    if text.count("<script>") != 1 or text.count("function render(side)") != 1:
        raise ValueError("Unexpected replay template")
    text = text.replace("<script>", '<p class="note" id="pairNotice" role="status"></p>\n<script>')
    anchor = "function render(side)"
    function = """function pairNotice(){const a=chosen('left'),b=chosen('right');
 el('pairNotice').textContent=rows.length===1?'当前场景只有一次已终止记录，两侧显示同一次尝试，尚无新旧配对。':
 a&&b&&a.attempt===b.attempt?'两侧选择了同一次尝试，请选择不同记录进行对比。':''}
"""
    text = text.replace(anchor, function + anchor)
    lines = text.splitlines(keepends=True)
    matches = [i for i,line in enumerate(lines) if "el(side+'Identity').textContent=" in line]
    if len(matches) != 1:
        raise ValueError("Unexpected replay identity renderer")
    lines.insert(matches[0] + 1, " pairNotice();\n")
    path.write_text("".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("index", type=Path)
    annotate(p.parse_args().index)
