"""Opt-in, read-only partial Office controls using the official AT-SPI reader.

This candidate leaves the installed OSWorld server and input driver unchanged.
It intentionally omits table cells and invisible menus; positive control facts
are useful, but this partial tree cannot prove absence or workbook correctness.
"""
from qw3_computer.linux_driver import foreground_tree_script, controls_from_xml


def chrome_tree_script(title):
    source = foreground_tree_script(title)
    injection = '''original=a.create_atspi_node
counts={"calls":0,"pruned_tables":0}
def bounded(node,depth=0,flag=None):
 counts["calls"]+=1
 if counts["calls"]>1024:
  raise RuntimeError("Accessible node budget reached")
 role=node.getRoleName().strip()
 if role in ("menu","menu item","check menu item","radio menu item","separator") and not node.getState().contains(a.pyatspi.STATE_SHOWING):
  return a.lxml.etree.Element("omitted-invisible-menu")
 if role in ("table","table cell"):
  counts["pruned_tables"]+=1
  depth=a.MAX_DEPTH
 return original(node,depth,flag)
a.MAX_WIDTH=80
a.MAX_DEPTH=16
a.create_atspi_node=bounded
'''
    marker = 'a.libreoffice_version_tuple=a.get_libreoffice_version()'
    return source.replace(marker, marker + '\n ' + injection.replace('\n', '\n ').rstrip())


def partial_controls(xml, foreground, width, height):
    controls = controls_from_xml(xml, foreground, width, height)
    return controls | {'truncated': True, 'scope': 'foreground_office_chrome',
                       'provider': 'official_atspi_with_bounded_read_only_pruning',
                       'excluded_subtrees': ['table_cells', 'invisible_menus'],
                       'completion_scope': 'Positive observed controls only; no absence or workbook correctness proof.'}
