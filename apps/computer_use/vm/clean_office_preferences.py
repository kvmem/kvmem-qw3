"""Remove only neutral qualification history, retaining XML QName namespaces.

Host administration only; run with the qualified evaluator Python. Never deploy
this code into Computer MCP or remove an installed application profile.
"""
from pathlib import Path
from lxml import etree


def clean(source, destination):
    source = Path(source); destination = Path(destination)
    tree = etree.parse(str(source))
    root = tree.getroot()
    namespace_map = dict(root.nsmap)
    # xs is used in attribute values (oor:type="xs:boolean"), so ordinary
    # ElementTree serialization can silently omit this necessary declaration.
    if namespace_map.get('xs') != 'http://www.w3.org/2001/XMLSchema':
        raise ValueError('Office QName namespace is not declared')
    registry = '{http://openoffice.org/2001/registry}'
    neutral = 'file:///C:/Qw3Computer/OfficeHandlerQualification/'
    removed = 0
    for item in list(root):
        references = [value for node in item.iter() for value in [node.get(registry + 'name'), node.text]
                      if isinstance(value, str) and value.startswith('file:///')]
        if not any(reference.startswith(neutral) for reference in references):
            continue
        if ('PickList' not in item.get(registry + 'path', '')
                or any(not reference.startswith(neutral) for reference in references)):
            raise ValueError('Refusing to remove mixed or non-history preferences')
        root.remove(item); removed += 1
    encoded = etree.tostring(tree, encoding='UTF-8', xml_declaration=True)
    if etree.fromstring(encoded).nsmap != namespace_map or neutral.encode() in encoded:
        raise ValueError('Office preference cleanup lost namespace or history integrity')
    with destination.open('xb') as handle:
        handle.write(encoded)
    return {'removed_neutral_history_items': removed, 'qname_namespace_map_preserved': True}
