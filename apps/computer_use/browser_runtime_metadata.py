"""Native Chrome resource metadata is distinct from task text.

Names, values and completion conditions are retained verbatim. This helper only
provides an additional revalidation proof; the caller must still prove exact
whole-window pixels, device identity and target bounds before sending input.
"""
import json
import re


PROVIDER = 'official-osworld-create_atspi_node+native-browser-runtime'
RESOURCE = r'(?:0|[1-9][0-9]{0,5})(?:\.[0-9]{1,3})? (?:KB|MB|GB)'
PATH = r'/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+'
FIELDS = {'provider', 'browser_version', 'bus_name', 'object_path', 'window_path',
          'owner_pid', 'owner_uid', 'owner_start_ticks', 'foreground_handle',
          'frame_title', 'page_title', 'selected', 'visible', 'showing', 'enabled',
          'native_class', 'parent_roles', 'resource', 'focus_objects'}


def valid_metadata(control, foreground):
    m = control.get('browser_runtime_metadata')
    if not isinstance(m, dict) or set(m) != FIELDS:
        return None
    if (m['provider'] != 'native-atspi-chrome-selected-tab'
            or not re.fullmatch(r'Google Chrome 130\.[0-9]+\.[0-9]+\.[0-9]+', str(m['browser_version']))
            or control.get('kind') != 'page-tab' or m['native_class'] != 'Tab'
            or any(m[k] is not True for k in ('selected', 'visible', 'showing', 'enabled'))
            or foreground.get('process') != 'Google-chrome'
            or m['foreground_handle'] != foreground.get('handle')
            or m['frame_title'] != foreground.get('title')):
        return None
    title = m['page_title']
    if (not isinstance(title, str) or not title or len(title) > 1000
            or m['frame_title'] != title + ' - Google Chrome'
            or not isinstance(m['resource'], str) or not re.fullmatch(RESOURCE, m['resource'])
            or control.get('name') != title + ' - Memory usage - ' + m['resource']):
        return None
    if (not isinstance(m['bus_name'], str) or not re.fullmatch(r':\d+\.\d+', m['bus_name'])
            or any(not isinstance(m[k], str) or not re.fullmatch(PATH, m[k])
                   for k in ('object_path', 'window_path'))
            or any(type(m[k]) is not int for k in ('owner_pid', 'owner_uid', 'owner_start_ticks'))
            or not 0 < m['owner_pid'] < 2**31 or m['owner_uid'] < 0 or m['owner_start_ticks'] <= 0):
        return None
    parents = m['parent_roles']
    if (not isinstance(parents, list) or not 2 <= len(parents) <= 16 or parents[0] != 'frame'
            or parents.count('frame') != 1 or parents.count('page tab list') != 1
            or any(not isinstance(role, str) or role not in {'frame', 'panel', 'page tab list'} for role in parents)):
        return None
    focus = m['focus_objects']
    if (not isinstance(focus, list) or len(focus) > 1
            or any(not isinstance(item, dict) or set(item) != {'object_path', 'role'}
                   or not isinstance(item['object_path'], str) or not re.fullmatch(PATH, item['object_path'])
                   or not isinstance(item['role'], str) or not item['role'] for item in focus)):
        return None
    return m


def metadata_from_attrs(attributes, control):
    """Decode only the extension supplied by the native reader, not HTML names."""
    try:
        value = json.loads(attributes.get('qw3-browser-runtime', 'null'))
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    return valid_metadata(control | {'browser_runtime_metadata': value},
                          {'process': 'Google-chrome', 'handle': value.get('foreground_handle'),
                           'title': value.get('frame_title')})


def runtime_only_changes(old, fresh):
    """Return positive source evidence only when every other control is equal.

    No exception is inferred for old observations, missing provenance, other
    platforms, task/field text, multiple selected tabs, focus or layout changes.
    """
    if any(o.get('capabilities', {}).get('platform') != 'linux' for o in (old, fresh)):
        return None
    identity = ('desktop_id', 'device_id', 'session_id', 'display_id', 'width', 'height', 'rotation')
    if (any(old.get(k) is None or old.get(k) != fresh.get(k) for k in identity)
            or any(not isinstance(old.get(k), str) or not old[k] for k in identity[:4])
            or any(type(old.get(k)) is not int or old[k] <= 0 for k in ('width', 'height'))
            or type(old.get('rotation')) is not int or old['rotation'] not in {0, 90, 180, 270}
            or not isinstance(old.get('foreground'), dict) or old['foreground'] != fresh.get('foreground')):
        return None
    first, second = (o.get('accessibility', {}) for o in (old, fresh))
    if (any(a.get('status') != 'ok' or a.get('provider') != PROVIDER for a in (first, second))
            or first.get('truncated') != second.get('truncated')
            or any(not isinstance(a.get('controls'), list) for a in (first, second))
            or len(first['controls']) != len(second['controls'])):
        return None
    # Both complete source reads must carry exactly one selected native tab.
    admitted = [[c for c in a['controls'] if isinstance(c, dict)
                 and valid_metadata(c, old['foreground']) is not None] for a in (first, second)]
    if any(len(items) != 1 for items in admitted):
        return None
    changes = []
    for left, right in zip(first['controls'], second['controls']):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return None
        l = {k: v for k, v in left.items() if k != 'control_id'}
        r = {k: v for k, v in right.items() if k != 'control_id'}
        if l == r:
            continue
        lm, rm = valid_metadata(left, old['foreground']), valid_metadata(right, fresh['foreground'])
        if lm is None or rm is None:
            return None
        if ({k: v for k, v in lm.items() if k != 'resource'} !=
                {k: v for k, v in rm.items() if k != 'resource'}):
            return None
        if ({k: v for k, v in l.items() if k not in {'name', 'browser_runtime_metadata'}} !=
                {k: v for k, v in r.items() if k not in {'name', 'browser_runtime_metadata'}}):
            return None
        if lm['resource'] == rm['resource']:
            return None
        changes.append({'old_raw_name': left['name'], 'fresh_raw_name': right['name'],
                        'source_identity': {k: v for k, v in lm.items() if k != 'resource'}})
    if len(changes) != 1:
        return None
    return {'basis': 'same_native_selected_chrome_tab_runtime_resource_only',
            'changes': changes, 'positive_focus_object_unchanged': bool(admitted[0][0]['browser_runtime_metadata']['focus_objects']),
            'raw_labels_and_completion_conditions_preserved': True}


def native_browser_tree_script(source, foreground):
    """Extend the official AT-SPI reader in process, without editing the guest.

    Native object ancestry, X11 window PID and executable generation establish
    source. Web documents are pruned only in the separate chrome-source walk;
    the official task reader still traverses its original full foreground tree.
    """
    anchor = " xml=a.lxml.etree.tostring(a.create_atspi_node(frames[0]),encoding='unicode')"
    if source.count(anchor) != 1:
        raise ValueError('Qualified official traversal anchor changed')
    injection = '''
 browser_runtime=None;browser_tab_path=None;browser_focus=[]
 browser_frame=frames[0]
 try:
  import os,re,subprocess
  from pathlib import Path
  expected=FOREGROUND_LITERAL
  assert expected['process']=='Google-chrome' and expected['title'].endswith(' - Google Chrome')
  current=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)
  assert re.search(r'0x[0-9a-fA-F]+',current).group()==expected['handle']
  window_pid=int(subprocess.check_output(['xprop','-id',expected['handle'],'_NET_WM_PID'],text=True).split(' = ',1)[1])
  assert browser_frame.get_process_id()==window_pid
  proc=Path('/proc')/str(window_pid)
  assert os.readlink(proc/'exe')=='/opt/google/chrome/chrome'
  native_bus=browser_frame.app.bus_name;native_window=browser_frame.path
  selected=[];visits=0
  def chrome_source(node,parents):
   global visits
   visits+=1
   if visits>512 or len(parents)>16:raise RuntimeError('Browser chrome source bounds exceeded')
   role=node.getRoleName();state=node.getState()
   if role=='page tab' and state.contains(a.pyatspi.STATE_SELECTED):
    assert node.app.bus_name==native_bus and node.get_process_id()==window_pid
    chain=[];ancestor=node.parent
    while ancestor is not None and len(chain)<=16:
     chain.append(ancestor.getRoleName())
     if ancestor.path==native_window and ancestor.app.bus_name==native_bus:break
     ancestor=ancestor.parent
    else:raise RuntimeError('Selected tab has no bounded native window ancestry')
    assert chain[-1]=='frame' and chain.count('page tab list')==1
    assert all(r in ('frame','panel','page tab list') for r in chain)
    assert node.get_attributes().get('class')=='Tab'
    selected.append((node,list(reversed(chain))))
   if role.startswith('document') or role in ('table','table cell'):return
   for child in node:chrome_source(child,parents+[role])
  chrome_source(browser_frame,[]);assert len(selected)==1
  tab,parents=selected[0];title=expected['title'][:-len(' - Google Chrome')]
  assert tab.name.startswith(title+' - Memory usage - ')
  resource=tab.name[len(title+' - Memory usage - '):]
  assert re.fullmatch(RESOURCE_LITERAL,resource)
  version=subprocess.run(['google-chrome','--version'],capture_output=True,text=True,timeout=2).stdout.strip()
  assert re.fullmatch(r'Google Chrome 130\\.[0-9]+\\.[0-9]+\\.[0-9]+',version)
  state=tab.getState();browser_tab_path=tab.path
  browser_runtime=dict(provider='native-atspi-chrome-selected-tab',browser_version=version,bus_name=native_bus,
   object_path=tab.path,window_path=native_window,owner_pid=window_pid,owner_uid=proc.stat().st_uid,
   owner_start_ticks=int((proc/'stat').read_text().rpartition(')')[2].split()[19]),
   foreground_handle=expected['handle'],frame_title=expected['title'],page_title=title,
   selected=state.contains(a.pyatspi.STATE_SELECTED),visible=state.contains(a.pyatspi.STATE_VISIBLE),
   showing=state.contains(a.pyatspi.STATE_SHOWING),enabled=state.contains(a.pyatspi.STATE_ENABLED),
   native_class='Tab',parent_roles=parents,resource=resource,focus_objects=[])
 except Exception:
  browser_runtime=None;browser_tab_path=None
 browser_original=a.create_atspi_node
 def browser_node(node,depth=0,flag=None):
  global browser_runtime
  xml=browser_original(node,depth,flag)
  try:
   if browser_runtime is not None and node.app.bus_name==browser_runtime['bus_name']:
    state=node.getState()
    if state.contains(a.pyatspi.STATE_FOCUSED) and state.contains(a.pyatspi.STATE_VISIBLE) and state.contains(a.pyatspi.STATE_SHOWING):
     item=dict(object_path=node.path,role=node.getRoleName())
     if item not in browser_focus:browser_focus.append(item)
    xml.set('qw3-native-focused','true' if state.contains(a.pyatspi.STATE_FOCUSED) else 'false')
    if node.path==browser_tab_path:xml.set('qw3-browser-tab-path',browser_tab_path)
  except Exception:
   browser_runtime=None
  return xml
 a.create_atspi_node=browser_node
 browser_root=a.create_atspi_node(frames[0])
 if browser_runtime is not None:
  try:
   assert proc.stat().st_uid==browser_runtime['owner_uid']
   assert os.readlink(proc/'exe')=='/opt/google/chrome/chrome'
   assert int((proc/'stat').read_text().rpartition(')')[2].split()[19])==browser_runtime['owner_start_ticks']
   assert re.search(r'0x[0-9a-fA-F]+',subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)).group()==expected['handle']
   assert browser_frame.name==expected['title']
  except Exception:browser_runtime=None
 if browser_runtime is not None and len(browser_focus)<=1:
  browser_runtime['focus_objects']=browser_focus
  for browser_xml in browser_root.iter():
   if browser_xml.get('qw3-browser-tab-path')==browser_tab_path:
    browser_xml.set('qw3-browser-runtime',json.dumps(browser_runtime))
 xml=a.lxml.etree.tostring(browser_root,encoding='unicode')'''
    injection = injection.replace('FOREGROUND_LITERAL', repr(foreground)).replace('RESOURCE_LITERAL', repr(RESOURCE))
    result = source.replace(anchor, injection, 1)
    compile(result, 'native-browser-source-extension', 'exec')
    return result
