"""Private spreadsheet fixture and saved workbook checks, outside Computer MCP."""
import base64
import io
import json


def initial_workbook():
    from openpyxl import Workbook
    book = Workbook()
    work = book.active
    work.title = 'Work'
    for row in [('Project','Hours','Rate','Total'), ('Alpha',2,50,'=B2*C2'),
                ('Beta',3,40,'=B3*C3'), ('Gamma',4,25,'=B4*C4'), ('Total',None,None,'=SUM(D2:D4)')]:
        work.append(row)
    work.column_dimensions['A'].width = 25
    summary = book.create_sheet('Summary')
    summary['A1'] = 'Project budget'
    summary['A2'] = 'Grand total'
    summary['B2'] = '=SUM(Work!D2:D3)'
    summary['A4'] = 'Do not change this note'
    summary.column_dimensions['A'].width = 30
    stream = io.BytesIO()
    book.save(stream)
    return stream.getvalue()


def cells(data):
    from openpyxl import load_workbook
    book = load_workbook(io.BytesIO(data), data_only=False)
    return {sheet.title: {cell.coordinate:cell.value for row in sheet for cell in row if cell.value is not None}
            for sheet in book.worksheets}


def grade_bytes(data, initial):
    if data is None:
        return {'kind':'independent-saved-calc-workbook','passed':False,'error':'Saved file missing'}
    try:
        actual = cells(data)
        target = cells(initial)
        target['Work']['B3'] = 5
        target['Work']['C3'] = 60
        target['Summary']['B2'] = '=SUM(Work!D2:D4)'
        checks = {'beta_hours': actual.get('Work', {}).get('B3') == 5,
                  'beta_rate': actual.get('Work', {}).get('C3') == 60,
                  'summary_formula': actual.get('Summary', {}).get('B2') == '=SUM(Work!D2:D4)',
                  'other_cells_and_formulas_preserved': actual == target}
        # Independently calculate the intended values from saved inputs. Cached
        # formulas can be stale or omitted by XLSX producers, so report them
        # separately instead of treating missing cache data as a false pass.
        from openpyxl import load_workbook
        cache = load_workbook(io.BytesIO(data), data_only=True)
        cached = {'work_total':cache['Work']['D5'].value, 'summary_total':cache['Summary']['B2'].value} if set(actual) == set(target) else {}
        cache_valid = all(value in (None,500) for value in cached.values()) and len(cached) == 2
        checks['cached_totals_not_contradictory'] = cache_valid
        return {'kind':'independent-saved-calc-workbook','passed':all(checks.values()), 'checks':checks,
                'saved_cells':actual,'cached_totals':cached, 'expected_total':500,
                'grading_scope':'Saved numeric inputs, exact formulas and all nonempty cells; missing calculation cache is reported'}
    except Exception as exc:
        return {'kind':'independent-saved-calc-workbook','passed':False,'error':type(exc).__name__+': '+str(exc)}


def initialize(linux_execute, directory):
    from pathlib import Path
    from browser_pilot import identity
    guest_file = '/home/user/Desktop/qw3-pilot-budget-' + identity(Path(directory).name) + '.xlsx'
    initial = initial_workbook()
    linux_execute('from pathlib import Path;import base64,subprocess;'
                  f'p=Path({guest_file!r});data=base64.b64decode({base64.b64encode(initial).decode()!r});'
                  'f=p.open("xb");f.write(data);f.close();'
                  'subprocess.Popen(["libreoffice","--calc",str(p)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)')
    (Path(directory)/'initial-workbook.xlsx').write_bytes(initial)
    prompt = (f'In LibreOffice Calc, edit the existing workbook {guest_file}. On the Work sheet change only '
              'the Beta row: Hours (B3) to 5 and Rate (C3) to 60. Keep its Total formula and all other Work cells unchanged. '
              'On the Summary sheet fix B2 so its formula sums Work!D2:D4, including Gamma. Keep all other cells unchanged. '
              'Save in the original XLSX format and verify both total cells show 500. Use only ordinary GUI interactions; no terminal, scripts or developer tools.')
    return {'guest_file':guest_file,'initial_file':str(Path(directory)/'initial-workbook.xlsx'),
            'administrative_fixture_only':True,'model_used':False}, prompt


def grade(linux_execute, fixture, directory):
    from pathlib import Path
    reply = linux_execute('from pathlib import Path;import json,base64;'
                          f'p=Path({fixture["guest_file"]!r});print(json.dumps(base64.b64encode(p.read_bytes()).decode() if p.exists() else None))')
    encoded = json.loads(reply['output'])
    saved = base64.b64decode(encoded) if encoded is not None else None
    if saved is not None:
        (Path(directory)/'saved-workbook.xlsx').write_bytes(saved)
    return grade_bytes(saved, Path(fixture['initial_file']).read_bytes())
