"""Neutral P2 rules and frozen answers. Only public HTML enters the Windows VM."""
import argparse
import hashlib
import html
import json
from pathlib import Path

CASES = [
 dict(id='calculation-1',category='calculation',title='Travel reimbursement queue',facts=[
  'Process eligible items in submission order; the task budget is 250.00.',
  'Each eligible item receives 70% of cost, capped at 180.00 per item.',
  'An item without a receipt receives 0.00; it consumes no budget.',
  'When the budget is insufficient, pay the remaining budget, then stop funding later items.',
  'Submission 1: item A; cost 210.00; receipt present.',
  'Submission 2: item B; cost 320.00; receipt present.',
  'Submission 3: item C; cost 125.00; receipt missing.',
  'Do not round intermediate results. Report final amounts with two decimal places.'
 ],question='按规则分配报销额度，返回 A、B、C 各自金额以及剩余额度。',
 schema={'A':'decimal string','B':'decimal string','C':'decimal string','remaining':'decimal string'},
 expected={'A':'147.00','B':'103.00','C':'0.00','remaining':'0.00'}),
 dict(id='calculation-2',category='calculation',title='Exact licence purchase',facts=[
  'Buy exactly 18 seats. A pack cannot be divided; do not buy surplus seats.',
  'Pack A: 6 seats; price 215.00; at most 2 packs.',
  'Pack B: 4 seats; price 152.00; at most 4 packs.',
  'Pack C: 1 seat; price 43.00; any nonnegative integer quantity.',
  'Add a single shipping fee of 35.00 if the pack subtotal is below 700.00; otherwise shipping is 0.00.',
  'Total payable, including shipping, must not exceed 710.00.',
  'Choose the feasible combination with the lowest total payable.',
  'If total payable ties, choose fewer packs overall; if still tied, choose more A packs.'
 ],question='选择最便宜的可行采购组合，返回三种包数量和含运费实付金额。',
 schema={'A':'integer','B':'integer','C':'integer','payable':'decimal string'},
 expected={'A':2,'B':1,'C':2,'payable':'703.00'}),
 dict(id='calculation-3',category='calculation',title='Offline workstation plan',facts=[
  'Select exactly 3 components. Python and LaTeX are mandatory.',
  'Available installation space after reserving recovery space: 3400 MB.',
  'Maximum download allowance: 2000 MB. Do not mix MB and GiB.',
  'Python: installed 780 MB; download 350 MB; utility 5.',
  'Editor: installed 1260 MB; download 800 MB; utility 8.',
  'Notebook: installed 920 MB; download 600 MB; utility 7.',
  'LaTeX: installed 1540 MB; download 1100 MB; utility 9.',
  'Git: installed 680 MB; download 250 MB; utility 4.',
  'Maximize total utility among feasible selections; alphabetical order breaks a tie.'
 ],question='列出应选择的组件（按英文字母排序）、总安装空间和总下载量。',
 schema={'components':['component names in alphabetical order'],'installed_mb':'integer','download_mb':'integer'},
 expected={'components':['Git','LaTeX','Python'],'installed_mb':3000,'download_mb':1700}),
 dict(id='revision-1',category='revision',title='Effective reimbursement policy',facts=[
  'Request date: 2026-10-02. Expense: 180.00. Contract signed: 2026-09-10.',
  'Only approved policies effective on or before the request date can apply.',
  'Use the latest such policy, including its stated exceptions.',
  'P1: approved; effective 2026-09-01; reimburse 80%; maximum 120.00.',
  'P2: approved; effective 2026-10-01; replaces P1; reimburse 60%; maximum 100.00.',
  'P2 exception: contracts signed before 2026-09-15 retain an 80% rate and a 120.00 maximum.',
  'P3: approved; effective 2026-10-03; reimburse 100%; maximum 200.00.',
  'Email sent 2026-10-02: quotes P1 and proposes a 150.00 cap. The proposal is a draft, not approved.'
 ],question='返回适用政策 ID、是否使用例外，以及最终报销金额。',
 schema={'policy':'string','exception_applied':'boolean','amount':'decimal string'},
 expected={'policy':'P2','exception_applied':True,'amount':'120.00'}),
 dict(id='revision-2',category='revision',title='Python release selection',facts=[
  'Installation date: 2026-10-02. Project tag: LTS.',
  'Use the latest approved release notice already effective; project exceptions in that notice take precedence.',
  'N15: approved; effective 2026-09-20; all projects use Python 3.13.15.',
  'N17: approved; effective 2026-10-01; standard projects use Python 3.13.16; LTS projects use Python 3.12.13.',
  'N18: draft; effective 2026-10-02; proposes Python 3.14.0 for all projects.',
  'N19: approved; effective 2026-10-05; standard projects use Python 3.14.0; LTS use Python 3.13.16.',
  'Ticket edited 2026-10-02: repeats the old N15 version 3.13.15; it is not an approved release notice.'
 ],question='返回应遵循的发布通知 ID 和安装版本。',schema={'notice':'string','version':'string'},
 expected={'notice':'N17','version':'3.12.13'}),
 dict(id='revision-3',category='revision',title='Purchase approvals after revision',facts=[
  'Decision time: 2026-10-02 16:00. Purchase: 8500.00. Urgent: yes.',
  'Use only approved rules effective by decision time. The latest effective rule replaces older rules.',
  'R4: approved; effective 2026-09-30 09:00; manager required; finance required only for amount >10000.',
  'R5: draft; dated 2026-10-01 12:00; finance proposed for amount >=8000.',
  'R6: approved; effective 2026-10-02 12:00; manager always required; finance required for amount >=9000.',
  'R6 also requires security approval for urgent purchases with amount >=7000.',
  'Manager approval: received. Finance approval: not received. Security approval: not received.',
  'Submission is allowed only after every approval required by the applicable rule is received.'
 ],question='返回有效规则 ID、所需审批列表和缺少的审批列表（均按英文字母排序），以及现在能否提交。',
 schema={'rule':'string','required':['approval names'],'missing':['approval names'],'can_submit':'boolean'},
 expected={'rule':'R6','required':['manager','security'],'missing':['security'],'can_submit':False}),
 dict(id='selection-1',category='selection',title='Application and attachment review',facts=[
  'Review date: 2026-10-02. Select every eligible applicant; do not rank or choose only one.',
  'Eligibility requires at least 3 complete years of Python experience, an unexpired certificate, and an attached portfolio.',
  'A certificate is valid through its expiry date. Only attachments currently present count.',
  'Application A: 3 years; certificate expires 2027-01-31; portfolio file A.zip present.',
  'Application B: 6 years; certificate expires 2026-10-01; portfolio file B.zip present.',
  'Application C: 2 years; certificate expires 2027-02-01; portfolio file C.zip present.',
  'Application D: 4 years; certificate expires 2026-10-02; initial email says portfolio pending.',
  'Attachment index updated 2026-10-02 10:00: D.zip is now present and belongs to D.',
  'Application E: 5 years; certificate expires 2027-03-01; email says portfolio sent.',
  'Current attachment index contains A.zip, B.zip, C.zip and D.zip only; E.zip is absent.'
 ],question='列出全部符合条件的申请 ID，按字母顺序返回。',schema={'eligible':['IDs in alphabetical order']},
 expected={'eligible':['A','D']}),
 dict(id='selection-2',category='selection',title='Windows offline installation assets',facts=[
  'Target: Windows x64. Required components: Python 3.13.16 and Chinese fonts 2.4.',
  'Choose one asset per component. It must match platform, architecture and exact version, have an approved publisher and a verified hash.',
  'A1: Python 3.13.16; Windows ARM64; approved publisher; hash verified.',
  'A2: Python 3.13.15; Windows x64; approved publisher; hash verified.',
  'A3: Python 3.13.16; Windows x64; approved publisher; hash mismatch.',
  'A4: Python 3.13.16; Windows x64; approved publisher; hash verified.',
  'F1: Chinese fonts 2.4; Windows any architecture; publisher not approved; hash verified.',
  'F2: Chinese fonts 2.4; Windows any architecture; approved publisher; hash verified.',
  'F3: Chinese fonts 2.5; Windows any architecture; approved publisher; hash verified.',
  'Mirror email calls A3 the newest upload. Upload recency cannot override hash verification.',
  'The dependency index says Python needs no other assets and Chinese fonts need no other assets.'
 ],question='选择最小的完整安装素材集合，返回素材 ID（按字母顺序）。',schema={'assets':['IDs in alphabetical order']},
 expected={'assets':['A4','F2']}),
 dict(id='selection-3',category='selection',title='Reimbursement records across documents',facts=[
  'Select reimbursable invoices and compute the sum using current evidence.',
  'Rules: require a receipt and manager approval. A confirmed refund reduces the invoice before any category cap.',
  'Meal reimbursement is capped at 200.00 per invoice. Equipment has no category cap.',
  'Invoice ledger: I11 equipment 480.00; I12 meal 250.00; I13 equipment 300.00.',
  'Receipt index: I11 and I12 present; I13 absent.',
  'Approval letter: manager approved I11, I12 and I13.',
  'Confirmed refund notice: I11 refunded 120.00. No other invoice has a refund.',
  'Older export: I11 paid 480.00 before the refund; its export date precedes the confirmed refund notice.',
  'Do not treat the approval letter as a substitute for a missing receipt.'
 ],question='返回可报销发票 ID（排序）、各张可报销金额和总额；只列出可报销发票。',
 schema={'invoices':['sorted IDs'],'amounts':{'invoice ID':'decimal string'},'total':'decimal string'},
 expected={'invoices':['I11','I12'],'amounts':{'I11':'360.00','I12':'200.00'},'total':'560.00'}),
 dict(id='workflow-1',category='workflow',title='Repair a project without changing global Python',facts=[
  'User request: repair the selected project; preserve source files, unrelated projects and system PATH.',
  'Current diagnosis: project requests module is absent, but its matching distribution metadata is still present.',
  'Pinned reviewed wheels are available locally. The existing generated launcher points at the wrong environment.',
  'Mandatory order: back up the launcher before any repair; fix packages before creating the corrected launcher.',
  'After repairs, verify in a fresh project terminal, then restart the app and verify again before reporting success.',
  'Available action B: back up the current generated launcher.',
  'Available action F: force-reinstall the exact reviewed project wheels into the project venv.',
  'Available action L: replace the generated launcher with one activating the project venv.',
  'Available action V: run main.py, pytest and pip check in a fresh project terminal.',
  'Available action R: restart the app and repeat independent project verification.',
  'Available action S: report completion with verified project and launcher paths.',
  'Available action U: uninstall the machine-wide Python.',
  'Available action P: ordinary pip install without force; it reports Requirement already satisfied.',
  'No action has been performed yet. Do not add actions not in the list.'
 ],question='返回可以完成任务的最短动作 ID 顺序。',schema={'actions':['ordered action IDs']},
 expected={'actions':['B','F','L','V','R','S']}),
 dict(id='workflow-2',category='workflow',title='Uncertain installer delivery',facts=[
  'Job identity: install-42. Only one installer may be active for this project.',
  'The installer was dispatched. The status-response connection then timed out.',
  'A timeout is not proof that the installer failed or stopped. No terminal receipt has been obtained.',
  'Process inspection: the original installer PID is still running.',
  'Task has 120 seconds remaining. A read-only query of the original job receipt takes at most 5 seconds.',
  'Policy: do not start a replacement while the original result is unknown. Do not kill or uninstall without reconciliation.',
  'Choice Q: query install-42 status and retained receipt without launching an installer.',
  'Choice N: create install-43 and launch the same installer again.',
  'Choice K: kill the original process, delete its files and retry.',
  'Choice C: report success because dispatch was acknowledged.',
  'No output artifact or fresh-terminal verification exists yet.'
 ],question='选择现在唯一合规且有用的下一动作，并判断能否宣告完成、能否重发安装。',
 schema={'next_action':'action ID','can_complete':'boolean','can_relaunch':'boolean'},
 expected={'next_action':'Q','can_complete':False,'can_relaunch':False}),
 dict(id='workflow-3',category='workflow',title='Workbook completion audit',facts=[
  'Task: preserve source.xlsx, write result.xlsx, use multiplication formulas for D2:D4 and a SUM formula in D5.',
  'Keep row records aligned. Rows must be in descending order of their computed totals.',
  'Original rows: A quantity 2 price 12; B quantity 3 price 8; C quantity 1 price 15.',
  'Tied totals must keep their original row order.',
  'Saved output path from file receipt: draft.xlsx. result.xlsx is absent.',
  'Source hash before and after: equal.',
  'Saved rows in order: B quantity 3 price 8 total 24; A quantity 2 price 12 total 24; C quantity 1 price 15 total 15.',
  'D2 formula: =B2*C2. D3 formula: =B3*C3. D4 stored value: 15, with no formula.',
  'D5 formula: =SUM(D2:D4). Cached result: 63.',
  'Agent final message says: all formulas and save path have been checked; task complete.',
  'Failure IDs: output_path, source_changed, tie_order, row_records, missing_row_formula, total_formula.',
  'Judge actual evidence. Return every applicable failure ID, not hypothetical problems.'
 ],question='是否通过完整验收？列出全部实际失败项（按字母顺序），以及这些数据的正确总额。',
 schema={'passed':'boolean','failures':['IDs in alphabetical order'],'correct_total':'integer'},
 expected={'passed':False,'failures':['missing_row_formula','output_path','tie_order'],'correct_total':63}),
]


def build(output):
    output.mkdir(parents=True,exist_ok=False);public=output/'public';public.mkdir()
    samples=[];hashes={}
    for index,case in enumerate(CASES):
        assert len(case['facts']) <= 15
        title=case['title'];lines=''.join('<p>'+html.escape(line)+'</p>' for line in case['facts'])
        page='<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:17px/1.35 Arial,sans-serif;color:#152438;background:#fff;margin:30px 40px}h1{font-size:24px;margin:0 0 20px}p{margin:0 0 12px;max-width:1170px}</style><h1>'+html.escape(title)+'</h1>'+lines+'</html>'
        name=case['id']+'.html';(public/name).write_text(page,encoding='utf-8');hashes[name]=hashlib.sha256(page.encode()).hexdigest()
        samples.append(case|{'split':'heldout' if index%3==2 else 'development','html':name,'equivalent_text':title+'\n'+'\n'.join(case['facts']),'ground_truth_reviewed':False})
    corpus={'kind':'P2_rules_pending_windows_capture_and_review','planned_requests':96,'samples':samples,'max_tokens':4096,'temperature':0,'repetitions':2,'thinking':[False,True],'representations':['screenshot','text']}
    (public/'materials.json').write_text(json.dumps({'files':hashes},indent=2)+'\n')
    (output/'corpus.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2)+'\n')
    return corpus


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
