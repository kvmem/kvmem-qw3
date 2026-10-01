"""Exercise the candidate CRM image using the original task's public leads."""
import argparse
import copy
import json
from pathlib import Path
from urllib.error import HTTPError

from .sites import Site
from .task038 import grade, initial_states
from .upstream import pure_task_module


def qualify(output):
    output.mkdir(parents=True,exist_ok=False)
    site=Site('cloudcrm',output.name,claims=output/'identities');site.origin='http://127.0.0.1:18115'
    original=initial_states()['cloudcrm'];site.initialize(original)
    cases=[]
    def check(name,condition):
        cases.append(dict(case=name,passed=bool(condition)));assert condition,name
    def rejects(lead,payload,code):
        before=site.state()['data']
        try:site.request('POST',f'/api/crm/leads/{lead}/convert',payload)
        except HTTPError as exc:return exc.code==code and site.state()['data']==before
        return False
    targets=pure_task_module('038')['CONVERTED_LEADS']
    for target in targets:
        lead=target['leadId'];before=site.state()['data']
        check(lead+'_initial_qualified_unconverted',next(r for r in before['leads'] if r['leadId']==lead)['status']=='Qualified')
        payload=dict(createAccount=True,createContact=True,createOpportunity=True,amount=target['expectedAmount'],
                     closeDate=target['expectedCloseDate'],stage='Prospecting')
        site.request('POST',f'/api/crm/leads/{lead}/convert',payload)
        after=site.state()['data'];current=next(r for r in after['leads'] if r['leadId']==lead)
        check(lead+'_converted_once',current['isConverted'] is True and current['status']=='Qualified'
              and all(len(after[k])==len(before[k])+1 for k in ('accounts','contacts','opportunities')))
        check(lead+'_repeat_rejected_without_mutation',rejects(lead,payload,409))
    scored=grade(site.state());(output/'conversions-grade.json').write_text(json.dumps(scored,indent=2))
    check('three_correctly_linked_conversions',scored['checks']['three_conversions_linked_correctly'])
    check('original_records_and_fields_preserved',all(scored['checks'][k] for k in ['all_40_leads_retained_once','all_protected_lead_fields_preserved','accounts_originals_preserved','contacts_originals_preserved','opportunities_originals_preserved']))
    # Changing the ordinary lead status cannot remove conversion history.
    site.request('PATCH','/api/crm/leads/lead-8',{'status':'Working'})
    check('status_change_cannot_bypass_duplicate_guard',rejects('lead-8',payload,409))
    check('opportunity_without_account_refused',rejects('lead-1',{'createAccount':False,'createOpportunity':True},422))
    check('unknown_lead_refused',rejects('missing-lead',payload,404))
    other=Site('cloudcrm',output.name+'-other',claims=output/'identities');other.origin=site.origin;other.initialize(copy.deepcopy(original))
    check('other_identity_unchanged',other.state()['data']==original['data'])
    (output/'after.json').write_text(json.dumps(site.state(),indent=2))
    report=dict(kind='candidate_crm_conversion_qualification',passed=all(c['passed'] for c in cases),cases=cases,
                formal_run=False,model_used=False,public_api_component_test=True)
    (output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':report['passed'],'cases':len(cases)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    qualify(parser.parse_args().output)
