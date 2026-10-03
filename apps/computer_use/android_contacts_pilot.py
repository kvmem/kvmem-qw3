"""Host-only fixtures and saved-contact checks in the owned test emulator."""
import re
import shlex

NAME='vnd.android.cursor.item/name'
PHONE='vnd.android.cursor.item/phone_v2'
EMAIL='vnd.android.cursor.item/email_v2'
RAW='content://com.android.contacts/raw_contacts'
DATA='content://com.android.contacts/data'


def content(adb,arguments):
    # adb shell joins argv into a command. Quote every fixture value explicitly.
    result=adb(['shell',shlex.join(['content',*arguments])],capture_output=True,text=True)
    text=result.stdout+result.stderr
    if 'Exception' in text or text.startswith('Error'):
        raise RuntimeError('Test contacts provider rejected the administrative request')
    return text


def data_rows(adb):
    raw=content(adb,['query','--uri',DATA,'--projection','_id:raw_contact_id:mimetype:data1:data2:data3',
                     '--sort','_id ASC'])
    rows=[]
    for line in raw.splitlines():
        if not line.startswith('Row:'):
            continue
        values=re.split(r', (?=[a-zA-Z0-9_]+=)',re.sub(r'^Row: \d+ ', '',line))
        row=dict(pair.split('=',1) for pair in values)
        if not {'_id','raw_contact_id','mimetype','data1'}<=row.keys():
            raise ValueError('Malformed independent contacts result')
        rows.append(row)
    if not rows and raw.strip()!='No result found.':
        raise ValueError('Unknown empty contacts result')
    return rows


def create_contact(adb,name,phone,email):
    def raw_ids():
        result=content(adb,['query','--uri',RAW,'--projection','_id'])
        return {value for value in re.findall(r'\b_id=(\d+)',result)}
    before=raw_ids()
    # New Android content versions do not print the inserted URI. Discover
    # the one new row; never retry the insert merely because stdout is empty.
    content(adb,['insert','--uri',RAW,'--bind','aggregation_mode:i:3'])
    created=raw_ids()-before
    if len(created)!=1:
        raise ValueError('Cannot identify exactly one newly created fixture contact')
    raw_id=created.pop()
    fields=[(NAME,name),(PHONE,phone),(EMAIL,email)]
    for mime,value in fields:
        content(adb,['insert','--uri',DATA,'--bind','raw_contact_id:i:'+raw_id,
                     '--bind','mimetype:s:'+mime,'--bind','data1:s:'+value])
    return raw_id


def initialize(adb,identity):
    # A unique target also makes earlier pilot contacts unrelated preservation
    # checks. Never clear the address book or edit an existing unknown contact.
    before=data_rows(adb)
    name='QW3 Pilot '+identity
    email='qw3.'+identity+'@example.invalid'
    target=create_contact(adb,name,'+12025550147',email)
    initial=data_rows(adb)
    if [r for r in initial if r['raw_contact_id']!=target]!=before:
        raise ValueError('Contact fixture changed unrelated rows')
    fixture={'raw_contact_id':target,'initial_name':name,'final_name':'QW3 Local Pilot '+identity,
             'phone':'+12025550186','email':email,'initial_rows':initial,'model_used':False}
    prompt=(f'In the Contacts app, edit the existing contact {name}. Change the name to {fixture["final_name"]} '
            f'and phone number to {fixture["phone"]}. Preserve its email address {email}. '
            'Save and verify the result. Do not change or delete any other contacts, and do not send messages or make calls.')
    return fixture,prompt


def grade(adb,fixture):
    rows=data_rows(adb)
    target=fixture['raw_contact_id']
    own=[r for r in rows if r['raw_contact_id']==target]
    values=lambda mime: sorted(r['data1'] for r in own if r['mimetype']==mime)
    checks={'name':values(NAME)==[fixture['final_name']],
            'phone':sorted(re.sub(r'\D','',x) for x in values(PHONE))==[re.sub(r'\D','',fixture['phone'])],
            'email_preserved':values(EMAIL)==[fixture['email']],
            'other_contacts_unchanged':[r for r in rows if r['raw_contact_id']!=target]==
                [r for r in fixture['initial_rows'] if r['raw_contact_id']!=target]}
    return {'kind':'independent-saved-contacts-check','passed':all(checks.values()),'checks':checks,
            'rows':rows,'target_raw_contact_id':target}
