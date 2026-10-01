"""Extract task mailbox bytes into an offline-only Thunderbird Local Folders profile.

Never copy source logins, cookies, account keys, SMTP/IMAP credentials or live
server settings. Original message bytes and attachments remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import tarfile

from .source import asset

MAILBOXES={
    "006":("thunderbird-profile.tar.gz","/Mail/outlook.office365.com/Inbox"),
    "010":("email_010.tar.gz","/ImapMail/imap.163.com/INBOX"),
}


def prepare_profile(task,output,guest_profile):
    archive_name,suffix=MAILBOXES[task]
    archive_path=asset(task,archive_name)
    output=Path(output);(output/"Mail/Local Folders").mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive_path) as archive:
        members=[m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
        if len(members)!=1:raise ValueError("Expected one original task inbox")
        member=members[0]
        data=archive.extractfile(member).read()
    (output/"Mail/Local Folders/Inbox").write_bytes(data)
    # A Local Folders account has no network transport or send identity. The
    # startup helper also passes -offline; both settings are part of the task.
    preferences={
        "mail.accountmanager.accounts":"account1",
        "mail.accountmanager.defaultaccount":"account1",
        "mail.accountmanager.localfoldersserver":"server1",
        "mail.account.account1.server":"server1",
        "mail.server.server1.type":"none",
        "mail.server.server1.hostname":"Local Folders",
        "mail.server.server1.name":"OSWorld task "+task,
        "mail.server.server1.directory":guest_profile+r"\Mail\Local Folders",
        "mail.server.server1.directory-rel":"[ProfD]Mail/Local Folders",
        "mail.server.server1.storeContractID":"@mozilla.org/msgstore/berkeleystore;1",
        "mail.server.server1.login_at_startup":False,
        "mail.server.server1.check_new_mail":False,
        "mail.provider.suppress_dialog_on_startup":True,
        "mail.provider.enabled":False,
        "mailnews.start_page.enabled":False,
        "mailnews.start_page.override_url":"",
        "mailnews.start_page_override.mstone":"157.0",
        "mail.rights.version":1,
        "mail.winsearch.firstRunDone":True,
        "mail.shell.checkDefaultClient":False,
        "offline.autoDetect":False,
        "offline.startup_state":3,
        "network.online":False,
        "app.update.auto":False,
        "app.update.enabled":False,
        "datareporting.healthreport.uploadEnabled":False,
        "datareporting.policy.dataSubmissionEnabled":False,
        "toolkit.telemetry.enabled":False,
    }
    (output/"prefs.js").write_text("\n".join(f"user_pref({json.dumps(k)}, {json.dumps(v)});" for k,v in preferences.items())+"\n")
    return {"task":task,"source_archive":str(archive_path),"source_member":member.name,
        "mailbox_bytes":len(data),"mailbox_sha256":hashlib.sha256(data).hexdigest(),
        "profile_files":[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()],
        "adaptation":"original inbox bytes in local-only profile; no original login/account credentials"}


REGISTER_DEFAULT_PROFILE_PS = r'''
    # Register this local-only profile as the VM's default so open_app and a
    # normal close/reopen do not silently switch to the first-run profile.
    # Preserve the original registrations in the unique task-owned directory.
    $registration="$env:APPDATA\Thunderbird"
    $backup="C:\Qw3Computer\ProfileBackups\RUN_ID"
    if (Test-Path $backup) {throw 'Mailbox registration backup already exists'}
    New-Item -ItemType Directory -Force $backup | Out-Null
    $installationIds=@()
    foreach ($name in @('profiles.ini','installs.ini')) {
        $path=Join-Path $registration $name
        if (Test-Path $path) {
            Copy-Item $path $backup
            if ($name -eq 'installs.ini') {
                $installationIds=@([regex]::Matches([IO.File]::ReadAllText($path),'(?m)^\[([A-F0-9]+)\]\s*$') | ForEach-Object {$_.Groups[1].Value})
            }
        }
    }
    if ($installationIds.Count -ne 1) {throw 'Expected one reviewed Thunderbird installation registration'}
    $profiles="[General]`r`nStartWithLastProfile=1`r`nVersion=2`r`n`r`n[Profile0]`r`nName=qw3-osworld-local`r`nIsRelative=0`r`nPath=$profile`r`nDefault=1`r`n`r`n"
    $installs=''
    foreach ($installationId in $installationIds) {
        $profiles+="[Install$installationId]`r`nDefault=$profile`r`nLocked=1`r`n`r`n"
        $installs+="[$installationId]`r`nDefault=$profile`r`nLocked=1`r`n`r`n"
    }
    [IO.File]::WriteAllText((Join-Path $registration 'profiles.ini'),$profiles)
    [IO.File]::WriteAllText((Join-Path $registration 'installs.ini'),$installs)
'''
