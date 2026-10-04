"""Version-qualified, read-only application behavior for Linux observations.

This candidate does not change input bytes, editor preferences, task prompts or
graders. Only the observed Gedit version/setting can enable its behavior note.
"""
import json

from qw3_computer.linux_driver import LinuxDriver


PROFILE_SCRIPT='''import json,subprocess
def read(argv):
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=1)
  return {"returncode":p.returncode,"stdout":p.stdout.strip()}
 except (OSError,subprocess.TimeoutExpired):
  return {"returncode":None,"stdout":""}
print(json.dumps({"version":read(["gedit","--version"]),"ensure_trailing_newline":read(["gsettings","get","org.gnome.gedit.preferences.editor","ensure-trailing-newline"])}))
'''


def describe(profile):
    version=profile.get('version',{})
    setting=profile.get('ensure_trailing_newline',{})
    if version.get('returncode')!=0 or version.get('stdout')!='gedit - Version 41.0':
        return {'status':'unqualified_version','process':'Gedit','input_transformed':False}
    value=setting.get('stdout')
    if setting.get('returncode')!=0 or value not in {'true','false'}:
        return {'status':'setting_unavailable','process':'Gedit','input_transformed':False}
    result={'status':'qualified','process':'Gedit','version':'41.0','ensure_trailing_newline':value=='true',
            'provenance':'Read-only installed version and current editor setting; native serialization qualification.',
            'input_transformed':False,'preferences_changed':False}
    if value=='true':
        result['save_behavior']='Saving appends one LF to the document buffer. A trailing empty buffer line becomes an extra blank file line. Account for this when verifying the requested saved text.'
    else:
        result['save_behavior']='The editor setting to append an LF on save is disabled; preserve the requested document text literally.'
    return result


class SemanticLinuxDriver(LinuxDriver):
    async def observe(self):
        observed=await super().observe()
        if observed['foreground'].get('process')=='Gedit':
            try:
                result=await self.execute(PROFILE_SCRIPT,read_timeout=3)
                observed['application_behavior']=describe(json.loads(result['output']))
            except Exception as exc:
                observed['application_behavior']={'status':'unavailable','process':'Gedit',
                    'error_kind':type(exc).__name__,'input_transformed':False}
        return observed
