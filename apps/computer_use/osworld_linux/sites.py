"""Host-only offline routing; official task logic and scoring remain unchanged."""
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import uuid

from osworld_windows.sites import Site, PORTS
from osworld_windows.source import ROOT

ASSETS = ROOT/'third_party/OSWorld-V2/cache/osworld_v2_assets'


def public_url(target):
    if target in PORTS:
        return f'http://{target}.qw3.test:18090'
    parsed = urlparse(target)
    if parsed.scheme != 'http' or parsed.hostname not in {f'{app}.qw3.test' for app in PORTS} or parsed.port != 18090:
        raise ValueError('Unknown local task website')
    return target


def payload(raw):
    if isinstance(raw, dict): return raw
    if not isinstance(raw,str): raise ValueError('Unsupported state payload')
    try: return json.loads(raw)
    except json.JSONDecodeError:
        path=Path(raw).resolve()
        if not path.is_relative_to(ASSETS.resolve()): raise ValueError('State material outside pinned local asset snapshot')
        return json.loads(path.read_text())


def install(claims):
    """Call before loading task modules. Browser cannot reach private state APIs."""
    from desktop_env.controllers import website
    user=uuid.uuid4().hex

    def prepare(app=None,state=None,additional_urls=None,cache_dir=None,cookie_save_name='state_cookie.json',*,url=None):
        if app is None: app=url
        elif url is not None: raise ValueError('Duplicate website arguments')
        apps=app if isinstance(app,list) else [app]
        states=state if isinstance(state,list) else [state]
        if isinstance(app,list)!=isinstance(state,list) or len(apps)!=len(states):raise ValueError('Mismatched state list')
        entries=[];urls=[]
        for target,raw in zip(apps,states):
            source=public_url(target);parsed=urlparse(source)
            site=Site(parsed.hostname.split('.')[0],user,claims)
            consumed=('time_data','action_data','enable_notifications') if site.app in {'teamchat','mailhub'} else ()
            site.initialize(payload(raw),consumed_fields=consumed)
            query=dict(parse_qsl(parsed.query,keep_blank_values=True));query['cookie']=user
            final=urlunparse(parsed._replace(query=urlencode(query)))
            entries.append({'cookie':'user_id='+user,'cookie_dict':{'user_id':user},'user_id_cookie':user,
                'source_url':source,'state_endpoint':site.origin+'/api/state','final_url':final})
            urls.append(final)
        if additional_urls:
            urls.extend(public_url(u) for u in ([additional_urls] if isinstance(additional_urls,str) else additional_urls))
        if cache_dir:
            path=Path(cache_dir)/cookie_save_name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'entries':entries},indent=2))
        return urls

    website.build_website_url=public_url
    website.resolve_website_url=public_url
    website.prepare_stateful_website_urls=prepare
    return user
