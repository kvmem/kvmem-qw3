"""Build source overlays for local decorative assets without changing task state.

Only copied frontend sources are edited. Existing backend containers and raw
state URLs stay unchanged; Calendar gets a separate frontend container. Assets
must already be frozen and verified. This module makes no network requests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from .source import ROOT, sha256


ASSET_HELPER = '''import urls from "./qw3_offline_urls.json";

export function offlineAsset(valueTYPE) RETURNTYPE {
  if (!value) return value;
  const pinned = URLS[value];
  if (pinned) return pinned;
  // Dynamic avatar names remain local. Never contact the external generator.
  try {
    const parsed = new URL(value);
    if (parsed.hostname === "ui-avatars.com" && parsed.pathname === "/api/") {
      const name = (parsed.searchParams.get("name") || "?").split("@")[0];
      const initials = name.trim().split(/[\\s._+-]+/).filter(Boolean)
        .map(part => Array.from(part)[0]).slice(0, 2).join("").toUpperCase() || "?";
      const escaped = initials.replace(/[&<>"']/g, c => "&#" + c.charCodeAt(0) + ";");
      const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
        + '<rect width="64" height="64" fill="#f0e9e9"/>'
        + '<text x="32" y="34" text-anchor="middle" dominant-baseline="middle"'
        + ' font-family="Arial,sans-serif" font-size="26" fill="#555">' + escaped + '</text></svg>';
      return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
    }
  } catch { /* Relative, data and blob URLs retain their normal behavior. */ }
  return value;
}
'''


def verified_assets(inventory):
    """Return public bytes and URL map, never private manifests or task labels."""
    public, mapping = {}, {}
    for candidate in ('candidate', 'calendar-static-candidate', 'supplemental-static-candidate-02'):
        path = inventory / candidate
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest.get('complete') is not True:
            raise ValueError('Decorative image freeze is incomplete: '+candidate)
        for row in manifest['records']:
            if not re.fullmatch(r'[0-9a-f]{64}\.img', row['file']):
                raise ValueError('Unexpected frozen asset path')
            payload = (path / 'files' / row['file']).read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != row['sha256'] or len(payload) != row['bytes']:
                raise ValueError('Frozen decorative asset changed')
            suffix = '.jpg' if payload.startswith(b'\xff\xd8\xff') else '.png' if payload.startswith(b'\x89PNG\r\n\x1a\n') else None
            if suffix is None:
                raise ValueError('Unexpected decorative image format')
            relative = 'images/'+digest+suffix
            public[relative] = payload
            local_url = '/_qw3_assets/'+relative
            if row['url'] in mapping and mapping[row['url']] != local_url:
                raise ValueError('Conflicting frozen URL')
            mapping[row['url']] = local_url
    for candidate, sheet in [('font-candidate', 'fonts.css'), ('google-sans-candidate', 'google-sans.css')]:
        path = inventory / candidate
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest.get('complete') is not True:
            raise ValueError('Font freeze incomplete: '+candidate)
        for row in manifest['records']:
            if not re.fullmatch(r'[0-9a-f]{64}\.(woff2|ttf)', row['file']):
                raise ValueError('Unexpected frozen font path')
            payload = (path / row['file']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != row['sha256'] or len(payload) != row['bytes']:
                raise ValueError('Frozen font changed')
            public['fonts/'+row['file']] = payload
        css = (path / 'fonts.css').read_bytes()
        if hashlib.sha256(css).hexdigest() != manifest['local_css_sha256'] or b'https://' in css or b'http://' in css:
            raise ValueError('Font stylesheet changed or still depends on the network')
        public[sheet] = css
    return public, mapping


def build(inventory, output):
    inventory, output = Path(inventory).resolve(), Path(output).resolve()
    shared = ROOT / 'apps/computer_use/.runtime/windows/shared'
    upstream = ROOT / 'third_party/OSWorld-web'
    if output == upstream or output.is_relative_to(upstream) or output == shared or output.is_relative_to(shared):
        raise ValueError('Build overlays must stay outside upstream sources and the guest share')
    public, mapping = verified_assets(inventory)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='offline_private_site_source_overlays', complete=False, formal_run=False,
                  source_checkout_modified=False, backend_state_rewritten=False,
                  frozen_url_count=len(mapping), public_asset_count=len(public), contexts={},
                  dynamic_avatar_policy='Local SVG initials for ui-avatars.com URLs; no emails transmitted. This decorative rendering adaptation is disclosed and common to all later trials.')
    try:
        for app in ('formcraft', 'teamchat', 'cloudcrm', 'mailhub', 'calendar'):
            source = upstream / (app+'_web')
            if app != 'calendar':
                source /= 'frontend'
            context = output / app
            shutil.copytree(source, context, ignore=shutil.ignore_patterns(
                '.git', '.env*', 'node_modules', '.next', 'dist', '__pycache__', '.pytest_cache', 'coverage'))
            patches = []
            typed = app == 'calendar'
            suffix = '.tsx' if typed else '.jsx'
            for path in sorted((context / 'src').rglob('*'+suffix)):
                text = path.read_text()
                if re.search(r'</img\s*>|<img\b[^>]*\bsrcSet\s*=', text):
                    raise ValueError('Unexpected image markup needing explicit review: '+str(path))
                count = len(re.findall(r'<img\b', text))
                if not count:
                    continue
                relative = str(path.relative_to(context))
                before = sha256(path)
                text = re.sub(r'<img\b', '<Qw3OfflineImage', text)
                module = os.path.relpath(context / 'src/qw3_offline_image', path.parent)
                if not module.startswith('.'):
                    module = './'+module
                addition = 'import Qw3OfflineImage from '+json.dumps(module)+';\n'
                directive = re.match(r'^(\s*[\'\"]use client[\'\"];?\s*\n)', text)
                point = directive.end() if directive else 0
                text = text[:point]+addition+text[point:]
                path.write_text(text)
                patches.append(dict(path=relative, before_sha256=before, after_sha256=sha256(path), image_elements=count))
            helper = ASSET_HELPER.replace('valueTYPE', 'value: string | undefined' if typed else 'value')
            helper = helper.replace('RETURNTYPE', ': string | undefined' if typed else '')
            helper = helper.replace('URLS[value]', '(urls as Record<string,string>)[value]' if typed else 'urls[value]')
            (context / ('src/qw3_offline_asset'+('.ts' if typed else '.js'))).write_text(helper)
            component = ('import type { ComponentProps } from "react";\n' if typed else 'import React from "react";\n')
            component += 'import { offlineAsset } from "./qw3_offline_asset";\n'
            component += 'export default function Qw3OfflineImage(props'+(': ComponentProps<"img">' if typed else '')+') {\n'
            component += '  return <img {...props} src={typeof props.src === "string" ? offlineAsset(props.src) : props.src} alt={props.alt} />;\n}\n'
            (context / ('src/qw3_offline_image'+suffix)).write_text(component)
            (context / 'src/qw3_offline_urls.json').write_text(json.dumps(mapping, sort_keys=True, indent=2))
            if app != 'calendar':
                css_path = context / 'src/index.css'
                css = css_path.read_text()
                pattern = r'^@import url\("https://fonts\.googleapis\.com/[^"\n]+"\);'
                css, count = re.subn(pattern, '@import url("/_qw3_assets/fonts.css");', css)
                if count != 1:
                    raise ValueError('Unexpected upstream font import')
                before = sha256(css_path); css_path.write_text(css)
                patches.append(dict(path='src/index.css', before_sha256=before, after_sha256=sha256(css_path), font_imports=count))
            if app == 'mailhub':
                html_path = context / 'index.html'
                html = html_path.read_text()
                remote = 'https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&display=swap'
                if html.count(remote) != 1:
                    raise ValueError('Unexpected MailHub font link')
                before = sha256(html_path)
                html_path.write_text(html.replace(remote, '/_qw3_assets/google-sans.css'))
                patches.append(dict(path='index.html', before_sha256=before, after_sha256=sha256(html_path), font_links=1))
            for relative, payload in public.items():
                path = context / 'public/_qw3_assets' / relative
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
            dockerfile = ROOT / 'apps/computer_use/vm' / ('osworld-calendar.Dockerfile' if typed else 'osworld-frontend.Dockerfile')
            shutil.copyfile(dockerfile, context / 'Dockerfile')
            (context / '.dockerignore').write_text('node_modules\n.next\ndist\n.git\n.env*\n')
            report['contexts'][app] = dict(path=str(context), patches=patches,
                source_hashes={str(p.relative_to(context)): sha256(p) for p in sorted(context.rglob('*')) if p.is_file()})
        report['complete'] = True
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build(args.inventory, args.output)
    print(json.dumps(dict(complete=report['complete'], contexts=len(report['contexts']), urls=report['frozen_url_count'])))
