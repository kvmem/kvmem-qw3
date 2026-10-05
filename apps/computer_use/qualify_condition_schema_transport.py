"""Real pinned Codex tool transport, with a loopback fixture backend only.

No QW3 inference or device exists in this qualification. The actual private MCP
proxy advertises platform catalogs; any attempted tool dispatch is refused.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time

from aiohttp import web

import condition_schema_candidate as candidate
import kvmem_osworld_queue as owned


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])


def descendants(parent):
    actual = owned.process_identity(parent['pid'])
    if not owned.same_process(parent, actual):raise RuntimeError('Actual Codex parent identity changed')
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name) <= 1:continue
        try:
            current = owned.process_identity(int(path.name))
            if current and current['uid'] == os.getuid() and current['state'] != 'Z':
                ppid = int((path/'stat').read_text().rpartition(')')[2].split()[1])
                processes[current['pid']] = (ppid, current)
        except (OSError, ValueError, UnicodeError):continue
    selected = {parent['pid']}
    while True:
        added = {pid for pid, (ppid, _) in processes.items() if ppid in selected} - selected
        if not added:break
        selected.update(added)
    return [processes[pid][1] for pid in sorted(selected - {parent['pid']})]


async def confirm_generation_retired(identity):
    end = time.monotonic() + 30
    while True:
        current = owned.process_identity(identity['pid'])
        if (current is None or current['state'] == 'Z'
                or current['start_ticks'] != identity['start_ticks']):
            return dict(pid=identity['pid'], original_generation_retired=True,
                actual_final_identity=current)
        if time.monotonic() >= end:
            raise RuntimeError('Original qualification generation remains present; no replacement started')
        await asyncio.sleep(.2)


async def qualify(build_path, output):
    build_path, output = Path(build_path).resolve(), Path(output).resolve()
    build = json.loads(build_path.read_text())
    if (build.get('kind') != 'isolated_native_button_inline_condition_wire_candidate'
            or build['base_build_sha256'] != candidate.BASE_BUILD_SHA
            or build['recipe_sha256'] != candidate.digest(candidate.__file__)
            or build['helper_sha256'] != candidate.digest(Path(__file__).with_name('condition_schema_wire.py'))
            or build['changed_package_files'] != ['qw3_computer/protocol.py', 'qw3_computer/condition_schema_wire.py']):
        raise ValueError('Require the unchanged isolated repair candidate')
    framework = build_path.parent/'framework'
    for name, digest in build['candidate_files_sha256'].items():
        if candidate.digest(framework/name) != digest:raise ValueError('Candidate source changed: ' + name)
    if not output.is_relative_to(candidate.RUNTIME) or output.is_relative_to(build_path.parent):
        raise ValueError('Use a new private qualification runtime')
    if 'qw3_computer' in sys.modules:raise ValueError('Private package must be selected before imports')
    sys.path.insert(0, str(framework))
    from qw3_computer.bridge import create_app
    from qw3_computer.codex import CodexClient, write_config, PINNED_VERSION
    from qw3_computer.protocol import convert_request
    from qw3_computer.tool_schema import WORKFLOW_TOOLS
    declared = WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sources = {}
    for name in ['condition_schema_wire.py', 'condition_schema_candidate.py', 'qualify_condition_schema_transport.py']:
        source = Path(__file__).with_name(name);shutil.copyfile(source, output/name);sources[name] = candidate.digest(source)
    record = dict(kind='actual_pinned_codex_private_inline_schema_transport_qualification', started_at=time.time(),
        passed=False, source_sha256=sources, build=str(build_path), build_sha256=candidate.digest(build_path),
        actual_codex_version=PINNED_VERSION, real_model=False, model_inference_requests=0, device_inputs=0,
        fixture_backend=True, fixture_backend_requests=0, tool_dispatch_attempts=0, business_improvement_proven=False,
        frozen_sources_modified=False, cases=[])
    save(output/'result.json', record)
    received, raw_requests, dispatches = [], [], []

    async def refused_tool(request):
        body = await request.json()
        dispatches.append(body.get('name'))
        return web.json_response({'error': 'Transport qualification has no device or input executor'}, status=403)

    async def backend(request):
        body = await request.json();received.append(body)
        save(output/('fixture-backend-'+str(len(received))+'.json'), body)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        chunk = {'choices': [{'index': 0, 'delta': {'content': 'Transport check completed.'}, 'finish_reason': None}]}
        await response.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
        chunk = {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}
        await response.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
        await response.write(b'data: [DONE]\n\n');await response.write_eof()
        return response

    fixture = web.Application();fixture.router.add_post('/v1/chat/completions', backend)
    fixture.router.add_post('/internal/tool', refused_tool)
    fixture_runner = bridge_runner = None
    try:
        fixture_runner, fixture_url = await serve(fixture)
        bridge = create_app(fixture_url+'/v1', max_tokens=16384, trace_dir=output/'inference', allow_thinking=True,
            allowed_tools={'mcp__computer__observe', 'mcp__computer__act', 'mcp__computer__workflow'})
        async def capture_raw(request, handler):
            if request.path == '/v1/responses':
                raw_requests.append(await request.json())
                save(output/('responses-raw-'+str(len(raw_requests))+'.json'), raw_requests[-1])
            return await handler(request)
        bridge.middlewares.append(web.middleware(capture_raw))
        bridge_runner, bridge_url = await serve(bridge)
        for platform in ('windows', 'linux', 'android'):
            for workflow in (False, True):
                folder = output/(platform + ('-verified' if workflow else '-plain'));folder.mkdir()
                token = folder/'local-token';token.write_text(secrets.token_urlsafe(32));token.chmod(0o600)
                args = [str(framework/'qw3_computer/mcp_proxy.py'), '--endpoint', fixture_url,
                    '--token-file', str(token), '--platform', platform] + (['--workflow'] if workflow else [])
                config = write_config(folder/'codex', bridge_url+'/v1', mcp_command=sys.executable,
                    mcp_args=args, thinking=True, workflow=workflow, context_window=524288)
                client = CodexClient(config)
                identity, children, retirements = None, [], []
                row = dict(platform=platform, workflow=workflow, passed=False)
                record['cases'].append(row);save(output/'result.json', record)
                before, raw_before = len(received), len(raw_requests)
                try:
                    await client.start()
                    identity = owned.process_identity(client.proc.pid)
                    if (not identity or identity['uid'] != os.getuid() or identity['cwd'] != str(config)
                            or identity['state'] == 'Z' or 'app-server' not in identity['command']):
                        raise RuntimeError('Actual private Codex identity not confirmed')
                    row['actual_codex_identity'] = identity;save(output/'result.json', record)
                    thread = await client.new_thread()
                    turn = await client.turn(thread, 'Read-only tool transport qualification. Return a brief completion message. Do not call any tool.')
                    row.update(thread=thread, turn=turn)
                    end = time.monotonic() + 90
                    with (folder/'codex-events.jsonl').open('x') as log:
                        while True:
                            # The same turn remains the authority; an observation
                            # timeout never causes another turn or process start.
                            try:event = await asyncio.wait_for(client.events.get(), 10)
                            except TimeoutError:
                                if client.proc.returncode is not None:raise RuntimeError('Private Codex exited before terminal turn')
                                if time.monotonic() >= end:raise RuntimeError('Transport turn remained nonterminal')
                                continue
                            log.write(json.dumps(event)+'\n');log.flush()
                            if event.get('method') == 'turn/completed' and event['params']['turn']['id'] == turn:
                                if event['params']['turn']['status'] != 'completed':raise RuntimeError('Actual Codex transport turn failed')
                                break
                            if event.get('method') == 'app/disconnected':raise RuntimeError('Private Codex disconnected')
                    if len(received)-before != 1 or len(raw_requests)-raw_before != 1 or dispatches:
                        raise RuntimeError('Require exactly one fixture response and zero tool dispatches')
                    raw, forwarded = raw_requests[-1], received[-1]
                    converted = convert_request(raw, 'qw3', 16384, allow_thinking=True)
                    excluded = set(converted.names) - {'mcp__computer__observe', 'mcp__computer__act', 'mcp__computer__workflow'}
                    if not excluded <= {'list_mcp_resources', 'list_mcp_resource_templates', 'read_mcp_resource', 'request_user_input'}:
                        raise RuntimeError('Unexpected tool offered by actual Codex')
                    converted.body['tools'] = [tool for tool in converted.body['tools'] if tool['function']['name'] not in excluded]
                    row['tools_excluded_by_unchanged_bridge_policy'] = sorted(excluded)
                    if converted.body != forwarded:raise RuntimeError('Actual forwarded body differs from qualified private conversion')
                    tools = {t['function']['name']:t['function']['parameters'] for t in forwarded['tools']}
                    expected = {'mcp__computer__observe', 'mcp__computer__act'} | ({'mcp__computer__workflow'} if workflow else set())
                    if set(tools) != expected:raise RuntimeError('Actual tool permissions differ from selected mode')
                    def flat_tools(request):
                        result = {}
                        for tool in request['tools']:
                            if tool['type'] == 'namespace':
                                result.update({tool['name']+'__'+t['name']:t['parameters'] for t in tool['tools']})
                            else:result[tool['name']] = tool['parameters']
                        return result
                    original = flat_tools(raw)
                    actual = tools['mcp__computer__act']
                    if workflow:
                        incoming = original['mcp__computer__act']['properties']['step']['properties']['conditions']
                        if actual['properties']['step']['properties']['conditions'] != declared:raise RuntimeError('Inline condition schema was not restored')
                        if original['mcp__computer__workflow'] != tools['mcp__computer__workflow']:raise RuntimeError('Different workflow contract was modified')
                        row['incoming_inline_items_opaque'] = incoming['items'] == {}
                        row['incoming_inline_shape'] = ('opaque' if incoming['items'] == {} else
                            'declared' if incoming['items'] == declared['items'] else 'known-codex-normalization')
                        row['restored_inline_variants'] = len(declared['items']['oneOf'])
                    elif actual != original['mcp__computer__act']:
                        raise RuntimeError('Plain mode gained or changed a step contract')
                    if forwarded['max_tokens'] != 16384 or forwarded['enable_thinking'] is not True:
                        raise RuntimeError('Actual model profile changed')
                    save(folder/'responses-request.json', raw);save(folder/'fixture-backend-request.json', forwarded)
                    row.update(passed=True, actual_request_sha256=candidate.digest(folder/'responses-request.json'),
                        forwarded_request_sha256=candidate.digest(folder/'fixture-backend-request.json'),
                        tool_names=sorted(tools), tool_dispatches=0)
                finally:
                    if identity is None and client.proc is not None:
                        identity = owned.process_identity(client.proc.pid)
                        if identity:row['actual_codex_identity'] = identity
                    if identity:
                        current = owned.process_identity(identity['pid'])
                        if owned.same_process(identity, current) and current['state'] != 'Z':children = descendants(identity)
                        row['actual_descendants'] = children;save(output/'result.json', record)
                        retirements.append(await asyncio.to_thread(owned.retire_registered, identity, 30))
                        await client.proc.wait()
                        # Stop the parent before its MCP children so it cannot
                        # restart a killed child while shutdown is pending.
                        for child in reversed(children):
                            retirements.append(await asyncio.to_thread(owned.retire_registered, child, 30))
                    await client.close()
                    row['retirements'] = retirements
                    row['generation_retirement_observations'] = []
                    for process in [identity]+children:
                        if process:row['generation_retirement_observations'].append(await confirm_generation_retired(process))
                    row['all_registered_processes_retired'] = bool(identity) and all(
                        r['original_generation_retired'] for r in row['generation_retirement_observations'])
                    save(output/'result.json', record)
        record['passed'] = len(record['cases']) == 6 and all(r['passed'] and r['all_registered_processes_retired'] for r in record['cases'])
    except BaseException as error:
        record.update(error_kind=type(error).__name__, error=str(error));raise
    finally:
        if bridge_runner:await bridge_runner.cleanup()
        if fixture_runner:await fixture_runner.cleanup()
        record.update(finished_at=time.time(), fixture_backend_requests=len(received), tool_dispatch_attempts=len(dispatches))
        save(output/'result.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, type=Path);parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(qualify(args.build, args.output))
    print(json.dumps({k:result[k] for k in ['passed', 'fixture_backend_requests', 'tool_dispatch_attempts', 'model_inference_requests', 'device_inputs']}))
    if not result['passed']:raise SystemExit(1)
