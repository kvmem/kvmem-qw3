"""Actual pinned Codex diagnostics with a local deterministic backend.

No QW3 inference or real desktop input. A complete function call followed by a
length termination must never escape to the MCP executor, either.
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import secrets
import sys
import time

from aiohttp import web

import kvmem_osworld_queue as owned
from diagnostic_priority_checks import digest, save
from qualify_condition_schema_transport import descendants, confirm_generation_retired, serve


async def run(args):
    output = args.output.resolve()
    framework = args.framework.resolve()
    runtime = Path(__file__).resolve().parent / '.runtime'
    if not output.is_relative_to(runtime) or output.is_relative_to(framework):
        raise ValueError('Use a separate private diagnostic runtime')
    if args.mode == 'wire':
        manifest = json.loads((framework.parent / 'build.json').read_text())
        files = manifest['candidate_files_sha256']
    else:
        manifest = json.loads((framework.parent / 'build.json').read_text())
        files = {'qw3_computer/' + name: sha for name, sha in manifest['framework_files'][framework.name].items()}
    for name, sha in files.items():
        if digest(framework / name) != sha:
            raise ValueError('Selected declared framework changed: ' + name)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    sys.path.insert(0, str(framework))
    from qw3_computer.bridge import create_app
    from qw3_computer.codex import CodexClient, PINNED_VERSION, write_config
    from qw3_computer.tool_schema import WORKFLOW_TOOLS
    calls, raw, dispatched = [], [], []
    result = dict(kind='focused_actual_codex_' + args.mode, started_at=time.time(), passed=False,
        real_qw3_requests=0, real_device_inputs=0, fixture_backend=True,
        actual_codex_version=PINNED_VERSION, source_sha256=digest(__file__), checks={})
    checks = result['checks']
    fixture_runner = bridge_runner = controller_runner = None
    controller = client = identity = None

    async def backend(request):
        body = await request.json()
        calls.append(body)
        save(output / ('fixture-request-' + str(len(calls)) + '.json'), body)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        if args.mode == 'wire':
            delta, finish = {'content': 'Read-only transport check completed.'}, 'stop'
        elif args.mode == 'cap-reasoning':
            delta, finish = {'reasoning_content': 'Need to inspect the field before continuing.'}, 'length'
        else:
            delta = {'tool_calls': [{'index': 0, 'id': 'truncated-fixture', 'type': 'function',
                'function': {'name': 'mcp__computer__workflow', 'arguments': json.dumps({
                    'operation': 'request_takeover', 'reason': 'This truncated tool must not execute'})}}]}
            finish = 'length'
        frames = [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                  {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]},
                  {'choices': [], 'usage': {'prompt_tokens': 100, 'completion_tokens': 16384 if finish == 'length' else 7,
                                          'total_tokens': 16484 if finish == 'length' else 107}}]
        for frame in frames:
            await response.write(('data: ' + json.dumps(frame) + '\n\n').encode())
        await response.write(b'data: [DONE]\n\n')
        return response

    async def refused_tool(request):
        dispatched.append(await request.json())
        return web.json_response({'error': 'No real device exists in this diagnostic'}, status=403)

    fixture = web.Application()
    fixture.router.add_post('/v1/chat/completions', backend)
    fixture.router.add_post('/internal/tool', refused_tool)
    try:
        fixture_runner, fixture_url = await serve(fixture)
        if args.mode == 'wire':
            bridge = create_app(fixture_url + '/v1', max_tokens=16384, trace_dir=output / 'inference',
                allow_thinking=True, allowed_tools={'mcp__computer__observe', 'mcp__computer__act',
                                                  'mcp__computer__workflow', 'mcp__computer__history'})
            async def capture(request, handler):
                if request.path == '/v1/responses':
                    raw.append(await request.json())
                    save(output / 'actual-raw-response-request.json', raw[-1])
                return await handler(request)
            bridge.middlewares.append(web.middleware(capture))
            bridge_runner, bridge_url = await serve(bridge)
            token = output / 'fixture-token'
            token.write_text(secrets.token_urlsafe(32)); token.chmod(0o600)
            config = write_config(output / 'codex', bridge_url + '/v1', mcp_command=sys.executable,
                mcp_args=[str(framework / 'qw3_computer/mcp_proxy.py'), '--endpoint', fixture_url,
                          '--token-file', str(token), '--platform', 'linux', '--workflow', '--history'],
                thinking=True, workflow=True, context_window=524288)
            client = CodexClient(config)
            await client.start()
            identity = owned.process_identity(client.proc.pid)
            thread = await client.new_thread()
            turn = await client.turn(thread, 'Read-only diagnostic. Return one short sentence; do not call any tools.')
            async with asyncio.timeout(60):
                while True:
                    event = await client.events.get()
                    with (output / 'codex-events.jsonl').open('a') as log:
                        log.write(json.dumps(event) + '\n')
                    if event.get('method') == 'turn/completed' and event['params']['turn']['id'] == turn:
                        checks['actual_turn_completed'] = event['params']['turn']['status'] == 'completed'
                        break
            if len(calls) != 1:
                raise RuntimeError('Actual tool request was rejected before fixture backend; inspect retained Codex events')
            declared = WORKFLOW_TOOLS[0]['inputSchema']['properties']['conditions']
            received = {t['function']['name']: t['function']['parameters'] for t in calls[0]['tools']}
            restored = received['mcp__computer__act']['properties']['step']['properties']['conditions']
            checks['restored_complete_condition_schema'] = restored == declared
            workflow = received['mcp__computer__workflow']['properties']['conditions']
            checks['standalone_workflow_condition_schema_complete'] = workflow == declared
            checks['field_reference_condition_retained'] = any(
                b.get('properties', {}).get('kind', {}).get('const') == 'field_value'
                for b in restored['items']['oneOf'])
            original = {}
            for t in raw[0]['tools']:
                if t['type'] == 'namespace':
                    original.update({t['name'] + '__' + f['name']: f['parameters'] for f in t['tools']})
                else:
                    original[t['name']] = t['parameters']
            incoming = original['mcp__computer__act']['properties']['step']['properties']['conditions']
            result['actual_incoming_inline_items'] = incoming.get('items')
            result['condition_variant_count'] = len(restored['items']['oneOf'])
            save(output / 'actual-raw-response-request.json', raw[0])
            checks['one_backend_request_no_tool_dispatch'] = len(calls) == 1 and not dispatched
        else:
            from PIL import Image
            from qw3_computer.app import Controller, web_app
            from qw3_computer.desktop import Desktop
            class SyntheticDesktop(Desktop):
                def __init__(self):
                    super().__init__('http://127.0.0.1:9', 'unused', output / 'device')
                    self.sequence = 0

                async def start(self):
                    return await self.observe()

                async def observe(self):
                    self.sequence += 1
                    data = io.BytesIO(); Image.new('RGB', (128, 96), (40, 70, 90)).save(data, 'PNG')
                    pixels = data.getvalue()
                    obs = dict(observation_id='obs' + str(self.sequence), desktop_id='synthetic-linux',
                        device_id='fixture', session_id='fixture-session', display_id='primary', width=128, height=96,
                        sha256=hashlib.sha256(pixels).hexdigest(), captured_at=time.time(),
                        foreground=dict(handle='one', process='fixture', title='Fixture visible'),
                        capabilities=dict(platform='linux'), accessibility=dict(status='ok', truncated=False, controls=[]),
                        image=base64.b64encode(pixels).decode())
                    return self.remember(obs) | {'image': obs['image']}

                async def control(self, command, acknowledge_uncertain=False):
                    self.paused = command != 'resume'
                    return dict(paused=self.paused)

                async def request(self, method, route, body=None):
                    if method == 'GET' and route == '/files':
                        return {'files': []}
                    dispatched.append(dict(method=method, route=route))
                    raise AssertionError('Real inputs are forbidden')

            desktop = SyntheticDesktop()
            controller = Controller(output / 'app', desktop, fixture_url + '/v1', 0,
                workflow=True, device_platform='linux', thinking=True, model_output_tokens=16384,
                tool_call_limit=250, output_policy='task')
            controller_runner, url = await serve(web_app(controller))
            controller.port = int(url.rsplit(':', 1)[1])
            await controller.start()
            client = controller.client; identity = owned.process_identity(client.proc.pid)
            await controller.start_task('Output-cap lifecycle diagnostic using synthetic evidence only.')
            async with asyncio.timeout(60):
                while controller.state['status'] in {'starting', 'running', 'verifying'}:
                    await asyncio.sleep(.02)
            state = controller.snapshot()
            save(output / 'terminal-state.json', state)
            checks['codex_reports_output_cap_failure'] = (state['status'] == 'failed' and
                'max_output_tokens' in json.dumps(state.get('turn_error')))
            checks['no_automatic_retry'] = len(calls) == 1
            checks['no_tool_dispatched_even_if_json_complete'] = state['tool_calls_used'] == 0 and not dispatched
            checks['device_paused_and_no_input_receipts'] = desktop.paused and (
                desktop.db.execute('SELECT count(*) FROM actions').fetchone()[0] == 0)
            checks['no_false_completion'] = not state.get('completion_origin')
            result['bounded_decision'] = state.get('bounded_turn_last_decision')
            result['expected_gap'] = 'Existing zero-tool continuation does not admit failed/incomplete turns.'
        checks['actual_codex_identity_confirmed'] = bool(identity and identity['uid'] == __import__('os').getuid())
        result.update(passed=all(checks.values()), backend_requests=len(calls), tool_dispatch_attempts=len(dispatched))
    finally:
        if identity:
            children = descendants(identity)
            result['registered_codex_identity'] = identity
            result['retirements'] = [await asyncio.to_thread(owned.retire_registered, identity, 30)]
            await client.proc.wait()
            for child in reversed(children):
                result['retirements'].append(await asyncio.to_thread(owned.retire_registered, child, 30))
            checks['registered_processes_retired'] = True
            for p in [identity] + children:
                if not (await confirm_generation_retired(p))['original_generation_retired']:
                    checks['registered_processes_retired'] = False
        if controller:
            await controller.close()
        elif client:
            await client.close()
        if controller_runner: await controller_runner.cleanup()
        if bridge_runner: await bridge_runner.cleanup()
        if fixture_runner: await fixture_runner.cleanup()
        result.update(passed=result['passed'] and all(checks.values()), finished_at=time.time())
        save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('wire', 'cap-reasoning', 'cap-tool'), required=True)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({k: result[k] for k in ('kind', 'passed', 'checks', 'backend_requests')}))
    raise SystemExit(0 if result['passed'] else 1)
