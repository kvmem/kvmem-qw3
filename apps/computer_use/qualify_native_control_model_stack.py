"""Exercise a real private VM/gateway/Codex client without any model task.

No task POST, device action, engine launch or cloud request is performed. This
qualifies isolation, observation, recording and retirement, not task success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import build_opener, ProxyHandler

import native_control_model_stack as stack


def perform(args):
    service = stack.NativeStack(args.study, args.trial_id, args.output, args.port_base)
    output = service.output
    source = Path(__file__)
    shutil.copyfile(source, output / 'source' / source.name)
    result = dict(kind='real_private_native_control_model_stack_qualification', passed=False,
        started_at=time.time(), trial_id=args.trial_id, stack_file=str(output / 'stack.json'),
        producer=stack.adapter.owned.process_identity(__import__('os').getpid()),
        source_sha256={source.name:stack.adapter.owned.digest(source)},
        checks={}, model_tasks_started=0, model_inference_requests=0, engine_starts=0,
        gui_inputs=0, business_success_claimed=False, engine_binding_qualified=False)
    recorder = None
    def save(**values):
        result.update(values)
        stack.adapter.queue.save(output / 'result.json', result)
    save()
    try:
        state = service.start()
        result['checks']['real_client_ready_with_original_idle_profile'] = stack.idle_profile(state)
        first = state['observation']
        result['checks']['client_observes_same_device_and_gateway_session'] = (
            first['device_id'] == service.device_id
            and first['session_id'] == service.record['gateway_initial_observation']['session_id'])
        result['checks']['only_declared_private_device_visible'] = state['devices']['devices'] == [
            dict(id='linux-vm', label='Private Linux', platform='linux')]
        result['checks']['original_explicit_registry_unchanged'] = (
            stack.adapter.owned.digest(service.record['registry']['path']) == service.record['registry']['sha256'])
        refreshed = stack.adapter.pilot.request(service.client['endpoint'], '/api/observe', {})
        stack.adapter.queue.save(output / 'refreshed-client-state.json', refreshed)
        latest = refreshed['observation']
        result['checks']['readonly_refresh_preserves_identity_and_pause'] = (
            stack.idle_profile(refreshed) and latest['desktop_id'] == first['desktop_id']
            and latest['observation_id'] != first['observation_id'])
        with build_opener(ProxyHandler({})).open(service.client['endpoint'] + '/api/screenshot', timeout=10) as response:
            png = response.read()
        (output / 'observed-screen.png').write_bytes(png)
        result['checks']['actual_screenshot_matches_observation_hash'] = hashlib.sha256(png).hexdigest() == latest['sha256']
        result['checks']['no_model_or_action_requested'] = (
            not list((Path(service.client['runtime']) / 'inference').glob('**/request*'))
            and stack.adapter.queue.ledger_terminal(Path(service.gateway['runtime'])) == {})
        result['checks']['real_codex_app_server_registered'] = any(
            'app-server' in row['command'] and row['cwd'] == str(Path(service.client['runtime']) / 'codex')
            for row in service.record['client_descendants'])
        result['actual_model_context_without_engine_binding'] = state['model_context_window']
        # Recording uses its own read-only console, never client observe calls.
        video_dir = output / 'recording'
        video_dir.mkdir()
        recorder = stack.adapter.PrivateRecorder(service.transport, video_dir, service.record['vm']['container_id'])
        recorder.start()
        time.sleep(3)
        receipt = recorder.finish()
        import av
        movie = video_dir / 'continuous.mp4'
        with av.open(str(movie)) as media:
            frames = sum(1 for _ in media.decode(video=0))
        result['checks']['actual_console_movie_complete_and_decodable'] = receipt['complete'] is True and frames > 0
        result['checks']['same_recorder_worker_retired'] = stack.retired(recorder.native_worker_identity)
        result['video'] = dict(path=str(movie), sha256=stack.adapter.owned.digest(movie), decoded_frames=frames)
        result['recorder_identity'] = recorder.native_worker_identity
        save()
    except BaseException as error:
        save(error_kind=type(error).__name__, error=str(error))
    finally:
        if recorder is not None and not recorder.collected:
            try:
                recorder.finish('interrupted.mp4')
            except Exception as error:
                save(recording_cleanup_error=str(error))
        try:
            service.close()
            result['checks']['vm_and_services_retired'] = (
                service.record.get('vm_retired') is True
                and all(stack.retired(row['identity']) for row in service.record['services'].values()))
            result['checks']['codex_descendants_retired'] = all(stack.retired(row)
                for row in service.record.get('client_descendants', []))
            result['checks']['readonly_seed_metadata_unchanged'] = service.record.get('seed_metadata_unchanged') is True
        except Exception as error:
            save(cleanup_error=str(error), live_work_preserved=True)
        save(passed=bool(result['checks']) and all(result['checks'].values())
             and not any(key.endswith('error') for key in result), finished_at=time.time(),
             stack_sha256=stack.adapter.owned.digest(output / 'stack.json'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--trial-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port-base', type=int, required=True)
    args = parser.parse_args()
    result = perform(args)
    print(json.dumps({key:result.get(key) for key in ('passed', 'checks', 'error', 'cleanup_error', 'video')}, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
