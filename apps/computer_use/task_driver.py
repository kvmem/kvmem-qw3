"""Run one GUI task to a recorded terminal state; grading remains on the host.

Ordinary task failures are data, not driver crashes. An infrastructure error
sets queue_may_continue=false, so a campaign can preserve evidence and halt.
This file contains no expected answers or evaluator implementation.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

import aiohttp


def terminal(state):
    status=state.get("status")
    if status in ("completed","failed","paused","stopped"):
        if not state.get("ready") or state.get("evidence_error"):
            return "infrastructure_error"
        if status=="paused":
            used,limit=state.get("tool_calls_used"),state.get("tool_call_limit")
            if type(used) is int and type(limit) is int and used>=limit>0:
                return "tool_budget_exhausted"
            return "paused_uncertain" if state.get("uncertain_actions") else "paused"
        return {"completed":"agent_completed","failed":"agent_failed","stopped":"external_stop"}[status]
    if status not in ("starting","running","verifying"):
        raise ValueError("Unexpected task state: "+str(status))
    return None


class ReviewMailbox:
    """Public event handoff to the host; contains no grader or expected answer.

    Each numbered directory is exclusive. A lost host or follow-up reply is
    never retried automatically. The enclosing task deadline includes waits.
    """
    def __init__(self, path, batches):
        if type(batches) is not int or not 1 <= batches <= 10:
            raise ValueError('Review batches must be 1..10')
        self.path, self.batches = Path(path), batches
        self.path.mkdir(parents=True, exist_ok=False)

    async def __call__(self, state, batch):
        if batch > self.batches:
            return None
        directory = self.path / str(batch)
        directory.mkdir(exist_ok=False)
        checkpoint = {'batch': batch, 'state': state, 'guest_checkpoint_monotonic_ns': time.monotonic_ns(),
                      'guest_checkpoint_unix_ns': time.time_ns()}
        temporary = directory / 'checkpoint.tmp'
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(checkpoint, handle, ensure_ascii=False, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        temporary.rename(directory / 'checkpoint.json')
        response = directory / 'followup.json'
        while not response.is_file():
            if (directory / 'abort.json').is_file():
                raise RuntimeError('Host public-event delivery failed; inspect the host receipt')
            await asyncio.sleep(.2)
        if (directory / 'abort.json').is_file():
            raise RuntimeError('Host public-event delivery was aborted')
        data = json.loads(response.read_text(encoding='utf-8-sig'))
        if data.get('batch') != batch or set(data) != {'batch', 'request'}:
            raise ValueError('Host follow-up belongs to another checkpoint')
        return data['request']


async def run_task(app, prompt, output, timeout, poll_interval=.5, request_timeout=10, on_completed=None):
    if not isinstance(prompt,str) or not prompt.strip() or timeout<=0:
        raise ValueError("Task needs a prompt and positive deadline")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/"prompt.txt").write_text(prompt,encoding="utf-8")
    result={"kind":"native_gui_task_terminal","outcome":"infrastructure_error","graded":False,
        "queue_may_continue":False,"timeout_seconds":timeout,"host_or_guest_clock":"local_monotonic",
        "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"started_unix_ns":time.time_ns()}
    started=time.monotonic();owns_task=False;state=None;cancelled=False;deadline_scope=None
    completed_reviews=[];followup_ids=set();last_review_tools=-1
    events=(output/"states.jsonl").open("w",encoding="utf-8")
    async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=request_timeout)) as client:
        async def api(route,body=None):
            async with client.request("POST" if body is not None else "GET",app.rstrip("/")+route,json=body) as response:
                data=await response.json()
                if response.status!=200:raise RuntimeError(f"Application HTTP {response.status}: {data}")
                return data

        def record(current):
            # Keep exact task-state evidence, including uncertainty and any
            # turn_error. Polling this API does not request a desktop observe.
            events.write(json.dumps({"elapsed_seconds":time.monotonic()-started,"state":current},ensure_ascii=False)+"\n")
            events.flush()

        try:
            before=await api("/api/state")
            if not before.get("ready") or before.get("status") not in ("idle","completed","failed","stopped"):
                raise RuntimeError("Application is not ready for a new owned task")
            # The native launcher creates a fresh, task-owned application. If
            # this request loses its reply, halt it before allowing the next VM.
            owns_task=True
            deadline_scope=asyncio.timeout(timeout)
            async with deadline_scope:
                state=await api("/api/task",{"prompt":prompt})
                result["task_id"]=state["id"]
                while True:
                    record(state)
                    if state.get("id")!=result["task_id"]:
                        owns_task=False
                        raise RuntimeError("Application task identity changed")
                    outcome=terminal(state)
                    if outcome:
                        if outcome == 'agent_completed' and on_completed is not None:
                            used, limit = state.get('tool_calls_used'), state.get('tool_call_limit')
                            if (state.get('turn_status') != 'completed' or not state.get('desktop_paused')
                                    or state.get('uncertain_actions') or not state.get('thread_id') or not state.get('turn_id')
                                    or type(used) is not int or type(limit) is not int or not 0 <= used <= limit
                                    or used < last_review_tools
                                    or completed_reviews and limit != completed_reviews[0]['tool_call_limit']
                                    or state['turn_id'] in {r['turn_id'] for r in completed_reviews}
                                    or completed_reviews and state['thread_id'] != completed_reviews[0]['thread_id']):
                                raise RuntimeError('Review checkpoint is not a new verified turn in the same task')
                            last_review_tools = used
                            completed_reviews.append({'turn_id':state['turn_id'], 'thread_id':state['thread_id'],
                                'tool_calls_used':used, 'tool_call_limit':limit,
                                'completed_local_monotonic_ns':time.monotonic_ns()})
                            result['completed_reviews'] = completed_reviews
                            if used >= limit:
                                result['outcome'] = 'tool_budget_exhausted'; break
                            followup = await on_completed(dict(state), len(completed_reviews))
                            if followup is not None:
                                if (not isinstance(followup, dict)
                                        or set(followup) != {'task_id','completed_turn_id','event_id','prompt'}
                                        or followup.get('task_id') != result['task_id']
                                        or followup.get('completed_turn_id') != state['turn_id']
                                        or not isinstance(followup.get('event_id'), str) or not followup['event_id']
                                        or followup['event_id'] in followup_ids
                                        or not isinstance(followup.get('prompt'), str) or not followup['prompt'].strip()):
                                    raise RuntimeError('Invalid or duplicate review follow-up')
                                followup_ids.add(followup['event_id'])
                                completed_reviews[-1]['event_id'] = followup['event_id']
                                completed_reviews[-1]['followup_sent_local_monotonic_ns'] = time.monotonic_ns()
                                state = await api('/api/followup', followup)
                                continue
                        result["outcome"]=outcome;break
                    await asyncio.sleep(poll_interval)
                    state=await api("/api/state")
        except TimeoutError:
            # A short HTTP request timeout is infrastructure; only expiry of
            # the overall task deadline is the task's time-limit outcome.
            result["outcome"]="time_limit" if deadline_scope is not None and deadline_scope.expired() else "infrastructure_error"
            result["error"]="Task deadline exceeded" if result["outcome"]=="time_limit" else "Application request timed out"
        except asyncio.CancelledError:
            cancelled=True;result["outcome"]="interrupted"
        except Exception as exc:
            result["error"]=repr(exc)
        finally:
            if state is not None:result["terminal_observation"]=state
            if owns_task:
                try:
                    if not state or not state.get("desktop_paused") or result["outcome"] in ("time_limit","infrastructure_error","interrupted"):
                        stopped=await api("/api/control",{"command":"stop"})
                        result["cleanup_state"]=stopped
                        if not stopped.get("desktop_paused") or stopped.get("status")!="stopped":
                            raise RuntimeError("Task stop was not acknowledged")
                    result["desktop_stopped"]=True
                except Exception as exc:
                    result["prior_outcome"]=result["outcome"]
                    result["outcome"]="infrastructure_error";result["cleanup_error"]=repr(exc)
            result["queue_may_continue"]=(result["outcome"] in
                ("agent_completed","agent_failed","paused","paused_uncertain","tool_budget_exhausted","time_limit")
                and result.get("desktop_stopped",False))
            result["elapsed_seconds"]=time.monotonic()-started
            result["finished_unix_ns"]=time.time_ns()
            events.close()
            (output/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    if cancelled:raise asyncio.CancelledError
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app",default="http://127.0.0.1:18083")
    parser.add_argument("--prompt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--timeout",type=float,required=True)
    parser.add_argument('--review-mailbox',type=Path)
    parser.add_argument('--review-batches',type=int,default=0)
    args=parser.parse_args()
    if bool(args.review_mailbox) != bool(args.review_batches):
        parser.error('Review mailbox and batch count must be supplied together')
    mailbox=ReviewMailbox(args.review_mailbox,args.review_batches) if args.review_mailbox else None
    result=asyncio.run(run_task(args.app,args.prompt.read_text(encoding="utf-8-sig"),args.output,args.timeout,on_completed=mailbox))
    print(json.dumps({k:result[k] for k in ("outcome","graded","queue_may_continue","elapsed_seconds")}))
    raise SystemExit(0 if result["queue_may_continue"] else 2)
