"""Run one GUI task to a recorded terminal state; grading remains on the host.

Ordinary task failures are data, not driver crashes. An infrastructure error
sets queue_may_continue=false, so a campaign can preserve evidence and halt.
This file contains no expected answers or evaluator implementation.
"""
import argparse
import asyncio
import hashlib
import json
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


async def run_task(app, prompt, output, timeout, poll_interval=.5, request_timeout=10):
    if not isinstance(prompt,str) or not prompt.strip() or timeout<=0:
        raise ValueError("Task needs a prompt and positive deadline")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/"prompt.txt").write_text(prompt,encoding="utf-8")
    result={"kind":"native_gui_task_terminal","outcome":"infrastructure_error","graded":False,
        "queue_may_continue":False,"timeout_seconds":timeout,"host_or_guest_clock":"local_monotonic",
        "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),"started_unix_ns":time.time_ns()}
    started=time.monotonic();owns_task=False;state=None;cancelled=False;deadline_scope=None
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
    args=parser.parse_args()
    result=asyncio.run(run_task(args.app,args.prompt.read_text(encoding="utf-8-sig"),args.output,args.timeout))
    print(json.dumps({k:result[k] for k in ("outcome","graded","queue_may_continue","elapsed_seconds")}))
    raise SystemExit(0 if result["queue_may_continue"] else 2)
