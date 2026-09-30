"""Worker for dflash_score_public.py. Run only inside its filesystem sandbox."""
import json
import os
import resource
import sys

job = json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
seconds = 8 if job["kind"] == "math" else 3
resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
result = {"nonce": job["nonce"], "status": "failed", "passed": False}
try:
    if job["kind"] == "math":
        from math_verify import parse, verify

        gold = parse(job["gold"], fallback_mode="no_fallback", parsing_timeout=1)
        if not gold:
            result.update(status="reference_parse_failure")
        else:
            result.update(status="scored", gold_parsed=[str(x) for x in gold])
            for key in ("whole", "final"):
                prediction = parse(job[key], fallback_mode="no_fallback", parsing_timeout=1) if job[key] else []
                result[key + "_parsed"] = [str(x) for x in prediction]
                result[key + "_correct"] = bool(prediction and verify(gold, prediction, timeout_seconds=1))
            result["passed"] = result["final_correct"]
    elif job["kind"] == "code":
        namespace = {"__name__": "candidate"}
        exec(compile(job.get("prelude", ""), "prompt_imports", "exec"), namespace)
        exec(compile(job["code"], "candidate", "exec"), namespace)
        exec(compile(job["tests"], "official_tests", "exec"), namespace)
        result.update(status="scored", passed=True)
    else:
        raise ValueError("unknown worker kind")
except BaseException as error:
    result.update(status="candidate_failure", error=type(error).__name__, detail=str(error)[:400])
os.write(1, ("\n" + json.dumps(result) + "\n").encode())
