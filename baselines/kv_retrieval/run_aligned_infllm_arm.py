"""Reuse the original owned-resource full-task harness for InfLLM variants only."""
import os
from pathlib import Path
import run_pilot_arm as arm

ROOT=Path(__file__).resolve().parent


def command(method,plan,out,port):
    if method!='infllm': raise ValueError('Only new InfLLM arms are authorized')
    setting=plan['methods']['infllm']
    return [str(ROOT/'.venv/bin/python'),str(ROOT/'serve_aligned_infllm.py'),
        '--model',str(ROOT.parents[1]/'models/Qwen3.8-27B-BF16'),
        '--gguf',str(ROOT.parents[1]/'models/Qwen3.8-27B-Q8_0.gguf'),
        '--out',str(out/'server'),'--port',str(port),
        '--max-context',str(setting['context_tokens']),
        '--max-gpu-gib',str(setting['allocator_cap_gib']),
        '--infllm-config',str(ROOT/setting['config']),
        '--temperature',str(plan['temperature']),'--top-p',str(plan['top_p']),
        '--top-k',str(plan['top_k']),'--seed',str(plan['seed']),
        '--prefill-chunk',str(plan['prefill_chunk']),
        '--max-tokens',str(plan['max_output_tokens']),
        '--thinking-budget',str(plan['thinking_budget'])]


def main():
    arm.server_command=command
    # Keep the same requested model capability marker as the reused Q8 runs.
    # The old harness assigns a BF16 label after clean_environment returns, so
    # intercept only the environment passed to its task supervisor.
    popen=arm.subprocess.Popen
    def start(cmd,*args,**kwargs):
        env=kwargs.get('env')
        if env and 'QWEN38_PRECISION_LABEL' in env:
            env=dict(env)
            env.update(QWEN38_PRECISION_LABEL='Q8_0 weights / FP8 KV',
                       QWEN38_CLAUDE_MODEL_ID='Qwen3.8-27B[3m]',
                       QWEN38_CLAUDE_AGENT_MEMORY='8g',
                       CLAUDE_AGENT_RUNNER=str(ROOT/'alignment/harness/container_claude_agent.sh'))
            kwargs['env']=env
        return popen(cmd,*args,**kwargs)
    arm.subprocess.Popen=start
    arm.main()


if __name__=='__main__': main()
