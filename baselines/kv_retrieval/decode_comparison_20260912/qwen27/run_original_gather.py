"""Calibration control for the Qwen port's non-deduplicated GQA gather."""
import run_hf

run_hf.OptimizedBackend = run_hf.Backend
original_save = run_hf.save


def save(path, report):
    report['infinigen_gather'] = 'original per-query-head gather; no deduplication'
    report['wrapper_sha256'] = run_hf.digest(__file__)
    original_save(path, report)


run_hf.save = save
if __name__ == '__main__':
    run_hf.main()
