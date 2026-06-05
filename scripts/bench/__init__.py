"""qw3 vs llama.cpp benchmark suite.

DEPRECATED for routine qw3-only benchmarking — prefer scripts/bench2/.

bench2 is the clean layer: qw3 only runs an OpenAI-compatible server (you
launch it), and Python does ALL timing via pure wall-clock over HTTP. It never
self-launches a server and never parses stderr.

This older `bench/` layer is KEPT because it still does things bench2 does not:
  - llama.cpp side-by-side comparison (llama_runner.py, build_llama_server.py)
  - 3D parameter sweep orchestration (orchestrator.py, run_bench.py)
  - rich HTML report (report.py)
  - self-launch + stderr scraping of qw3 (qw3_runner.py)

Use bench2/ for fast qw3-vs-qw3 (e.g. fp8-vs-fp16) wall-clock checks with a
text/markdown report. Use this suite only when you need the llama.cpp
comparison or the HTML output.
"""
