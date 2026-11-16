# LouisKV executable-baseline audit, 2026-09-12

LouisKV is relevant because it reduces retrieval frequency using semantic
boundaries and distinguishes input-history and generated-history organization.
The [official ICLR paper page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b241c515433caae3051266668d808b7-Abstract-Conference.html)
describes these mechanisms.

The [arXiv reproducibility statement](https://arxiv.org/html/2510.11292v1#S8)
states an intention to release source code. The
[coauthor's publication page](https://wangye0523.github.io/) links the LouisKV
paper but has no code link for that entry. A direct, proxy-free GitHub
repository-name search returned zero matches on this audit date; its raw
response summary is `reports/louiskv_code_search_20260912.json`.

Audit status: no verified official source repository was located in these
checks. There is no executable LouisKV result in this comparison. Replacing it
with a KVMem query-similarity heuristic would not constitute a LouisKV baseline.
FlexiCache remains the source-available candidate for a low-frequency retrieval
comparison, with its original component validation under way.
