# OSWorld-V2 evaluation provenance

This file records the immutable inputs selected for the local QW3 evaluation.

| Component | Pinned value |
| --- | --- |
| QW3 source | `9ae4c31bcf9e050fc2a772d44bf3c531907e23d8` |
| OSWorld-V2 source | tag `v2026.08.08`, commit `d578d2d4e0dc82b43e270fdaa7fa89d9708cd154` |
| OSWorld-GitLab source | commit `8655d651722f4254e59e813de9f68a6732ea525c` |
| Benchmark release | `osworld-v2-2026.08.08` (108 tasks) |
| Gated task repository | tag `v2026.08.08`, commit `3736efa55d9d5dc78f57e873ef78886663e41200` |
| Task manifest | `manifests/task_hashes.json`, SHA-256 `42f8f6f8939b8712997d5891456a575f8a2a5f53465e9e3e6747af5d6efd0915` |
| Gated asset repository | tag `v2026.08.08`, commit `acad110ef3136405f95434b54862bf9066176c2a` (1061 files) |
| Local asset manifest | 1061 files, `4525476196` bytes, SHA-256 `137373379dd5b44335131730e4e04d034aa384c6020b7bf7c3cc096332c9d5df` |
| VM archive | `osworld-v2-ubuntu-x86.before-20260716T053042Z-repo-update.qcow2.zip` |
| VM archive size | `14189763267` bytes |
| VM archive SHA-256 | `eb737ae70b49849e24af407de6a518439a23de05a8497096a948334ce0a909aa` |
| Extracted VM image size | `27402633216` bytes |
| Docker runtime | `happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9` |
| GitLab CE | `gitlab/gitlab-ce@sha256:9c17686c3cd1b72f52282543d2dcf7c96add29b4698ec367bfb1455be8f13ce0`, `1753368468` bytes |
| GitLab Runner | `gitlab/gitlab-runner@sha256:9d3b2b8be462e18176b82eb40a5f54a5ee6eaa2b4f46dcc362c4801aef90030f`, `64927159` bytes |
| GitLab token initializer | `docker@sha256:3f4743208d2338c934d7b8bcfbe1bb54c0b2355c510ad5e0f31c0c4a54bd704e`, `70255738` bytes |
| Hosted benchmark sites | `site.hku.icu` |
| QW3 language model | `models/Qwen3.8-27B-Q8_0.gguf`, `29047086048` bytes |
| QW3 vision model | `models/Qwen3.8-27B-BF16/model.safetensors.index.json`, `112216` bytes |
| Hugging Face endpoint | `https://hf-mirror.com` |

The pinned OSWorld checkout has local integration and measurement patches.
`scripts/python/run_multienv.py` selects the repository's `Qwen35VLAgent` when
`OSWORLD_AGENT_KIND=qwen35vl`, supplies the local endpoint, configures guest
resource tracing, and assigns a per-task trace directory. The adapter supplies
the Qwen-VL XML action protocol, bounded screenshot history, and deterministic
1000x1000-to-screen coordinate conversion. It also translates QW3's native
OpenAI `message.tool_calls` response back into the XML form expected by the
bundled parser; upstream otherwise reads only `message.content` and drops the
action.

`mm_agents/qwen35vl_agent.py` records phase timings, screenshot folding state,
OpenAI usage/server timings, exact processed input screenshots, sanitized
message payloads, model responses, and parsed actions. `lib_run_single.py` and
`desktop_env/desktop_env.py` add end-to-end step, action dispatch, observation,
trajectory, guest resource, and evaluator timings. These patches do not alter
the task definitions, assets, evaluator logic, or pinned runtime image.

Every Hugging Face download command removes uppercase and lowercase HTTP,
HTTPS, and ALL proxy variables. The gated task and asset rows can only be
validated after local Hugging Face authentication and acceptance of both
dataset access agreements.
