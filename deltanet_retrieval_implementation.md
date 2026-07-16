All done. Full build is clean and all 9 tests pass (including the new DeltaNet scoring parity test). Let me give a final summary.
# Todos
[â] Add KvMemRetrievalMethod::DeltaNet enum + config fields (deltanet layers, mem budget, decay, topk_q/topk_h) in kvmem_store.hpp
[â] Plumb flags through EngineOptions (qw3.hpp), CLI parse (qw3_cli.cpp), string->enum + config in qwen_native_backend.cpp
[â] Add CUDA kernels: block-boundary state snapshot + per-block cumulative decay in delta kernels; E_j formation + G_j; deltanet query capture (L2-norm)
[â] Add executor mirror fields + kvmem_resolve_deltanet_layers() + buffer sizing/allocation with mem-budget-driven layer count
[â] Add prefill capture hook in recurrent branch (kvmem_capture_deltanet_multi) + deltanet query capture
[â] Add kvmem_retrieval_score_deltanet() boundary scorer (doc Â§10) + dispatch in kvmem_prepare_reselect with mean-k fallback
[â] Add host unit test for aggregation math + CUDA parity test for E_j/scoring
[â] Build and verify compilation + run tests
All steps are complete. The build is clean and all 9 tests pass (including the 2 new ones). Here's a summary of what was implemented.
Summary
Implemented the DeltaNet-state block-recall method from deltanet_retrieval.md as a new --kvmem-retrieval-method deltanet scorer, using the memory-budget-driven layer selection we agreed on.
What it does
Scores each historical KV block by the net edit it made to the DeltaNet recurrent state (E_j = S_j â a_jÂ·S_{j-1}), read by the current DeltaNet query, aggregated per doc Â§10:
s_j = Î£_l w_l Â· RMSnorm_j[ TopKMean_h( TopKMean_t( d_jÂ·âE_jáµ q_tââ ) ) ]
with d_j = exp(G_M â G_j) (decay, default on).
Files changed (10 files, ~1090 lines)
- include/qw3/kvmem_store.hpp â KvMemRetrievalMethod::DeltaNet + config fields (layers, mem-budget, decay, topk_q/h).
- include/qw3/qw3.hpp, src/qw3_cli.cpp, src/qwen_native_backend.cpp â flag plumbing: --kvmem-deltanet-{layers,mem-budget-gb,decay,topk-q,topk-h}.
- src/kernels_cuda.cu â new kernels: block-boundary S_j snapshot (added to deltanet_batch_kernel), deltanet_decay_scan_kernel (per-block Î£ log a), deltanet_pack_query_kernel (L2-normed DeltaNet query capture), deltanet_block_score_kernel (the âE_jáµqâ + TopKMean scorer) + backend overrides.
- include/qw3/device_backend.hpp â 3 new virtual backend methods (recurrent_batch gained DeltaNet-capture params, deltanet_pack_query_device, deltanet_block_score_device).
- src/qwen_executor.{hpp,cpp} â kvmem_resolve_deltanet_layers() (evenly-spaced, budget-capped, hard-errors + logs estimated GB), buffer allocation, prefill capture hook in the recurrent branch, and kvmem_retrieval_score_deltanet() boundary scorer with mean-k fallback.
- tests/kvmem_store_test.cpp, tests/kvmem_deltanet_score.cu, CMakeLists.txt â host config/scoring test + CUDA parity test (block score max_rel ~1e-7, RMS-norm invariant verified).
Key design points
- Memory-budget-driven layers: default = Â½ the DeltaNet layers, auto-clamped so the E_j snapshot buffer fits --kvmem-deltanet-mem-budget-gb (default 32); logs L_dn, v_heads, and estimated GB at startup. For 256K context with block-256, this lands at ~1/3 the layers when v_heads=32, exactly as planned.
- Forces the qw3 delta kernel when capturing (the ported warp kernel has no snapshot side-channel); zero overhead when the deltanet method isn't selected.
- Falls back to mean-k / single-token content scoring if capture isn't live.
To run
--kvmem-enabled --kvmem-query-conditioned \
--kvmem-retrieval-method deltanet \
--kvmem-block-tokens 256 \
--kvmem-deltanet-mem-budget-gb 32
Set QW3_KVMEM_DELTANET_TRACE=1 to see the resolved layer count / buffer size and per-reselect scoring logs.
One note for when you run it at scale: the decay-coefficient reduction (G_j prefix + exp(G_MâG_j)) and the head/layer aggregation tail currently run on the host (kvmem_retrieval_score_deltanet). That's LÂ·blocksÂ·v_heads floats â fine for a first version, but a candidate to move on-GPU if it shows up in profiling at 256K+.