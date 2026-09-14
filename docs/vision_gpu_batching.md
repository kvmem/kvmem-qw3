# Bounded native vision batches

The native BF16 vision encoder processes at most four uncached images per
forward call. `QW3_VISION_MAX_BATCH_IMAGES` overrides this limit; `0` retains the
old unbounded batch for diagnostics. Cache hits, deduplication, image order,
image grids and all encoder layers are preserved. A request with one new
screenshot retains the same forward call.

The bound addresses activation tensors that were sized for every uncached
image simultaneously. In a skipped-history stress replay, both old and initial
memory-optimized servers reached about 77.96 GiB while encoding 97 new images.
The completed chronological 053 trajectories stayed below 48 GiB, so ordinary
stepwise recordings alone did not cover that allocation pattern.

This bounds batch activation residency, not all GPU allocations. Cached
features and features held by a request are additional storage, and image
resolution also matters. Total memory must be measured on the intended inputs.
Changing the GEMM batch shape can affect low BF16 bits; the batch parity test
reports exact differences and uses the existing CPU/GPU vision parity tolerance.
Complete chronological model-output checks remain separate acceptance gates.

The regression mode of `qw3-vision-gpu-parity` is:

```text
--batch-parity MODEL_DIR PYTHON WORKER IMAGE...
```

It requires at least nine distinct images and compares two unbounded forwards
with two bounded forwards, including duplicate images and exact cache-based
reordering. The current experiment supplies 97 real screenshots and separately
checks allocation peaks under bulk ingestion. Runtime weights, CUDA kernels,
guided query generation and query replay are unchanged.
