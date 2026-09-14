# Dense prefix checkpoint storage

`QW3_PREFIX_CACHE_GPU_ENTRIES=4` enables CPU backing for older immutable
recurrent/conv snapshots in the ordinary continuous-batching prefix cache.
The default is `0`, which preserves the existing GPU-only behavior. This is
separate from KVMem block GC and is not enabled for the serial KVMem arm.

The logical entry limit (normally 64), LRU selection, token/image matching,
main/MTP KV-page pins and snapshot numerical precision remain unchanged.
Only the recurrent/conv and hidden snapshot storage moves. It does not move
model weights, the vision encoder or active model computation to the CPU.

On a hot hit, an immutable shared owner keeps the cached snapshot alive while
the executor copies it into its mutable state. This removes the intermediate
deep clone. Cold snapshots retain the original FP32/BF16 bytes in pinned CPU
memory; a cold hit restores the packed state through the existing transfer
stream before the same executor restore operation. MTP and position metadata
travel with the snapshot. Moving the snapshot does not invalidate KV pages or
turn the hit into a cold prefill.

Older unreferenced snapshots start asynchronous D2H copies after capture and
before decoding. The request-completion path waits for the transfer fence
before recycling device storage. Active readers and in-flight transfers retain
their ownership, including during eviction and cancellation. One released
device allocation can be reused for a later capture or cold restore. No new
worker calls the shared device backend.

For Qwen3.8-27B, a snapshot is about 149.6 MiB; 64 pinned backing buffers reserve
about 9.35 GiB of CPU RAM at model startup. This trades CPU memory and startup
allocation time for lower GPU residency. Four hot entries are a residency
target, not a hard VRAM cap: active leases, an in-flight copy and a recycled
allocation may temporarily add device storage. A finite logical cache limit
is required. Setting a hot-entry target at or above that limit leaves storage
unchanged; combining the feature with CB KVMem is rejected.

Transfers can consume PCIe and memory bandwidth. Cold hits also wait for H2D,
so an efficiency claim requires measurement on the relevant workload. The
current OSWorld validation keeps the original requests and full computation,
compares output hashes, checkpoint choices and prefilled-token counts, and
separates uninstrumented timing runs from allocation-instrumented memory runs.
The implementation is opt-in while this acceptance work is pending.

`qw3-prefix-state-storage` tests bitwise state restoration, metadata, reader
lifetime, recycling and destruction during an in-flight copy. Experiment
artifacts and frozen builds are under
`benchmark/osworld_v2/prefix_checkpoint_memory_20260910/`.
