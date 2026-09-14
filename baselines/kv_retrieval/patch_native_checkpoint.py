"""Fix stale post-selection promotion only in the isolated native snapshot."""
import difflib
import json
from prepare_native import ROOT, TARGET


def main():
    path = TARGET / "src/qwen_native_backend.cpp"
    source = path.read_text()
    include = '#include "../../baseline_checkpoint_choice.hpp"\n'
    while include + include in source:
        source = source.replace(include + include, include)
    changes = []
    def replace_once(path, before, after):
        nonlocal source
        if source.count(after) == 1:
            pass
        elif source.count(before) == 1:
            source = source.replace(before, after)
        else:
            raise RuntimeError("unexpected source for atomic checkpoint patch")
        changes.append((before, after))
        return ""
    diff = replace_once(path, '#include "qwen_executor.hpp"',
        '#include "qwen_executor.hpp"\n#include "../../baseline_checkpoint_choice.hpp"')
    old = '''                const bool postselection_checkpoint_current =
                    kvmem_postselection_ckpt.ready &&
                    kvmem_postselection_pos <= pos &&
                    detail::kvmem_checkpoint_selection_compatible(
                        kvmem_postselection_ckpt
                            .kvmem_selection_generation,
                        executor_->kvmem_selection_generation());'''
    new = '''                const auto checkpoint_choice =
                    baseline::choose_latest_aligned_checkpoint(
                        {kvmem_warm_prompt_pos,
                         kvmem_warm_ckpt_prompt_.kvmem_selection_generation,
                         kvmem_warm_ckpt_prompt_.ready &&
                             kvmem_warm_ckpt_prompt_.position == kvmem_warm_prompt_pos},
                        {kvmem_postselection_pos,
                         kvmem_postselection_ckpt.kvmem_selection_generation,
                         kvmem_postselection_ckpt.ready &&
                             kvmem_postselection_ckpt.position == kvmem_postselection_pos},
                        static_cast<uint32_t>(pos),
                        executor_->kvmem_selection_generation(),
                        executor_->block_store()->config().block_tokens);
                // Replay can capture a newer aligned P after the provisional
                // post-selection origin. Promote the latest actual state, not
                // blindly the state from before replaying this request's tail.
                const bool postselection_checkpoint_current =
                    checkpoint_choice.postselection;'''
    diff += replace_once(path, old, new)
    diff += replace_once(path, '''                const bool prompt_index_finalized =
                    !postselection_checkpoint_current ||
                    executor_->kvmem_finalize_content_index_from_raw_k(
                        kvmem_postselection_pos);''', '''                const bool prompt_index_finalized =
                    checkpoint_choice.position == 0 ||
                    executor_->kvmem_finalize_content_index_from_raw_k(
                        checkpoint_choice.position);''')
    old = '''                    kvmem_warm_prompt_pos =
                        kvmem_postselection_pos;
                    kvmem_warm_prompt_resumable =
                        prompt_index_finalized &&
                        executor_->kvmem_content_index_resume_compatible(
                            kvmem_warm_prompt_pos) &&
                        executor_->kvmem_mtp_checkpoint_status(
                            kvmem_warm_prompt_pos).complete;
                }
                executor_->capture_state(kvmem_warm_ckpt_end_);
                const bool prompt_source_index_ready ='''
    new = old.replace('                executor_->capture_state(kvmem_warm_ckpt_end_);', '''                if (!postselection_checkpoint_current &&
                    checkpoint_choice.position > 0) {
                    kvmem_warm_prompt_resumable =
                        prompt_index_finalized &&
                        executor_->kvmem_content_index_resume_compatible(
                            kvmem_warm_prompt_pos) &&
                        executor_->kvmem_mtp_checkpoint_status(
                            kvmem_warm_prompt_pos).complete;
                }
                executor_->capture_state(kvmem_warm_ckpt_end_);''')
    diff += replace_once(path, old, new)
    original = source
    for before, after in reversed(changes):
        original = original.replace(after, before, 1)
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True),
        source.splitlines(keepends=True), fromfile="src/qwen_native_backend.cpp",
        tofile="src/qwen_native_backend.cpp"))
    if path.read_text() != source:
        path.write_text(source)
    (ROOT / "reports/native_checkpoint_patch.diff").write_text(diff)
    manifest = ROOT / "reports/native_source_manifest.json"
    value = json.loads(manifest.read_text())
    value["additional_checkpoint_patch"] = "native_checkpoint_patch.diff"
    value["checkpoint_patch_validated"] = False
    manifest.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
