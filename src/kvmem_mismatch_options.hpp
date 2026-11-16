#pragma once
#include "qw3/qw3.hpp"
namespace qw3::detail {
inline void validate_mismatch_options(const EngineOptions &e) {
    e.kvmem_mismatch.validate();
    if (!e.kvmem_mismatch.enabled()) return;
    if (!e.kvmem_enabled || !e.kvmem_query_conditioned)
        throw std::invalid_argument("CPU mismatch observation requires --kvmem and --kvmem-query-conditioned");
    if (e.kvmem_method!="retrieval" || e.kvmem_index_placement!="cpu" || e.kvmem_retrieval_method!="mean-k")
        throw std::invalid_argument("CPU mismatch observation requires --kvmem-method retrieval, --kvmem-index-placement cpu and --kvmem-retrieval-method mean-k");
    if (e.kvmem_repository_blocks || e.kvmem_gc_max_blocks || !e.kvmem_archive_dir.empty())
        throw std::invalid_argument("CPU mismatch observation requires online KVMem without archive/repository/GC");
    if (e.kvmem_mismatch.query_mode=="block-mean" && !e.kvmem_immutable_source_k)
        throw std::invalid_argument("block-mean mismatch observation requires --kvmem-immutable-k");
}
}
