#include "kvmem_mismatch_observer.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <cstdio>
#include <type_traits>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace qw3::detail {
void MismatchQueryMean::reset() {
    discarded_tokens += count;
    count = 0; next = 0; sum.clear();
}
std::vector<MismatchMeanQuery> MismatchQueryMean::append(
        const uint16_t *q, uint32_t layers, uint32_t captured, uint32_t accepted,
        uint32_t width, uint64_t base, uint32_t block_tokens, uint32_t first_row) {
    if (!block_tokens || !layers || !width || accepted > captured || first_row>accepted)
        throw std::invalid_argument("invalid block mean query shape");
    const uint64_t size = uint64_t(layers)*width;
    if ((count && next != base+first_row) || sum.size() != size) reset();
    if (sum.empty()) sum.assign(size, 0);
    std::vector<MismatchMeanQuery> result;
    for (uint32_t r=first_row;r<accepted;++r) {
        for (uint32_t l=0;l<layers;++l) for (uint32_t d=0;d<width;++d)
            sum[uint64_t(l)*width+d] += mismatch_half_to_float(q[(uint64_t(l)*captured+r)*width+d]);
        ++count; next=base+r+1;
        if (next % block_tokens == 0) {
            MismatchMeanQuery mean; mean.position=next-1; mean.tokens=count;
            mean.values=sum;
            for (float &v:mean.values) v/=count;
            result.push_back(std::move(mean));
            std::fill(sum.begin(),sum.end(),0); count=0;
        }
    }
    return result;
}
float mismatch_half_to_float(uint16_t v) {
    uint32_t sign = uint32_t(v & 0x8000) << 16;
    uint32_t e = (v >> 10) & 31, m = v & 1023, bits;
    if (!e) {
        if (!m) bits = sign;
        else {
            int shift = 0;
            while (!(m & 1024)) { m <<= 1; ++shift; }
            bits = sign | (uint32_t(113-shift) << 23) | ((m & 1023) << 13);
        }
    } else bits = sign | ((e == 31 ? 255u : e + 112u) << 23) | (m << 13);
    float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
double mismatch_decay(double previous, double missing, double epsilon,
                      uint64_t delta, double half_life) {
    const double weight = -std::expm1(-std::log(2.0) * double(delta) / half_life);
    return std::max(0.0, (1-weight)*previous + weight*(missing-epsilon));
}
namespace {
float dot_scalar(const float *a, const float *b, uint32_t n) {
    float sum = 0; for (uint32_t i=0;i<n;++i) sum += a[i]*b[i]; return sum;
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
float dot_avx2(const float *a, const float *b, uint32_t n) {
    __m256 acc = _mm256_setzero_ps(), a1=acc, a2=acc, a3=acc; uint32_t i = 0;
    for (;i+32<=n;i+=32) {
        acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),acc);
        a1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),a1);
        a2=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16),a2);
        a3=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24),a3);
    }
    acc=_mm256_add_ps(_mm256_add_ps(acc,a1),_mm256_add_ps(a2,a3));
    for (;i+8<=n;i+=8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    alignas(32) float v[8]; _mm256_store_ps(v, acc);
    float sum = 0; for (float x:v) sum+=x;
    for (;i<n;++i) sum+=a[i]*b[i];
    return sum;
}
#endif
auto dot_function() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return &dot_avx2;
#endif
    return &dot_scalar;
}
float raw_value(const uint8_t *p, uint32_t bytes) {
    if (bytes==4) { float v;std::memcpy(&v,p,4);return v; }
    if (bytes==2) { uint16_t v;std::memcpy(&v,p,2);return mismatch_half_to_float(v); }
    if (bytes!=1) throw std::runtime_error("invalid raw K dtype");
    // Raw KV storage is E4M3FN, without a scale. 0x7f/0xff are NaNs.
    const unsigned v=*p, e=(v>>3)&15, m=v&7;
    const float f=e==15 && m==7 ? std::numeric_limits<float>::quiet_NaN()
        : e ? std::ldexp(float(8+m),int(e)-10) : std::ldexp(float(m),-9);
    return v&128 ? -f:f;
}
std::mutex log_mutex;
std::atomic<uint64_t> next_id{0};
void json_number(std::ostream &out, double value) {
    if (std::isfinite(value)) out<<value;
    else out<<"null";
}
template<class T> void json_numbers(std::ostream &out, const char *name, const std::vector<T> &values) {
    out<<",\""<<name<<"\":[";
    for (size_t i=0;i<values.size();++i) {
        if (i) out<<",";
        if constexpr (std::is_integral_v<T>) out<<values[i];
        else json_number(out,values[i]);
    }
    out<<"]";
}
}

MismatchScore score_mismatch(const MismatchJob &j,
        std::vector<std::shared_ptr<const MismatchKeyBlock>> &ids,
        std::vector<std::vector<float>> &values) {
    const size_t L=j.layers.size(), Q=j.query_positions.size(), N=j.blocks.size();
    if (!L || !Q || !j.dim || !j.kv_heads || !j.heads || j.heads%j.kv_heads ||
        !j.block_tokens || j.selected_tokens.size()!=N || j.live.size()!=N ||
        (!j.mandatory_tokens.empty() && j.mandatory_tokens.size()!=N) ||
        (j.mean_queries.empty() ? j.queries.size():j.mean_queries.size())!=L*Q*j.heads*j.dim ||
        j.indexed_tokens!=N*j.block_tokens)
        throw std::runtime_error("invalid CPU mismatch probe shape");
    for (size_t b=0;b<j.mandatory_tokens.size();++b)
        if (j.mandatory_tokens[b]>std::min(j.selected_tokens[b],j.block_tokens))
            throw std::runtime_error("CPU mismatch mandatory prefix is not a visible selected subset");
    ids.resize(N); values.resize(N);
    const uint64_t key_width=L*j.kv_heads*j.dim;
    for (size_t b=0;b<N;++b) {
        if (!j.live[b]) continue;
        if (!j.blocks[b])
            throw std::runtime_error("CPU mismatch index has a hole");
        if (ids[b]!=j.blocks[b]) {
            const auto &block=*j.blocks[b];
            values[b].assign(key_width,0);
            if (block.raw_tokens) {
                if (block.raw_tokens!=j.block_tokens ||
                    block.raw.size()!=key_width*block.raw_tokens*block.raw_elem_bytes)
                    throw std::runtime_error("invalid raw block supplement");
                const uint64_t width=uint64_t(j.kv_heads)*j.dim;
                for (size_t l=0;l<L;++l) for (uint32_t t=0;t<block.raw_tokens;++t)
                    for (size_t d=0;d<width;++d)
                        values[b][l*width+d]+=raw_value(block.raw.data()+
                            ((l*block.raw_tokens+t)*width+d)*block.raw_elem_bytes,block.raw_elem_bytes);
                for (float &v:values[b]) v/=block.raw_tokens;
            } else {
                if (block.keys.size()!=key_width) throw std::runtime_error("invalid Mean-K block");
                std::transform(block.keys.begin(), block.keys.end(),values[b].begin(), mismatch_half_to_float);
            }
            for (float x:values[b]) if (!std::isfinite(x)) throw std::runtime_error("invalid Mean-K value");
            ids[b]=j.blocks[b];
        }
    }
    std::vector<float> queries=j.mean_queries;
    if (queries.empty()) {
        queries.resize(j.queries.size());
        std::transform(j.queries.begin(), j.queries.end(), queries.begin(), mismatch_half_to_float);
    }
    for (float x:queries) if (!std::isfinite(x)) throw std::runtime_error("invalid probe Q");
    static const auto dot=dot_function();
    const float scale=1.0f/std::sqrt(float(j.dim));
    std::vector<float> outside(Q*N);
    std::vector<double> mandatory_fraction(Q*N);
    std::vector<size_t> eligible(Q);
    std::vector<uint64_t> candidates(Q,0);
    MismatchScore out;
    out.query_candidate_tokens.assign(Q,0);
    out.query_visible_tokens.assign(Q,0);
    out.query_mandatory_tokens.assign(Q,0);
    for (size_t r=0;r<Q;++r) {
        const uint64_t end=j.query_positions[r]+1;
        eligible[r]=std::min<uint64_t>(N, end/j.block_tokens);
        for (size_t b=0;b<eligible[r];++b) if (j.live[b]) {
            ++candidates[r];
            const uint64_t lo=b*j.block_tokens, hi=lo+j.block_tokens;
            const uint64_t selected=std::min(j.selected_tokens[b],j.block_tokens);
            const uint64_t tail_lo=std::max({lo, j.tail_begin, lo+selected});
            const uint64_t tail_hi=std::min(hi,end);
            const uint64_t visible=selected+(tail_hi>tail_lo ? tail_hi-tail_lo : 0);
            outside[r*N+b]=1.0f-float(visible)/j.block_tokens;
            const uint64_t prefix=j.mandatory_tokens.empty()?0:j.mandatory_tokens[b];
            const uint64_t fixed_tail_lo=std::max({lo,j.tail_begin,lo+prefix});
            const uint64_t fixed=prefix+(tail_hi>fixed_tail_lo ? tail_hi-fixed_tail_lo : 0);
            if (fixed>visible || visible>j.block_tokens)
                throw std::runtime_error("CPU mismatch invalid mandatory/visible union");
            mandatory_fraction[r*N+b]=double(fixed)/j.block_tokens;
            out.query_candidate_tokens[r]+=j.block_tokens;
            out.query_visible_tokens[r]+=visible;
            out.query_mandatory_tokens[r]+=fixed;
        }
        out.candidate_blocks=std::max(out.candidate_blocks,candidates[r]);
    }
    const size_t D=L*Q*j.heads;
    std::vector<double> maxima(D,-std::numeric_limits<double>::infinity()), sums(D,0), missing(D,0), mandatory_mass(D,0);
    // Block-major traversal reuses each Key across all Query rows/GQA heads.
    // Online softmax keeps only 4 scalars per distribution, not an N*Q*H matrix.
    // Mandatory mass reuses the same logit and softmax rescaling; no extra QK.
    for (size_t b=0;b<N;++b) if (j.live[b]) {
        for (size_t l=0;l<L;++l) for (size_t r=0;r<Q;++r) if (b<eligible[r]) {
            for (uint32_t h=0;h<j.heads;++h) {
                const size_t d=(l*Q+r)*j.heads+h;
                const size_t offset=(l*j.kv_heads+h/(j.heads/j.kv_heads))*j.dim;
                const double z=dot(queries.data()+d*j.dim,values[b].data()+offset,j.dim)*scale;
                if (!std::isfinite(z)) throw std::runtime_error("invalid probe logit");
                if (z>maxima[d]) {
                    const double rescale=std::exp(maxima[d]-z);
                    sums[d]=sums[d]*rescale+1;
                    missing[d]=missing[d]*rescale+outside[r*N+b];
                    mandatory_mass[d]=mandatory_mass[d]*rescale+mandatory_fraction[r*N+b];
                    maxima[d]=z;
                } else {
                    const double w=std::exp(z-maxima[d]);
                    sums[d]+=w; missing[d]+=w*outside[r*N+b];
                    mandatory_mass[d]+=w*mandatory_fraction[r*N+b];
                }
            }
        }
    }
    uint32_t valid_rows=0;
    const double undefined=std::numeric_limits<double>::quiet_NaN();
    out.query_coverage.assign(Q,undefined);
    out.query_mandatory_coverage.assign(Q,undefined);
    out.query_random_coverage.assign(Q,undefined);
    out.query_lift.assign(Q,undefined);
    out.query_baseline_valid.assign(Q,0);
    out.query_baseline_status.assign(Q,"no_candidates");
    double random_sum=0;
    for (size_t r=0;r<Q;++r) if (candidates[r]) {
        double row_missing=0, row_mandatory=0;
        for (size_t l=0;l<L;++l) for (uint32_t h=0;h<j.heads;++h) {
            const size_t d=(l*Q+r)*j.heads+h;
            row_missing+=missing[d]/sums[d]; ++out.distributions;
            row_mandatory+=mandatory_mass[d]/sums[d];
        }
        row_missing/=L*j.heads;
        out.query_coverage[r]=std::clamp(1-row_missing,0.0,1.0);
        // First average normalized distributions over layers/heads, then form
        // E and C/E. This preserves the original coverage aggregation order.
        const double cm=std::clamp(row_mandatory/(L*j.heads),0.0,1.0);
        out.query_mandatory_coverage[r]=cm;
        const uint64_t n=out.query_candidate_tokens[r], b=out.query_visible_tokens[r], m=out.query_mandatory_tokens[r];
        double expected;
        if (m==n || b==n) {
            expected=1;
            out.query_lift[r]=1;
            out.query_baseline_status[r]=m==n?"no_selection":"full_retention";
        } else {
            expected=cm+(double(b-m)/double(n-m))*(1-cm);
            if (expected>0) {
                out.query_lift[r]=out.query_coverage[r]/expected;
                out.query_baseline_status[r]="ok";
            } else out.query_baseline_status[r]="zero_expectation";
        }
        out.query_random_coverage[r]=expected;
        out.query_baseline_valid[r]=std::isfinite(out.query_lift[r]);
        random_sum+=expected;
        out.missing+=row_missing; ++valid_rows;
        out.min_missing=std::min(out.min_missing,row_missing);
        out.max_missing=std::max(out.max_missing,row_missing);
    }
    if (valid_rows) {
        out.missing/=valid_rows;
        out.random_coverage=random_sum/valid_rows;
        if (out.random_coverage>0) out.lift=(1-out.missing)/out.random_coverage;
    }
    else out.min_missing=0;
    return out;
}

MismatchObserver::MismatchObserver(KvMemMismatchConfig config):config_(std::move(config)),id_(++next_id) {
    config_.validate();
    log_.open(config_.log_path,std::ios::app);
    if (!log_) throw std::runtime_error("cannot open --kvmem-mismatch-log: "+config_.log_path);
    worker_=std::thread([this]{run();});
}
MismatchObserver::~MismatchObserver() { finish(); }
void MismatchObserver::finish() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_=true; }
    wake_.notify_one();
    if (worker_.joinable()) worker_.join();
}
bool MismatchObserver::submit(MismatchJob job) {
    bool expected=false;
    if (!busy_.compare_exchange_strong(expected,true,std::memory_order_acq_rel)) return false;
    { std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) { busy_=false; return false; }
      pending_=std::make_unique<MismatchJob>(std::move(job)); }
    wake_.notify_one(); return true;
}
void MismatchObserver::run() {
    std::vector<std::shared_ptr<const MismatchKeyBlock>> ids;
    std::vector<std::vector<float>> values;
    uint64_t epoch=UINT64_MAX, selection=UINT64_MAX, last[2]={0,0};
    double debt[2]={0,0};
    for (;;) {
        std::unique_ptr<MismatchJob> j;
        { std::unique_lock<std::mutex> lock(mutex_);
          wake_.wait(lock,[&]{return stopping_ || pending_;});
          if (!pending_) return;
          j=std::move(pending_); }
        const auto start=std::chrono::steady_clock::now();
        std::ostringstream line; line<<std::setprecision(10);
        const auto unix_ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        line<<"{\"kind\":\"kvmem_mismatch\",\"time_unix_ms\":"<<unix_ms<<",\"observer\":"<<id_
            <<",\"epoch\":"<<j->epoch<<",\"selection_generation\":"<<j->selection
            <<",\"phase\":\""<<(j->prefill?"prefill":"output")<<"\",\"position\":"<<j->position
            <<",\"progress_tokens\":"<<j->progress<<",\"indexed_tokens\":"<<j->indexed_tokens
            <<",\"unindexed_tokens\":"<<(j->position>j->indexed_tokens?j->position-j->indexed_tokens:0)
            <<",\"query_samples\":"<<j->query_positions.size()<<",\"layers\":[";
        for (size_t i=0;i<j->layers.size();++i) line<<(i?",":"")<<j->layers[i];
        line<<"],\"query_positions\":[";
        for (size_t i=0;i<j->query_positions.size();++i) line<<(i?",":"")<<j->query_positions[i];
        line<<"],\"schema_version\":3,\"sink_excluded\":false,\"output_scope\":\"all-output\""
            <<",\"timestamp_semantics\":\"cpu-worker-start\""
            <<",\"baseline\":\"conditional-uniform-token-subset\""
            <<",\"baseline_mandatory\":\"resident-policy-fixed-prefixes-union-accepted-tail\""
            <<",\"baseline_mass_model\":\"uniform-within-mean-k-block\""
            <<",\"baseline_selector_distribution\":false"
            <<",\"aggregate_random_coverage\":\"mean-over-coverage-valid-query-rows\""
            <<",\"aggregate_lift\":\"aggregate-coverage/aggregate-random-coverage\""
            <<",\"query_mode\":\""<<config_.query_mode<<"\",\"block_tokens\":"<<j->block_tokens
            <<",\"official_indexed_tokens\":"<<j->official_indexed_tokens
            <<",\"skipped_query_blocks\":"<<j->skipped_query_blocks
            <<",\"discarded_query_tokens\":"<<j->discarded_query_tokens
            <<",\"proxy\":\"mean-k-all-visible-history\",\"refresh_triggered\":false"
            <<",\"query_token_counts\":[";
        for (size_t i=0;i<j->query_token_counts.size();++i) line<<(i?",":"")<<j->query_token_counts[i];
        line<<"]";
        try {
            if (epoch!=j->epoch) { ids.clear(); values.clear(); }
            const bool reset=epoch!=j->epoch || selection!=j->selection;
            if (reset) { debt[0]=debt[1]=0; last[0]=last[1]=0; }
            epoch=j->epoch; selection=j->selection;
            const auto s=score_mismatch(*j,ids,values);
            const uint32_t phase=j->prefill?0:1;
            const uint64_t delta=last[phase] && j->progress>last[phase]
                ? j->progress-last[phase] : config_.interval;
            if (s.distributions) {
                debt[phase]=mismatch_decay(debt[phase],s.missing,config_.epsilon,delta,config_.half_life);
                last[phase]=j->progress;
                line<<",\"status\":\"ok\",\"missing\":"<<s.missing<<",\"debt\":"<<debt[phase]
                    <<",\"coverage\":"<<1-s.missing
                    <<",\"min_missing\":"<<s.min_missing<<",\"max_missing\":"<<s.max_missing;
            } else line<<",\"status\":\"no_candidates\",\"missing\":null,\"debt\":null,\"coverage\":null";
            json_numbers(line,"query_coverage",s.query_coverage);
            json_numbers(line,"query_candidate_tokens",s.query_candidate_tokens);
            json_numbers(line,"query_visible_tokens",s.query_visible_tokens);
            json_numbers(line,"query_mandatory_tokens",s.query_mandatory_tokens);
            json_numbers(line,"query_mandatory_coverage",s.query_mandatory_coverage);
            json_numbers(line,"query_random_coverage",s.query_random_coverage);
            json_numbers(line,"query_lift",s.query_lift);
            line<<",\"query_baseline_valid\":[";
            for (size_t i=0;i<s.query_baseline_valid.size();++i)
                line<<(i?",":"")<<(s.query_baseline_valid[i]?"true":"false");
            line<<"],\"query_baseline_status\":[";
            for (size_t i=0;i<s.query_baseline_status.size();++i)
                line<<(i?",":"")<<"\""<<s.query_baseline_status[i]<<"\"";
            line<<"],\"random_coverage\":"; json_number(line,s.random_coverage);
            line<<",\"lift\":"; json_number(line,s.lift);
            line<<",\"debt_reset\":"<<(reset?"true":"false")<<",\"delta_tokens\":"<<delta
                <<",\"epsilon\":"<<config_.epsilon<<",\"half_life\":"<<config_.half_life
                <<",\"candidate_blocks\":"<<s.candidate_blocks<<",\"distributions\":"<<s.distributions;
        } catch (const std::exception &e) {
            line<<",\"status\":\"error\",\"missing\":null,\"debt\":null";
            std::fprintf(stderr,"[kvmem-mismatch] CPU probe failed: %s\n",e.what());
        }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        line<<",\"cpu_ms\":"<<ms<<",\"capture_ms\":"<<j->capture_ms
            <<",\"snapshot_ms\":"<<j->snapshot_ms
            <<",\"index_update_ms_total\":"<<j->index_update_ms_total
            <<",\"index_copy_bytes_total\":"<<j->index_copy_bytes_total<<",\"skipped_busy\":"<<j->skipped_busy<<"}\n";
        { std::lock_guard<std::mutex> lock(log_mutex); log_<<line.str(); log_.flush(); }
        busy_.store(false,std::memory_order_release);
    }
}
}
