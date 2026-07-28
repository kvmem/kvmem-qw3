#include "fp8_scaled_mm_adapter.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace qw3::fp8_scaled_mm_adapter {
namespace {

using namespace cute;

template <typename Kernel>
struct EnableSm120Family : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args &&...args) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
        Kernel::operator()(cute::forward<Args>(args)...);
#else
        asm("trap;");
#endif
    }
};

template <typename ElementAcc, typename ElementD, typename TileShape>
struct ScaledEpilogue {
private:
    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
    using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, TileShape, float, float, Stride<_1, _0, _0>,
        128 / cutlass::sizeof_bits<float>::value, false>;
    using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, float, float, Stride<_0, _1, _0>,
        128 / cutlass::sizeof_bits<float>::value, false>;
    using ApplyWeightScale = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using WeightScaled = cutlass::epilogue::fusion::Sm90EVT<
        ApplyWeightScale, ScaleB, Accum>;
    using ApplyActivationScale = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, float,
        cutlass::FloatRoundStyle::round_to_nearest>;

public:
    using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<
        ApplyActivationScale, ScaleA, WeightScaled>;
    using ArgumentType = typename EVTCompute::Arguments;

    static ArgumentType prepare_args(const float *activation_scales,
                                     const float *weight_scales) {
        typename ScaleA::Arguments activation_args{
            activation_scales, 0.0f, {}};
        typename ScaleB::Arguments weight_args{
            weight_scales, 0.0f, {}};
        typename WeightScaled::Arguments weight_scaled_args{
            weight_args, {}, {}};
        return ArgumentType{
            activation_args, weight_scaled_args, {}};
    }
};

template <typename ElementAB,
          typename ElementD,
          template <typename, typename, typename> typename Epilogue,
          typename TileShape,
          typename ClusterShape,
          typename KernelSchedule,
          typename EpilogueSchedule>
struct Sm120Gemm {
    using ElementOutput = ElementD;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using ElementC = void;
    using ElementAcc = float;
    using EpilogueOp = Epilogue<ElementAcc, ElementD, TileShape>;

    static constexpr int AlignmentA =
        128 / cutlass::sizeof_bits<ElementAB>::value;
    static constexpr int AlignmentB = AlignmentA;
    static constexpr int AlignmentD =
        128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = AlignmentD;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm120,
            cutlass::arch::OpClassTensorOp,
            TileShape,
            ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAcc,
            float,
            ElementC,
            LayoutC,
            AlignmentC,
            ElementD,
            LayoutD,
            AlignmentD,
            EpilogueSchedule,
            typename EpilogueOp::EVTCompute>::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm120,
            cutlass::arch::OpClassTensorOp,
            ElementAB,
            LayoutA,
            AlignmentA,
            ElementAB,
            LayoutB,
            AlignmentB,
            ElementAcc,
            TileShape,
            ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(
                    sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelSchedule,
            void>::CollectiveOp;

    using GemmKernel = EnableSm120Family<
        cutlass::gemm::kernel::GemmUniversal<
            Shape<int, int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue,
            void>>;
};

template <typename Gemm>
bool launch(const void *activation_fp8,
            const void *weight_fp8,
            void *out_bf16,
            const float *activation_scales,
            const float *weight_scales,
            uint32_t rows,
            uint32_t out_cols,
            uint32_t inner_cols,
            void *workspace,
            size_t workspace_capacity,
            cudaStream_t stream) {
    using GemmKernel = typename Gemm::GemmKernel;
    using ElementAB = cutlass::float_e4m3_t;
    using ElementD = typename Gemm::ElementOutput;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = StrideC;

    typename GemmKernel::ProblemShape problem{
        static_cast<int>(rows),
        static_cast<int>(out_cols),
        static_cast<int>(inner_cols),
        1};
    auto [m, n, k, l] = problem;
    StrideA stride_a = cutlass::make_cute_packed_stride(
        StrideA{}, make_shape(m, k, l));
    StrideB stride_b = cutlass::make_cute_packed_stride(
        StrideB{}, make_shape(n, k, l));
    StrideC stride_c = cutlass::make_cute_packed_stride(
        StrideC{}, make_shape(m, n, l));
    StrideD stride_d = cutlass::make_cute_packed_stride(
        StrideD{}, make_shape(m, n, l));

    typename GemmKernel::MainloopArguments mainloop_args{
        reinterpret_cast<const ElementAB *>(activation_fp8),
        stride_a,
        reinterpret_cast<const ElementAB *>(weight_fp8),
        stride_b};
    auto *out = reinterpret_cast<ElementD *>(out_bf16);
    typename GemmKernel::EpilogueArguments epilogue_args{
        Gemm::EpilogueOp::prepare_args(
            activation_scales, weight_scales),
        out,
        stride_c,
        out,
        stride_d};

    cutlass::KernelHardwareInfo hardware_info;
    typename GemmKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        mainloop_args,
        epilogue_args,
        hardware_info,
        {}};
    using GemmOp =
        cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    GemmOp op;
    if (op.can_implement(arguments) != cutlass::Status::kSuccess) {
        return false;
    }
    const size_t required = op.get_workspace_size(arguments);
    if (required > workspace_capacity) {
        return false;
    }
    return op.run(arguments, workspace, stream) ==
           cutlass::Status::kSuccess;
}

} // namespace

bool launch_bf16(void *out_bf16,
                 const void *activation_fp8,
                 const void *weight_fp8,
                 const float *activation_scales,
                 const float *weight_scales,
                 uint32_t rows,
                 uint32_t out_cols,
                 uint32_t inner_cols,
                 void *workspace,
                 size_t workspace_capacity,
                 cudaStream_t stream) {
    using Config = Sm120Gemm<
        cutlass::float_e4m3_t,
        cutlass::bfloat16_t,
        ScaledEpilogue,
        Shape<_128, _128, _128>,
        Shape<_1, _1, _1>,
        cutlass::gemm::collective::KernelScheduleAuto,
        cutlass::epilogue::collective::EpilogueScheduleAuto>;
    return launch<Config>(
        activation_fp8, weight_fp8, out_bf16,
        activation_scales, weight_scales,
        rows, out_cols, inner_cols,
        workspace, workspace_capacity, stream);
}

bool launch_f32(float *out_f32,
                const void *activation_fp8,
                const void *weight_fp8,
                const float *activation_scales,
                const float *weight_scales,
                uint32_t rows,
                uint32_t out_cols,
                uint32_t inner_cols,
                void *workspace,
                size_t workspace_capacity,
                cudaStream_t stream) {
    using Config = Sm120Gemm<
        cutlass::float_e4m3_t,
        float,
        ScaledEpilogue,
        Shape<_128, _128, _128>,
        Shape<_1, _1, _1>,
        cutlass::gemm::collective::KernelScheduleAuto,
        cutlass::epilogue::collective::EpilogueScheduleAuto>;
    return launch<Config>(
        activation_fp8, weight_fp8, out_f32,
        activation_scales, weight_scales,
        rows, out_cols, inner_cols,
        workspace, workspace_capacity, stream);
}

} // namespace qw3::fp8_scaled_mm_adapter
