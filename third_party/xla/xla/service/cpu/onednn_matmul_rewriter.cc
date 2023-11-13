/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tsl/util/onednn_threadpool.h"
#include "xla/service/cpu/onednn_matmul_rewriter.h"

#include "xla/executable_run_options.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_matmul.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;
using dnnl::engine;
using dnnl::stream;

Status ValidateDotDimensionNumbers(const DotDimensionNumbers& dim_numbers) {
  // Checks some invariants that do not hold in general, but DotDecomposer
  // should have established for us.
  TF_RET_CHECK(dim_numbers.lhs_contracting_dimensions_size() == 1);
  std::vector<int64_t> batch_dim_numbers(
      dim_numbers.lhs_batch_dimensions_size());
  absl::c_iota(batch_dim_numbers, 0);
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.lhs_batch_dimensions()));
  TF_RET_CHECK(
      absl::c_equal(batch_dim_numbers, dim_numbers.rhs_batch_dimensions()));
  return OkStatus();
}

template <typename Pattern>
auto IntermediateAllowedInstructions(HloInstruction** opt, Pattern pattern) {
  // Checks the presence of some intermediate operations that can be moved /
  // folded to allow dot fusion with add.
  return m::AnyOf<HloInstruction>(
             m::Bitcast(opt,
                        m::AnyOf<HloInstruction>(
                            m::Convert(m::AnyOf<HloInstruction>(
                                           m::Convert(pattern).WithElementType(
                                               PrimitiveType::BF16),
                                           std::move(pattern))
                                           .WithOneUser())
                                .WithElementType(PrimitiveType::F32),
                            m::AnyOf<HloInstruction>(
                                m::Convert(pattern).WithElementType(
                                    PrimitiveType::BF16),
                                std::move(pattern))
                                .WithOneUser())
                            .WithOneUser()),
             m::AnyOf<HloInstruction>(
                 m::Convert(m::AnyOf<HloInstruction>(
                                m::Convert(pattern).WithElementType(
                                    PrimitiveType::BF16),
                                std::move(pattern))
                                .WithOneUser())
                     .WithElementType(PrimitiveType::F32),
                 m::AnyOf<HloInstruction>(
                     m::Convert(pattern).WithElementType(PrimitiveType::BF16),
                     std::move(pattern))
                     .WithOneUser())
                 .WithOneUser())
      .WithOneUser();
}

// We also check if the convert instruction has only one use.
bool AllOperandsConvertedFromBF16ToF32(const HloInstruction* instr) {
  return absl::c_all_of(instr->operands(), [](HloInstruction* operand) {
    return Match(operand,
                 m::Convert(m::Op().WithElementType(PrimitiveType::BF16))
                     .WithElementType(PrimitiveType::F32)
                     .WithOneUse());
  });
}

// FIRME(intel-tf): GELU formula prevents .WithOneUser() at the end of pattern
template<typename Pattern>
auto ElementwiseSafeIntermediate(HloInstruction** instr, Pattern pattern) {
  return m::AnyOf<HloInstruction>(
    m::Broadcast(instr, pattern.WithOneUser()),
    m::Slice(instr, pattern.WithOneUser()),
    m::Bitcast(instr, pattern.WithOneUser()),
    m::Reshape(instr, pattern.WithOneUser()),
    pattern
  );
}

auto OneDnnMatmul(HloInstruction **instr) {
    return m::CustomCall(instr, {"__onednn$matmul"});
}

auto ConvertPattern(HloInstruction **instr) {
  return m::Convert(m::Op(instr)
                    .WithElementType(PrimitiveType::BF16)).WithElementType(
                    PrimitiveType::F32);
}

void GetBF16Bias(HloInstruction *dot, HloInstruction **old_bias, HloInstruction **new_bias) {
  if (dot->shape().element_type() == PrimitiveType::BF16 &&
    (((*old_bias)->operand_count() == 1 &&
      Match((*old_bias)->mutable_operand(0), ConvertPattern(new_bias))) ||
      Match(*old_bias, ConvertPattern(new_bias)))) {
    *old_bias = *new_bias;
  }
}

auto BcastConstScalar(HloInstruction **instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}

auto BcastConstScalar(double value) { return BcastConstScalar(nullptr, value); }

auto BcastConstScalarNear(double value) {
  return m::Broadcast(m::ConstantScalar().WithPredicate(
      [expected = value](const HloInstruction *instr) {
        // Not a very robust floating-point comparison, but good enough for our
        // purposes.
        std::optional<double> actual =
            static_cast<const HloConstantInstruction *>(instr)
                ->literal()
                .GetAsDouble({});
        if (!actual.has_value()) return false;
        double epsilon;
        switch (instr->shape().element_type()) {
          case F16:
            epsilon = 128 * std::numeric_limits<Eigen::half>::epsilon();
            break;
          case BF16:
            epsilon = 128 * std::numeric_limits<bfloat16>::epsilon();
            break;
          case F32:
            epsilon = 128 * std::numeric_limits<float>::epsilon();
            break;
          case F64:
            epsilon = 128 * std::numeric_limits<double>::epsilon();
            break;
          default:
            return false;
        }
        // return abs(*actual - expected) < (abs(*actual + expected) * epsilon);
        // Temporarily relax the comparison requirement since the above one seems too restrictive.
        return abs(*actual - expected) <= 1e-2;
      }));
}

bool IsScalar(const HloInstruction* instr) {
  return ShapeUtil::IsEffectiveScalar(instr->shape());
}

std::optional<float> GetConstantValue(const HloInstruction* inst) {
  if (!IsScalar(inst)) {
    return std::nullopt;
  }
  switch (inst->shape().element_type()) {
    case F16:
      return inst->literal().GetFirstElement<half>();
    case BF16:
      return inst->literal().GetFirstElement<bfloat16>();
    case F32:
      return inst->literal().GetFirstElement<float>();
    default:
      return std::nullopt;
  }
}

auto GELUActivation(HloInstruction *instr, HloInstruction **src) {
  // Attempt to match GELU_TANH activation
  // (https://arxiv.org/abs/1606.08415), where:
  // gelu_tanh(x) = x * cdf(x)
  // cdf(x) = 0.5 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3))
  HloInstruction *errf;
  return
    Match(instr,
      m::MultiplyAnyOrder(
        m::Op(src),
        m::MultiplyAnyOrder(
          BcastConstScalar(0.5),
          m::AddAnyOrder(
            BcastConstScalar(1.0),
            m::Op(&errf).WithOneUser()
          )
        )
      )
    ) &&
    Match(errf,
      m::Tanh(
        m::MultiplyAnyOrder(
          BcastConstScalarNear(sqrt(M_2_PI)), // For GPTJ BF16 run actual value is 0.796875
          m::AddAnyOrder(
            m::Op().Is(*src),
            m::MultiplyAnyOrder(
              BcastConstScalarNear(0.044715), // For GPTJ BF16 run value is 0.0446777344
              m::MultiplyAnyOrder(
                m::Op().Is(*src),
                m::MultiplyAnyOrder(
                    m::Op().Is(*src),
                    m::Op().Is(*src))
                    .WithOneUser())
                .WithOneUser())
              .WithOneUser())
            .WithOneUser())
          .WithOneUser())
        .WithOneUser());
}

}  // namespace

bool OneDnnMatMulRewriter::ShouldRewrite(const HloInstruction* dot_instr) {
  // Currently, blocking control dependencies
  if (dot_instr->HasControlDependencies()) return false;
  if (!IsSupportedType(dot_instr->shape().element_type())) return false;

  // Currently, we rewrite when the data type is F32 or BF16. Note we do not
  // need to check equality of contraction dim-size of the operands. HLO
  // verifier already does the job. We, however, need to check if contraction
  // is over only 1 dimension (a.k.a. K dimension in matrix-multiplication
  // parlance). We also restrict that batch dimensions of the operands
  // matches.
  const Shape& lhs_shape = dot_instr->operand(0)->shape();
  const Shape& rhs_shape = dot_instr->operand(1)->shape();
  const Shape& output_shape = dot_instr->shape();
  bool should_rewrite = true;
  // None of the operands and result should be ZeroElementArray.
  should_rewrite &= !ShapeUtil::IsZeroElementArray(lhs_shape);
  should_rewrite &= !ShapeUtil::IsZeroElementArray(rhs_shape);
  should_rewrite &= !ShapeUtil::IsZeroElementArray(output_shape);
  // OneDNN only supports 2 <= rank <= kOneDnnMaxNDims.
  should_rewrite &= (lhs_shape.rank() == rhs_shape.rank());
  should_rewrite &= (rhs_shape.rank() == output_shape.rank());
  should_rewrite &=
      (lhs_shape.rank() >= 2 && lhs_shape.rank() <= kOneDnnMaxNDims);
  if (!should_rewrite) return false;
  // Layout should be row-major, contraction dimensions captures transpose
  // scenarios in last two dimensions.
  should_rewrite &= LayoutUtil::IsMonotonicWithDim0Major(lhs_shape.layout());
  if (!should_rewrite) return false;
  should_rewrite &= LayoutUtil::IsMonotonicWithDim0Major(rhs_shape.layout());
  if (!should_rewrite) return false;
  should_rewrite &=
      LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout());
  if (!should_rewrite) return false;

  auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
  int64_t lhs_dim_k = dot_dim_numbers.lhs_contracting_dimensions(0);
  int64_t rhs_dim_k = dot_dim_numbers.rhs_contracting_dimensions(0);
  // Supported contraction is only in one of last two dimensions.
  should_rewrite &= (lhs_dim_k >= lhs_shape.rank() - 2);
  should_rewrite &= (rhs_dim_k >= rhs_shape.rank() - 2);
  return should_rewrite;
  }

class OneDnnMatMulRewriteVisitor : public DfsHloRewriteVisitor {
 public:
  // Matches patterns for possible MatMul fusions that are supported by oneDNN
  // library. Matched hlo instruction(s) are replaced by custom call.
  Status HandleDot(HloInstruction* instr) override {
    HloInstruction* dot_instr;
    auto pattern = m::Op(&dot_instr).WithOpcode(HloOpcode::kDot);
    if (!Match(instr, pattern)) return OkStatus();

    auto dot_dim_numbers = dot_instr->dot_dimension_numbers();
    TF_RETURN_IF_ERROR(ValidateDotDimensionNumbers(dot_dim_numbers));
    if (!OneDnnMatMulRewriter::ShouldRewrite(dot_instr)) return OkStatus();
    const Shape& lhs_shape = dot_instr->operand(0)->shape();
    const Shape& rhs_shape = dot_instr->operand(1)->shape();
    const Shape& output_shape = dot_instr->shape();

    int64_t lhs_dim_k = dot_dim_numbers.lhs_contracting_dimensions(0);
    int64_t rhs_dim_k = dot_dim_numbers.rhs_contracting_dimensions(0);

    HloInstruction* matmul_call =
        dot_instr->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape,
            {dot_instr->mutable_operand(0), dot_instr->mutable_operand(1)},
            "__onednn$matmul"));
    // Set additional info via config, e.g., transpose and fusion info.
    BackendConfig backend_config;
    OneDnnMatMulConfig* matmul_config =
        backend_config.mutable_onednn_matmul_config();
    bool transpose_a = (lhs_dim_k == lhs_shape.rank() - 1) ? false : true;
    bool transpose_b = (rhs_dim_k == rhs_shape.rank() - 2) ? false : true;
    matmul_config->set_transpose_a(transpose_a);
    matmul_config->set_transpose_b(transpose_b);
    TF_RETURN_IF_ERROR(matmul_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot_instr, matmul_call));
    return OkStatus();
  }

  Status HandleConvert(HloInstruction* convert) override {
    HloInstruction* matmul_instr;
    auto pattern =
        m::Convert(m::CustomCall(&matmul_instr, {"__onednn$matmul"})
                       .WithOneUse()
                       .WithElementType(PrimitiveType::F32)
                       .WithPredicate(AllOperandsConvertedFromBF16ToF32))
            .WithElementType(PrimitiveType::BF16);

    if (!Match(convert, pattern)) return OkStatus();
    if (!IsSupportedType(convert->shape().element_type())) return OkStatus();

    // BFloat16 operands.
    std::vector<HloInstruction*> bf16_operands;
    for (auto operand : matmul_instr->operands()) {
      bf16_operands.push_back(operand->mutable_operand(0));
    }

    HloInstruction* matmul_call = convert->AddInstruction(
        matmul_instr->CloneWithNewOperands(convert->shape(), bf16_operands));
    TF_RETURN_IF_ERROR(ReplaceInstruction(convert, matmul_call));
    return OkStatus();
  }

  Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *addend, *dot, *link, *inter;
    HloInstruction* opt = nullptr;

    auto pattern =
        m::Op(&instr)
            .WithOpcode(HloOpcode::kAdd)
            .WithBinaryOperandsAnyOrder(
                IntermediateAllowedInstructions(
                    &opt, m::Op(&dot)
                              .WithOneUser()
                              .WithOpcode(HloOpcode::kCustomCall)
                              .WithCustomCallTarget({"__onednn$matmul"})),
                m::Op(&addend).WithOneUser());

    auto nonscalar_broadcast =
        m::Broadcast(m::Op(&link)
                         .WithPredicate([](const HloInstruction* ins) {
                           return !ShapeUtil::IsEffectiveScalar(ins->shape());
                         })
                         .WithOneUser())
            .WithOneUser();

    auto addend_reshape = m::Bitcast(m::Op(&inter).WithOneUser()).WithOneUser();

    if (Match(instr, pattern)) {
      if (!IsSupportedType(dot->shape().element_type())) return OkStatus();
      if (!dot->backend_config<BackendConfig>()
               ->mutable_onednn_matmul_config()
               ->fused_ops()
               .empty() &&
          dot->backend_config<BackendConfig>()
                  ->mutable_onednn_matmul_config()
                  ->fused_ops(0) == OneDnnMatMulConfig::BIAS) {
        return OkStatus();
      }
      std::vector<HloInstruction*> new_operands;
      for (auto operand : dot->operands()) {
        new_operands.push_back(operand);
      }
      bool bias_broad = Match(addend, nonscalar_broadcast);
      bool check_addend = Match(addend, addend_reshape);

      HloInstruction *bf16_addend, *bf16_link, *bf16_inter;
      GetBF16Bias(dot, &addend, &bf16_addend);
      if (bias_broad) {
          GetBF16Bias(dot, &link, &bf16_link);
      }
      if (check_addend) {
          GetBF16Bias(dot, &inter, &bf16_inter);
      }

      bias_broad
          ? new_operands.push_back(link)
          : (absl::c_equal(dot->shape().dimensions(),
                           addend->shape().dimensions()))
                ? new_operands.push_back(addend)
                : (check_addend && absl::c_equal(dot->shape().dimensions(),
                                                 inter->shape().dimensions()))
                      ? new_operands.push_back(inter)
                      : new_operands.push_back(nullptr);

      if (new_operands.back() == nullptr) {
        return OkStatus();
      }

      auto matmul_call = Cast<HloCustomCallInstruction>(instr->AddInstruction(
          dot->CloneWithNewOperands(dot->shape(), new_operands)));

      auto backend_config = matmul_call->backend_config<BackendConfig>();
      backend_config->mutable_onednn_matmul_config()->add_fused_ops(
          OneDnnMatMulConfig::BIAS);
      backend_config->mutable_onednn_matmul_config()->set_bias_broadcast(
          bias_broad);

      TF_RETURN_IF_ERROR(matmul_call->set_backend_config(*backend_config));

      HloInstruction *new_instr;
      if (opt != nullptr && opt->opcode() == HloOpcode::kBitcast) {
        if (matmul_call->shape().element_type() == PrimitiveType::BF16) {
          auto bitcast_call = matmul_call->AddInstruction(
              HloInstruction::CreateReshape(ShapeUtil::ChangeElementType(instr->shape(),
                                            PrimitiveType::BF16), matmul_call));
          new_instr = bitcast_call->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(bitcast_call->shape(), PrimitiveType::F32),
            bitcast_call
          ));
        } else {
          new_instr = matmul_call->AddInstruction(
              HloInstruction::CreateReshape(instr->shape(), matmul_call));
        }
      } else {
        if (matmul_call->shape().element_type() == PrimitiveType::BF16) {
          new_instr = matmul_call->AddInstruction(
            HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(matmul_call->shape(), PrimitiveType::F32),
              matmul_call)
          );
        } else {
          new_instr = matmul_call;
        }
      }
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_instr));
    }

    return OkStatus();
  }

  Status HandleMaximum(HloInstruction *instr) override {
    HloInstruction *matmul_call;
    HloInstruction *slice_or_bitcast = nullptr;
    // Attempt to elide maximum and fuse ReLU activation into GEMM, including
    // when slicing or bitcasting is applied to the result.
    if (Match(instr,
              m::MaximumAnyOrder(
                  ElementwiseSafeIntermediate(&slice_or_bitcast,
                      OneDnnMatmul(&matmul_call)
                  ).WithOneUser(),
                  BcastConstScalar(0)))) {
      return FuseActivation(OneDnnMatMulConfig::RELU, instr, matmul_call, slice_or_bitcast);
    }
    return OkStatus();
  }

  Status HandleMultiply(HloInstruction *instr) override {
    HloInstruction *matmul_call;
    HloInstruction *slice_or_bitcast = nullptr;
    HloInstruction *src, *bf16_src;
    if (GELUActivation(instr, &src)) {
        if (Match(src, ElementwiseSafeIntermediate(
                        &slice_or_bitcast,
                        OneDnnMatmul(&matmul_call))) ||
           (Match(src, ConvertPattern(&bf16_src)) &&
            Match(src->mutable_operand(0), ElementwiseSafeIntermediate(
                        &slice_or_bitcast,
                        OneDnnMatmul(&matmul_call))))) {
          return FuseActivation(OneDnnMatMulConfig::GELU_TANH, instr, matmul_call, slice_or_bitcast);
        }
    }

    HloInstruction *dot, *constant;

    auto pattern = m::Op(&instr)
                           .WithOpcode(HloOpcode::kMultiply)
                           .WithBinaryOperandsAnyOrder(m::Op(&dot)
                           .WithOneUser()
                           .WithOpcode(HloOpcode::kCustomCall)
                           .WithCustomCallTarget({"__onednn$matmul"}),
                           m::Broadcast(m::Constant(&constant))
                                .WithOneUser());

    if (Match(instr, pattern)) {
      std::vector<HloInstruction*> new_operands;
      auto constant_value = *GetConstantValue(constant);

      for (auto operand : dot->operands()) {
        new_operands.push_back(operand);
      }

      auto matmul_call = Cast<HloCustomCallInstruction>(instr->AddInstruction(
        dot->CloneWithNewOperands(instr->shape(), new_operands)));

      auto backend_config = matmul_call->backend_config<BackendConfig>();
      backend_config->mutable_onednn_matmul_config()
                    ->add_fused_ops(OneDnnMatMulConfig::LINEAR);
      backend_config->mutable_onednn_matmul_config()
                    ->set_constant_val(*(reinterpret_cast<int32_t*>(&constant_value)));
      TF_RETURN_IF_ERROR(matmul_call->set_backend_config(*backend_config));
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, matmul_call));
    }
    return OkStatus();
  }

  Status FuseActivation(OneDnnMatMulConfig_FusionKind kind, HloInstruction *activation,
                        HloInstruction *matmul, HloInstruction *slice_or_bitcast = nullptr) {

    TF_ASSIGN_OR_RETURN(auto backend_config, matmul->backend_config<BackendConfig>());
    auto* matmul_config = backend_config.mutable_onednn_matmul_config();
    matmul_config->add_fused_ops(kind);

    std::unique_ptr<HloInstruction> output = matmul->Clone();
    TF_RETURN_IF_ERROR(output->set_backend_config(backend_config));

    if (slice_or_bitcast) {
      output = slice_or_bitcast->CloneWithNewOperands(
          slice_or_bitcast->shape(),
          {matmul->parent()->AddInstruction(std::move(output))});
    }

    if (activation->shape().element_type() != output->shape().element_type()) {
      auto shape = output->shape();
      auto convert = matmul->parent()->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(shape, activation->shape().element_type()),
              matmul->parent()->AddInstruction(std::move(output))));
      TF_RETURN_IF_ERROR(ReplaceInstruction(activation, convert));
      return OkStatus();
    }

    return ReplaceWithNewInstruction(activation, std::move(output));
  }

};

class OneDnnMatMulReorderVisitor : public DfsHloRewriteVisitor {
 public:
  OneDnnMatMulReorderVisitor(const Eigen::ThreadPoolDevice* threadpool_device)
    : threadpool_device_(threadpool_device) {}

  Status HandleCustomCall(HloInstruction* custom_call) override {
    HloInstruction *matmul;
    if (Match(custom_call, OneDnnMatmul(&matmul))) {
      TF_ASSIGN_OR_RETURN(auto backend_config,
                          matmul->backend_config<BackendConfig>());
      auto& matmul_config = backend_config.onednn_matmul_config();

      auto operands = custom_call->operands();
      auto input = operands[0];
      auto weight = operands[1]; // assuming weights is the second operand

      auto input_shape  = input->shape();
      auto weight_shape = weight->shape();
      if (weight_shape.rank() != 2) {
        // pre-pack only 2D weights
        return DefaultAction(custom_call);
      }

      auto bias_shape = operands.size() > 2 ? operands[2]->shape() : Shape();
      auto output_shape = custom_call->shape();

      engine cpu_engine(engine::kind::cpu, 0);
#ifndef ENABLE_ONEDNN_OPENMP
      tsl::OneDnnThreadPool thread_pool;
      auto onednn_stream = [&]{
        if (threadpool_device_->getPool() != nullptr) {
          thread_pool = tsl::OneDnnThreadPool(threadpool_device_->getPool(), false);
          return stream(dnnl::threadpool_interop::make_stream(cpu_engine,
                                                              &thread_pool));
        } else {
        return stream(cpu_engine);
        }
      }();
#else
      auto onednn_stream = stream(cpu_engine);
#endif  // ENABLE_ONEDNN_OPENMP

      auto new_weight_shape = OneDnnMatMulOptWeightsShape(input_shape,
                                                          weight_shape,
                                                          bias_shape,
                                                          output_shape,
                                                          &matmul_config);


      auto cmpt = custom_call->parent();
      std::vector<HloInstruction*> new_operands {
        cmpt->AddInstruction(
                HloInstruction::CreateConstant(Literal(input_shape))),
        weight,
        cmpt->AddInstruction(
                HloInstruction::CreateConstant(Literal(output_shape))),
      };

      if (ShapeUtil::IsInitialized(bias_shape)) {
        new_operands.push_back(
          cmpt->AddInstruction(
            HloInstruction::CreateConstant(Literal(bias_shape))));
      }

      HloInstruction* reorder_call =
        custom_call->AddInstruction(HloInstruction::CreateCustomCall(
            new_weight_shape,
            new_operands,
            "__onednn$matmul_reorder"));

      reorder_call->CopyBackendConfigFrom(custom_call);

      Literal result;
      HloEvaluator evaluator(/*max_loop_iterations=*/0);
      auto custom_call_handler = [&matmul_config, this](
                                     const HloInstruction* custom_call_instr,
                                     absl::Span<const Literal*> operands) {
        auto output = Literal::CreateFromShape(custom_call_instr->shape());

        int64_t nargs = operands.size() + 3;
        std::vector<void*> args;
        args.push_back(&nargs);

        ExecutableRunOptions run_options;
        run_options.set_intra_op_thread_pool(threadpool_device_);
        args.push_back(&run_options);  // No ExecutableRunOptions.

        // OneDnnMatMulConfig
        std::string config;
        matmul_config.SerializeToString(&config);
        args.push_back(config.data());

        std::vector<MemrefInfoHandler> minfo_ptrs(operands.size());
        std::transform(operands.begin(), operands.end(), minfo_ptrs.begin(), CreateMemrefInfoFromLiteral);
        for (auto& minfo_ptr : minfo_ptrs) {
          args.push_back(static_cast<void*>(minfo_ptr.get()));
        }

        auto result_ptr = CreateMemrefInfoFromLiteral(&output);
        __xla_cpu_runtime_OneDnnMatMulReorder(result_ptr.get(), args.data());

        return output;
      };

      evaluator.set_custom_call_handler(custom_call_handler);
      if (evaluator.TryEvaluate(reorder_call, &result, true)) {
        HloInstruction* reordered_weight = custom_call->AddInstruction(
            HloInstruction::CreateConstant(std::move(result)));
        return custom_call->ReplaceOperandWithDifferentShape(1,
                                                             reordered_weight);

      } else {
        return DefaultAction(custom_call);
      }
    }
    return DefaultAction(custom_call);
  }

 private:
  const Eigen::ThreadPoolDevice* threadpool_device_;
};

StatusOr<bool> OneDnnMatMulRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnMatMulRewriteVisitor visitor;
  TF_ASSIGN_OR_RETURN(auto result, visitor.RunOnModule(module,
                                                       execution_threads));

  OneDnnMatMulReorderVisitor reorder_visitor(threadpool_device_);
  TF_ASSIGN_OR_RETURN(auto result2, reorder_visitor.RunOnModule(module,
                                                       execution_threads));

  return {result || result2};
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
