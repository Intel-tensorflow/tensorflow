/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "dnnl.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace dnnl;
using dnnl::batch_normalization_forward;
using dnnl::prop_kind;
using dnnl::stream;
using CPUDevice = Eigen::ThreadPoolDevice;

namespace tensorflow {
template <typename Device, typename T, typename U>
class MklFusedInstanceNormOp : public OpKernel {
 public:
  explicit MklFusedInstanceNormOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    std::vector<int> mean_reduction_axes;
    OP_REQUIRES_OK(context,
                   context->GetAttr("reduction_axes", &mean_reduction_axes));
    OP_REQUIRES(context, InferDataFormat(mean_reduction_axes),
                errors::InvalidArgument(
                    "Failed to infer data format from reduction axes"));
    if (context->HasAttr("use_mean_variance"))
      OP_REQUIRES_OK(
          context, context->GetAttr("use_mean_variance", &use_mean_variance_));
    CheckFusedActivation(context);
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      const Tensor& src_tensor = ctx->input(kSrcIndex);
      const Tensor& scale_tensor = ctx->input(kScaleIndex);
      const Tensor& shift_tensor = ctx->input(kShiftIndex);
      const Tensor& mean_tensor = ctx->input(kMeanIndex);
      const Tensor& variance_tensor = ctx->input(kVarianceIndex);

      OP_REQUIRES(ctx,
                  (src_tensor.dims() == 4 && data_format_ == "NHWC") ||
                      (src_tensor.dims() == 4 && data_format_ == "NCHW") ||
                      (src_tensor.dims() == 5 && data_format_ == "NDHWC") ||
                      (src_tensor.dims() == 5 && data_format_ == "NCDHW"),
                  errors::InvalidArgument(
                      "Unsupported input: ", src_tensor.shape().DebugString(),
                      ", ", data_format_));
      num_elements_scale_ = scale_tensor.NumElements();
      size_t num_elements_shift = shift_tensor.NumElements();
      OP_REQUIRES(ctx, num_elements_scale_ == num_elements_shift,
                  errors::InvalidArgument(
                      "Number of elements in scale and shift",
                      "tensors are not same, got ", num_elements_scale_,
                      " and ", num_elements_shift));

      TensorFormat tensor_format;
      OP_REQUIRES(ctx, FormatFromString(data_format_, &tensor_format),
                  errors::InvalidArgument("Invalid data format"));

      MklDnnThreadPool eigen_tp(ctx);
      std::shared_ptr<stream> engine_stream_ptr;
      engine_stream_ptr.reset(CreateStream(&eigen_tp, cpu_engine_));

      const int batch_size = src_tensor.shape().dim_size(0);
      const int64_t elems_per_batch =
          src_tensor.shape().num_elements() / batch_size;

      memory::dims src_dims =
          (src_tensor.dims() == 5)
              ? TFShapeToMklDnnDimsInNCDHW(src_tensor.shape(), tensor_format)
              : TFShapeToMklDnnDimsInNCHW(src_tensor.shape(), tensor_format);

      // oneDNN has no direct support for instancenorm, use a workaround
      // with performing multiple batchnorm computations for each sample
      // in the batched input.
      src_dims[0] = 1;

      memory::format_tag tag;
      if (data_format_ == "NCHW" || data_format_ == "NCDHW") {
        tag = (src_dims.size() == 5) ? memory::format_tag::ncdhw
                                     : memory::format_tag::nchw;
      } else {
        tag = (src_dims.size() == 5) ? memory::format_tag::ndhwc
                                     : memory::format_tag::nhwc;
      }
      auto src_md = memory::desc(src_dims, MklDnnType<T>(), tag);

      memory::dims scale_shift_dims = {
          2, static_cast<dnnl_dim_t>(num_elements_scale_)};
      auto scale_shift_md = memory::desc(scale_shift_dims, MklDnnType<float>(),
                                         memory::format_tag::nc);
      Tensor scale_shift_tensor;
      int64_t tensor_shape = scale_shift_md.get_size() / sizeof(float);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<float>::v(), {tensor_shape},
                                  &scale_shift_tensor));
      void* scale_shift_buf =
          static_cast<void*>(scale_shift_tensor.flat<float>().data());
      SetupScaleShiftBuffer(ctx, scale_tensor, shift_tensor, engine_stream_ptr,
                            scale_shift_buf);
      auto scale_shift_mem =
          memory(scale_shift_md, cpu_engine_, scale_shift_buf);

      auto flags = normalization_flags::use_scale_shift;
      if (use_mean_variance_) flags |= normalization_flags::use_global_stats;

      // Create oneDNN primitive
      auto bnorm_desc = batch_normalization_forward::desc(
          prop_kind::forward_inference, src_md, epsilon_, flags);
      batch_normalization_forward::primitive_desc bnorm_pd;
      if (fuse_activation_) {
        dnnl::post_ops post_ops;
        dnnl::primitive_attr post_ops_attr;
        post_ops.append_eltwise(1.0, dnnl::algorithm::eltwise_relu,
                                leakyrelu_alpha_, 0.0);
        post_ops_attr.set_post_ops(post_ops);
        bnorm_pd = batch_normalization_forward::primitive_desc(
            bnorm_desc, post_ops_attr, cpu_engine_);
      } else {
        bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_desc,
                                                               cpu_engine_);
      }
      auto bnorm_prim = batch_normalization_forward(bnorm_pd);

      // dst memory
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                              {0}, 0, src_tensor.shape(), &output_tensor));
      void* dst_buf =
          static_cast<void*>(const_cast<T*>(output_tensor->flat<T>().data()));

      std::unique_ptr<dnnl::memory> dst_mem_ptr(
          new memory(src_md, cpu_engine_, (char*)nullptr));
      std::unique_ptr<dnnl::memory> src_mem_ptr(
          new memory(src_md, cpu_engine_, (char*)nullptr));

      const T* src_buf_batch = const_cast<T*>(src_tensor.flat<T>().data());
      const T* dst_buf_batch = const_cast<T*>(output_tensor->flat<T>().data());

      std::unordered_map<int, memory> bnorm_args;
      bnorm_args.insert({DNNL_ARG_SRC, *src_mem_ptr});
      bnorm_args.insert({DNNL_ARG_SCALE_SHIFT, scale_shift_mem});
      bnorm_args.insert({DNNL_ARG_DST, *dst_mem_ptr});

      Tensor scaled_mean_tensor;
      SetupMeanVarianceBuffers(ctx, mean_tensor, variance_tensor,
                               &scaled_mean_tensor, &bnorm_args);

      // Perform batchnorm computation for each batch in input
      for (int i = 0; i < batch_size; i++) {
        src_mem_ptr->set_data_handle(static_cast<void*>(
            const_cast<T*>(src_buf_batch + i * elems_per_batch)));
        dst_mem_ptr->set_data_handle(static_cast<void*>(
            const_cast<T*>(dst_buf_batch + i * elems_per_batch)));
        bnorm_prim.execute(*engine_stream_ptr, bnorm_args);
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
  float epsilon_ = 0.0001f;
  float leakyrelu_alpha_ = 0.2f;
  string data_format_ = "";
  size_t num_elements_scale_ = 0;
  const int kSrcIndex = 0;
  const int kScaleIndex = 1;
  const int kShiftIndex = 2;
  const int kMeanIndex = 3;
  const int kVarianceIndex = 4;
  bool fuse_activation_ = false;

  void CheckFusedActivation(OpKernelConstruction* context) {
    string activation_mode;
    OP_REQUIRES_OK(context,
                   context->GetAttr("activation_mode", &activation_mode));

    OP_REQUIRES(context,
                activation_mode == "Relu" || activation_mode == "LeakyRelu" ||
                    activation_mode == "Identity",
                errors::InvalidArgument("_MklFusedInstanceNorm only supports ",
                                        "Relu and Leaky-Relu activations, got ",
                                        activation_mode));

    if (activation_mode == "Relu") {
      fuse_activation_ = true;
      leakyrelu_alpha_ = 0.0f;
    } else if (activation_mode == "LeakyRelu") {
      fuse_activation_ = true;
      OP_REQUIRES_OK(context,
                     context->GetAttr("leakyrelu_alpha", &leakyrelu_alpha_));
    }
  }

  // Given the reduction axes of mean computation, infer data format of input
  bool InferDataFormat(const std::vector<int>& mean_reduction_axes) {
    bool valid = true;
    if (mean_reduction_axes == std::vector<int>{1, 2}) {
      data_format_ = "NHWC";
    } else if (mean_reduction_axes == std::vector<int>{2, 3}) {
      data_format_ = "NCHW";
    } else if (mean_reduction_axes == std::vector<int>{1, 2, 3}) {
      data_format_ = "NDHWC";
    } else if (mean_reduction_axes == std::vector<int>{2, 3, 4}) {
      data_format_ = "NCDHW";
    } else {
      valid = false;
    }
    return valid;
  }

  // Helper function to add scale and shift data into same buffer in float
  // type as requested by oneDNN
  void SetupScaleShiftBuffer(OpKernelContext* context,
                             const Tensor& scale_tensor,
                             const Tensor& shift_tensor,
                             std::shared_ptr<stream> engine_stream_ptr,
                             void* scale_shift_buf) {
    void* scale_buf_src =
        static_cast<void*>(const_cast<U*>(scale_tensor.flat<U>().data()));
    Tensor scaled_shift;
    void* shift_buf_src =
        GetMeanOffsetDataHandle(context, shift_tensor, &scaled_shift);
    auto scale_offset = sizeof(float) * num_elements_scale_;
    void* scale_buf_dst = scale_shift_buf;
    void* shift_buf_dst = static_cast<char*>(scale_shift_buf) + scale_offset;

    if (std::is_same<U, float>::value) {
      memcpy(scale_buf_dst, scale_buf_src, scale_offset);
      memcpy(shift_buf_dst, shift_buf_src, scale_offset);
    } else {
      // oneDNN requires float type for scale_shift, need to convert to float
      // type
      auto scale_mem_src = memory(
          {{num_elements_scale_}, MklDnnType<U>(), memory::format_tag::x},
          cpu_engine_, scale_buf_src);
      auto scale_mem_dst = memory(
          {{num_elements_scale_}, MklDnnType<float>(), memory::format_tag::x},
          cpu_engine_, scale_buf_dst);
      auto scale_reorder_prim = reorder(scale_mem_src, scale_mem_dst);
      std::unordered_map<int, memory> scale_reorder_args;
      scale_reorder_args.insert({DNNL_ARG_FROM, scale_mem_src});
      scale_reorder_args.insert({DNNL_ARG_TO, scale_mem_dst});
      scale_reorder_prim.execute(*engine_stream_ptr, scale_reorder_args);

      auto shift_mem_src = memory(
          {{num_elements_scale_}, MklDnnType<U>(), memory::format_tag::x},
          cpu_engine_, shift_buf_src);
      auto shift_mem_dst = memory(
          {{num_elements_scale_}, MklDnnType<float>(), memory::format_tag::x},
          cpu_engine_, shift_buf_dst);
      auto shift_reorder_prim = reorder(shift_mem_src, shift_mem_dst);
      std::unordered_map<int, memory> shift_reorder_args;
      shift_reorder_args.insert({DNNL_ARG_FROM, shift_mem_src});
      shift_reorder_args.insert({DNNL_ARG_TO, shift_mem_dst});
      shift_reorder_prim.execute(*engine_stream_ptr, shift_reorder_args);
    }
  }

  void SetupMeanVarianceBuffers(OpKernelContext* context,
                                const Tensor& mean_tensor,
                                const Tensor& variance_tensor,
                                Tensor* scaled_mean_tensor,
                                std::unordered_map<int, memory>* bnorm_args) {
    if (!use_mean_variance_) return;

    OP_REQUIRES(context, mean_tensor.dims() == 1,
                errors::InvalidArgument("mean tensor must be 1-dimensional",
                                        mean_tensor.shape().DebugString()));

    OP_REQUIRES(context, variance_tensor.dims() == 1,
                errors::InvalidArgument("variance tensor must be 1-dimensional",
                                        variance_tensor.shape().DebugString()));

    OP_REQUIRES(context, mean_tensor.NumElements() == num_elements_scale_,
                errors::InvalidArgument(
                    "mean tensor must have the same number of "
                    "elements as scale tensor, got ",
                    mean_tensor.NumElements(), " and ", num_elements_scale_));

    OP_REQUIRES(
        context, variance_tensor.NumElements() == num_elements_scale_,
        errors::InvalidArgument("variance tensor must have the same number of "
                                "elements as scale tensor, got ",
                                variance_tensor.NumElements(), " and ",
                                num_elements_scale_));

    auto mean_var_md = memory::desc({num_elements_scale_}, MklDnnType<U>(),
                                    memory::format_tag::x);
    void* mean_data = static_cast<void*>(const_cast<U*>(
        GetMeanOffsetDataHandle(context, mean_tensor, scaled_mean_tensor)));
    void* variance_data =
        static_cast<void*>(const_cast<U*>(variance_tensor.flat<U>().data()));

    auto mean_mem = memory(mean_var_md, cpu_engine_, mean_data);
    auto variance_mem = memory(mean_var_md, cpu_engine_, variance_data);
    bnorm_args->insert({DNNL_ARG_MEAN, mean_mem});
    bnorm_args->insert({DNNL_ARG_VARIANCE, variance_mem});
  }

 protected:
  bool use_mean_variance_ = false;

  virtual U* GetMeanOffsetDataHandle(OpKernelContext* context,
                                     const Tensor& tensor_in,
                                     Tensor* tensor_out) {
    return static_cast<U*>(const_cast<U*>(tensor_in.flat<U>().data()));
  }

  engine GetEngine() const { return cpu_engine_; }

  size_t GetNumScaleElems() const { return num_elements_scale_; }
};

template <typename Device, typename T, typename U>
class QuantizedFusedInstanceNormOp
    : public MklFusedInstanceNormOp<Device, T, U> {
 public:
  explicit QuantizedFusedInstanceNormOp(OpKernelConstruction* context)
      : MklFusedInstanceNormOp<Device, T, U>(context) {
    DataType input_dt;
    OP_REQUIRES_OK(context, context->GetAttr("T", &input_dt));
    OP_REQUIRES(context, input_dt == DT_QINT8,
                errors::InvalidArgument("_QuantizedFusedInstanceNorm only "
                                        "supports qint8 input data type."));

    DataType out_dt;
    OP_REQUIRES_OK(context, context->GetAttr("Tout", &out_dt));
    OP_REQUIRES(
        context, out_dt == DT_QINT8,
        errors::InvalidArgument("_QuantizedFusedInstanceNorm only supports "
                                "qint8 output data type."));

    this->use_mean_variance_ = true;

    // Inputs to this op are expected as follows, {} means optional inputs:
    // 0. x
    // 1. scale
    // 2. offset
    // 3. mean
    // 4. variance
    // 5. x_min
    // 6. x_max
    // 7. {output_min}
    // 8. {output_max}

    const int actual_num_inputs = context->num_inputs();
    const int expected_num_inputs = 9;

    OP_REQUIRES(context, actual_num_inputs == expected_num_inputs,
                errors::InvalidArgument("_QuantizedFusedInstanceNorm: number "
                                        "of inputs is invalid, expected ",
                                        expected_num_inputs, ", got ",
                                        actual_num_inputs));
  }

  void Compute(OpKernelContext* context) override {
    MklFusedInstanceNormOp<Device, T, U>::Compute(context);
    Tensor* output_min = nullptr;
    Tensor* output_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &output_min));
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &output_max));
    output_min->flat<float>()(0) = context->input(5).flat<float>()(0);
    output_max->flat<float>()(0) = context->input(6).flat<float>()(0);
  }

  U* GetMeanOffsetDataHandle(OpKernelContext* context, const Tensor& tensor_in,
                             Tensor* tensor_out) override {
    // Scale offset tensor using reorder
    const float min = context->input(5).flat<float>()(0);
    const float max = context->input(6).flat<float>()(0);
    const float max_abs = std::max(std::abs(min), std::abs(max));
    const float scale = 127.0f / max_abs;

    dnnl::primitive_attr scale_attr;
    scale_attr.set_output_scales(0, {scale});

    auto input_md = memory::desc({1, this->GetNumScaleElems()}, MklDnnType<U>(),
                                 memory::format_tag::nc);
    U* input_buf = static_cast<U*>(const_cast<U*>(tensor_in.flat<U>().data()));

    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<U>::value,
                                       tensor_in.shape(), tensor_out));
    U* scaled_input_buf =
        static_cast<U*>(const_cast<U*>(tensor_out->flat<U>().data()));

    const auto& engine = this->GetEngine();
    std::shared_ptr<memory> input_mem_(new memory(input_md, engine, input_buf));
    std::shared_ptr<memory> scaled_input_mem_(
        new memory(input_md, engine, scaled_input_buf));

    auto reorder_desc =
        ReorderPd(engine, input_md, engine, input_md, scale_attr);
    CreateAndExecuteReorder(reorder_desc, *input_mem_, *scaled_input_mem_,
                            engine, context);
    return scaled_input_buf;
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_MklFusedInstanceNorm").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MklFusedInstanceNormOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("_MklFusedInstanceNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<bfloat16>("T"),
                        MklFusedInstanceNormOp<CPUDevice, bfloat16, bfloat16>);

REGISTER_KERNEL_BUILDER(Name("_QuantizedFusedInstanceNorm")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("T")
                            .TypeConstraint<float>("U")
                            .TypeConstraint<qint8>("Tout"),
                        QuantizedFusedInstanceNormOp<CPUDevice, qint8, float>);

}  // namespace tensorflow

#endif  // INTEL_MKL
