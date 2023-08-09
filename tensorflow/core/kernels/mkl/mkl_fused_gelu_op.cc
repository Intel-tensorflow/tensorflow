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

// See docs in ../ops/mkl_nn_ops.cc.

#ifdef INTEL_MKL

#include "tensorflow/core/kernels/mkl/mkl_eltwise_activation_base_op.h"

namespace tensorflow {

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440 /* 1/sqrt(2) */
#endif

// TODO: Add implementation for gelu approximate using tanh
template <typename Device, typename T>
class MklFusedGeluOp
    : public MklEltwiseFwdActivationOpBase<Device, T,
                                           dnnl::algorithm::eltwise_gelu_erf> {
 public:
  ~MklFusedGeluOp() {}

  explicit MklFusedGeluOp(OpKernelConstruction* context)
      : MklEltwiseFwdActivationOpBase<Device, T, dnnl::algorithm::eltwise_gelu_erf>(
            context, 0.0f, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const Tensor& src_tensor = context->input(0);
    // Get input tensor shape
    const TensorShape src_shape = src_tensor.shape();

    Tensor* dst_tensor = nullptr;
    void* src_buf =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));

    const TensorShape dst_shape = src_shape;
    OP_REQUIRES_OK(context, context->allocate_output(
                                GetTensorDataIndex(0, context->num_outputs()),
                                dst_shape, &dst_tensor));

    T* dst_buf = static_cast<T*>(dst_tensor->flat<T>().data());
    T features = (static_cast<T*>(src_buf))[0];
    // y = x * normcdf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    (static_cast<T*>(dst_buf))[0] = static_cast<T>(0.5) * features *
                (static_cast<T>(1) +
                Eigen::numext::erf(features * static_cast<T>(M_SQRT1_2)));
    return;

  }
};

// register dnn kernels for supported operations and supported types
#define REGISTER_GELU_MKL_SUPPORTED_KERNELS_TYPES(type)                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklFusedGelu")                                                  \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T"),                                         \
      MklFusedGeluOp<CPUDevice, type>);
TF_CALL_float(REGISTER_GELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_GELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_half(REGISTER_GELU_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
