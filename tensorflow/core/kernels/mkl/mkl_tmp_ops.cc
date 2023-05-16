/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

// This file contains temporary registrations for some of the Eigen CPU backend
// operators for BFloat16 and Float16 type. The kernels registered for all these ops simply
// raise errors. We do this so that MKL graph pass can rewrite these ops into
// corresponding MKL ops. Without such registrations, Placer component in
// TensorFlow fails because Eigen CPU backend does not support these ops in
// BFloat16 or Float16 type.

namespace {
class RaiseIncompatibleDTypeError : public OpKernel {
 public:
  explicit RaiseIncompatibleDTypeError(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES(context, false,
                errors::InvalidArgument("Op does not support bfloat16 or float16 inputs"));
  }

  void Compute(OpKernelContext* context) override {}
};
}  // namespace

#define REGISTER_CPU(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_FusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
      RaiseIncompatibleDTypeError);

TF_CALL_bfloat16(REGISTER_CPU);
TF_CALL_half(REGISTER_CPU);
#undef REGISTER_CPU

}  // namespace tensorflow
