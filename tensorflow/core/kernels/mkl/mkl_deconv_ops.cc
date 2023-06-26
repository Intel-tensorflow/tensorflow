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

// See docs in ../ops/nn_ops.cc. This opkernel uses MKL library, create MKL
// layout and primitives, use MKL dnn primitives to compute convolution backward
// input

#ifdef INTEL_MKL

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/mkl/mkl_deconv_ops.h"

#include <algorithm>
#include <exception>
#include <vector>

#include "absl/strings/str_join.h"
#include "dnnl.hpp"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

using dnnl::deconvolution_forward;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {

#ifdef ENABLE_ONEDNN_V2
#define APPEND_ELTWISE(scale, alg, alpha, beta) \
  append_eltwise(scale, alg, alpha, beta)
#else
#define APPEND_ELTWISE(scale, alg, alpha, beta) append_eltwise(alg, alpha, beta)
#endif  // !ENABLE_ONEDNN_V2

using DeconvFwdPd = dnnl::deconvolution_forward::primitive_desc;

// Utility classes for enabling primitive reuse for deconv forward.
struct MklDeconvFwdParams {
  memory::dims dst_dims;
  memory::dims filter_dims;
  memory::dims bias_dims;
  memory::dims src_dims;
  memory::dims strides;
  memory::dims dilations;
  memory::dims padding_left;
  memory::dims padding_right;
  memory::dims fuse_bn_dims;
  memory::format_tag fmt_tag;
  string dtypes = string("");
  struct PostOpParam {
    string name;
    dnnl::algorithm alg;
    std::vector<float> param;
    std::string partial_key;
  };
  std::vector<PostOpParam> post_op_params;

  MklDeconvFwdParams(memory::dims dst_dims, memory::dims filter_dims,
                     memory::dims bias_dims, memory::dims src_dims,
                     memory::dims strides, memory::dims dilations,
                     memory::dims padding_left, memory::dims padding_right,
                     memory::dims fuse_bn_dims, memory::format_tag fmt_tag)
      : dst_dims(dst_dims),
        filter_dims(filter_dims),
        bias_dims(bias_dims),
        src_dims(src_dims),
        strides(strides),
        dilations(dilations),
        padding_left(padding_left),
        padding_right(padding_right),
        fuse_bn_dims(fuse_bn_dims),
        fmt_tag(fmt_tag) {}
};

template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput>
class MklDeconvFwdPrimitive : public MklPrimitive {
 public:
  explicit MklDeconvFwdPrimitive(const MklDeconvFwdParams& deconvFwdParams)
      : MklPrimitive(engine(engine::kind::cpu, 0)) {
    // Create deconv fwd primitive
    if (context_.deconv_fwd == nullptr) {
      Setup(deconvFwdParams);
    }
  }

  ~MklDeconvFwdPrimitive() {}

  dnnl::memory::desc GetScratchPadDesc() {
    return context_.fwd_pd->scratchpad_desc();
  }

  // Deconvolution forward execution.
  //   src_data: input data buffer for src
  //   filter_data: input data buffer for filter (weights)
  //   Bias_data: bias_data
  //   dst_data: output data buffer for dst
  void Execute(const Tinput* src_data, const Tfilter* filter_data,
               const Tbias* bias_data, const Toutput* dst_data,
               const float* bn_scale_data, const float* bn_mean_data,
               const float* bn_offset_data, const float* bn_rsqrt_data,
               std::shared_ptr<stream> fwd_stream, void* sp_data = nullptr) {
#if !defined(ENABLE_ONEDNN_OPENMP) && defined(ENABLE_ONEDNN_V2)
    context_.src_mem->set_data_handle(
        static_cast<Tinput*>(const_cast<Tinput*>(src_data)), *fwd_stream);
    context_.filter_mem->set_data_handle(
        static_cast<Tfilter*>(const_cast<Tfilter*>(filter_data)), *fwd_stream);
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(
          static_cast<void*>(const_cast<Tbias*>(bias_data)), *fwd_stream);
    }
    if (bn_scale_data != nullptr) {
      context_.bn_scale_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_scale_data)), *fwd_stream);
      context_.bn_mean_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_mean_data)), *fwd_stream);
      context_.bn_rsqrt_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_rsqrt_data)), *fwd_stream);
      context_.bn_offset_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_offset_data)), *fwd_stream);
    }
    context_.dst_mem->set_data_handle(
        static_cast<Toutput*>(const_cast<Toutput*>(dst_data)), *fwd_stream);
    if (sp_data) {
      context_.sp_mem->set_data_handle(static_cast<void*>(sp_data),
                                       *fwd_stream);
    }
#else
    context_.src_mem->set_data_handle(
        static_cast<Tinput*>(const_cast<Tinput*>(src_data)));
    context_.filter_mem->set_data_handle(
        static_cast<Tfilter*>(const_cast<Tfilter*>(filter_data)));
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(
          static_cast<void*>(const_cast<Tbias*>(bias_data)));
    }
    if (bn_scale_data != nullptr) {
      context_.bn_scale_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_scale_data)));
      context_.bn_mean_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_mean_data)));
      context_.bn_rsqrt_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_rsqrt_data)));
      context_.bn_offset_mem->set_data_handle(
          static_cast<void*>(const_cast<float*>(bn_offset_data)));
    }
    context_.dst_mem->set_data_handle(
        static_cast<Toutput*>(const_cast<Toutput*>(dst_data)));
    if (sp_data) context_.sp_mem->set_data_handle(static_cast<void*>(sp_data));
#endif  // !ENABLE_ONEDNN_OPENMP && !ENABLE_ONEDNN_V2

    // if (sp_data) {
    //   context_.sp_mem->set_data_handle(static_cast<void*>(sp_data),
    //                                    *fwd_stream);
    // }
    context_.fwd_primitives.at(0).execute(*fwd_stream,
                                          context_.fwd_primitives_args.at(0));

    context_.src_mem->set_data_handle(DummyData);
    context_.filter_mem->set_data_handle(DummyData);
    if (bias_data != nullptr) {
      context_.bias_mem->set_data_handle(DummyData);
    }
    if (bn_scale_data != nullptr) {
      context_.bn_scale_mem->set_data_handle(DummyData);
      context_.bn_mean_mem->set_data_handle(DummyData);
      context_.bn_rsqrt_mem->set_data_handle(DummyData);
      context_.bn_offset_mem->set_data_handle(DummyData);
    }
    context_.dst_mem->set_data_handle(DummyData);
    if (sp_data) {
      context_.sp_mem->set_data_handle(DummyData);
    }

    return;
  }

  std::shared_ptr<DeconvFwdPd> GetPrimitiveDesc() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for deconv fwd.
  struct DeconvFwdContext {
    // oneDNN memory.
    std::shared_ptr<dnnl::memory> src_mem;
    std::shared_ptr<dnnl::memory> filter_mem;
    std::shared_ptr<dnnl::memory> bias_mem;
    std::shared_ptr<dnnl::memory> dst_mem;
    std::shared_ptr<dnnl::memory> sp_mem;

    // FusedBatchNorm related memory
    std::shared_ptr<dnnl::memory> bn_scale_mem;
    std::shared_ptr<dnnl::memory> bn_mean_mem;
    std::shared_ptr<dnnl::memory> bn_rsqrt_mem;
    std::shared_ptr<dnnl::memory> bn_offset_mem;

    // Deconv forward primitive descriptor and descriptor.
    std::shared_ptr<DeconvFwdPd> fwd_pd;
#ifdef ENABLE_ONEDNN_V2
    std::shared_ptr<dnnl::deconvolution_forward::desc> fwd_desc;
#endif  // !ENABLE_ONEDNN_V2

    // Deconv fwd primitive.
    std::shared_ptr<dnnl::primitive> deconv_fwd;

    // Memory descriptors.
    std::shared_ptr<memory::desc> src_md;
    std::shared_ptr<memory::desc> filter_md;
    std::shared_ptr<memory::desc> bias_md;
    std::shared_ptr<memory::desc> dst_md;

    // TODO(intel-tf): Only need one? FusedBatchNorm related.
    std::shared_ptr<dnnl::memory::desc> bn_scale_md;
    std::shared_ptr<dnnl::memory::desc> bn_mean_md;
    std::shared_ptr<dnnl::memory::desc> bn_rsqrt_md;
    std::shared_ptr<dnnl::memory::desc> bn_offset_md;

    // oneDNN pipeline for executing primitives.
    std::vector<dnnl::primitive> fwd_primitives;
    std::vector<std::unordered_map<int, memory>> fwd_primitives_args;

    DeconvFwdContext()
        : src_mem(nullptr),
          filter_mem(nullptr),
          bias_mem(nullptr),
          dst_mem(nullptr),
          sp_mem(nullptr),
          bn_scale_mem(nullptr),
          bn_mean_mem(nullptr),
          bn_rsqrt_mem(nullptr),
          bn_offset_mem(nullptr),
          fwd_pd(nullptr),
#ifdef ENABLE_ONEDNN_V2
          fwd_desc(nullptr),
#endif  // !ENABLE_ONEDNN_V2
          deconv_fwd(nullptr),
          src_md(nullptr),
          filter_md(nullptr),
          bias_md(nullptr),
          dst_md(nullptr),
          bn_scale_md(nullptr),
          bn_mean_md(nullptr),
          bn_rsqrt_md(nullptr),
          bn_offset_md(nullptr) {
    }
  };

  void Setup(const MklDeconvFwdParams& deconvFwdParams) {
    context_.src_md.reset(new memory::desc({deconvFwdParams.src_dims},
                                           MklDnnType<Tinput>(),
                                           deconvFwdParams.fmt_tag));
    context_.dst_md.reset(new memory::desc({deconvFwdParams.dst_dims},
                                           MklDnnType<Toutput>(),
                                           deconvFwdParams.fmt_tag));
    context_.filter_md.reset(new memory::desc({deconvFwdParams.filter_dims},
                                              MklDnnType<Tfilter>(),
                                              memory::format_tag::any));
    if (!deconvFwdParams.bias_dims.empty()) {
      context_.bias_md.reset(new memory::desc({deconvFwdParams.bias_dims},
                                              MklDnnType<Tbias>(),
                                              memory::format_tag::any));
    }

    // Create descriptors for forward deconv op.
#ifdef ENABLE_ONEDNN_V2
    if (!deconvFwdParams.bias_dims.empty()) {
      context_.fwd_desc.reset(new deconvolution_forward::desc(
          prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
          *context_.src_md, *context_.filter_md, *context_.bias_md,
          *context_.dst_md, deconvFwdParams.strides, deconvFwdParams.dilations,
          deconvFwdParams.padding_left, deconvFwdParams.padding_right));
    } else {
      context_.fwd_desc.reset(new deconvolution_forward::desc(
          prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
          *context_.src_md, *context_.filter_md, *context_.dst_md,
          deconvFwdParams.strides, deconvFwdParams.dilations,
          deconvFwdParams.padding_left, deconvFwdParams.padding_right));
    }
#endif  // !ENABLE_ONEDNN_V2

    if (!deconvFwdParams.fuse_bn_dims.empty()) {
      const memory::format_tag fused_bn_arg_fmt = deconvFwdParams.fmt_tag;

      context_.bn_scale_md.reset(
          new memory::desc({deconvFwdParams.fuse_bn_dims}, MklDnnType<float>(),
                           fused_bn_arg_fmt));
      context_.bn_mean_md.reset(new memory::desc({deconvFwdParams.fuse_bn_dims},
                                                 MklDnnType<float>(),
                                                 fused_bn_arg_fmt));
      context_.bn_rsqrt_md.reset(
          new memory::desc({deconvFwdParams.fuse_bn_dims}, MklDnnType<float>(),
                           fused_bn_arg_fmt));
      context_.bn_offset_md.reset(
          new memory::desc({deconvFwdParams.fuse_bn_dims}, MklDnnType<float>(),
                           fused_bn_arg_fmt));
    }

    // Check if there is any fusions as post-ops
    auto const& post_op_params = deconvFwdParams.post_op_params;
    dnnl::primitive_attr post_ops_attr;
    dnnl::post_ops post_ops;
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (!post_op_params.empty()) {
      for (auto const& post_op_param : post_op_params) {
        if (post_op_param.name == "activation") {
          DCHECK_EQ(post_op_param.param.size(), 3);
          float op_scale = post_op_param.param[0];
          float op_alpha = post_op_param.param[1];
          float op_beta = post_op_param.param[2];
          // TODO(intel-tf): Enable this for int8 when using oneDNN v3.x
          post_ops.APPEND_ELTWISE(op_scale, post_op_param.alg, op_alpha,
                                  op_beta);
        } else if (post_op_param.name == "output_scale") {
#ifdef ENABLE_ONEDNN_V2
          if (post_op_param.param.size() == 1) {
            post_ops_attr.set_output_scales(0, post_op_param.param);
          } else {
            post_ops_attr.set_output_scales(2, post_op_param.param);
          }
#else
          // TODO(intel-tf): Enable this for int8 when using oneDNN v3.x
          assert(post_op_param.param.size() == 1);
          post_ops_attr.set_scales_mask(DNNL_ARG_SRC, 0);
          post_ops_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
          post_ops_attr.set_scales_mask(DNNL_ARG_DST, 0);
#endif  // !ENABLE_ONEDNN_V2
        } else if (post_op_param.name == "fuse_bn") {
          post_ops.append_binary(dnnl::algorithm::binary_sub,
                                 *context_.bn_mean_md);
          post_ops.append_binary(dnnl::algorithm::binary_mul,
                                 *context_.bn_rsqrt_md);
          post_ops.append_binary(dnnl::algorithm::binary_mul,
                                 *context_.bn_scale_md);
          post_ops.append_binary(dnnl::algorithm::binary_add,
                                 *context_.bn_offset_md);
        } else {
          DCHECK((post_op_param.name == "activation") ||
                 (post_op_param.name == "output_scale") ||
                 (post_op_param.name == "fuse_bn"));
        }
      }
      post_ops_attr.set_post_ops(post_ops);
    }

#ifdef ENABLE_ONEDNN_V2
    context_.fwd_pd.reset(
        new DeconvFwdPd(*context_.fwd_desc, post_ops_attr, cpu_engine_));
#else
    if (!deconvFwdParams.bias_dims.empty()) {
      context_.fwd_pd.reset(new DeconvFwdPd(
          cpu_engine_, prop_kind::forward_inference,
          dnnl::algorithm::deconvolution_direct, *context_.src_md,
          *context_.filter_md, *context_.bias_md, *context_.dst_md,
          deconvFwdParams.strides, deconvFwdParams.dilations,
          deconvFwdParams.padding_left, deconvFwdParams.padding_right,
          post_ops_attr));
    } else {
      context_.fwd_pd.reset(new DeconvFwdPd(
          cpu_engine_, prop_kind::forward_inference,
          dnnl::algorithm::deconvolution_direct, *context_.src_md,
          *context_.filter_md, *context_.dst_md, deconvFwdParams.strides,
          deconvFwdParams.dilations, deconvFwdParams.padding_left,
          deconvFwdParams.padding_right, post_ops_attr));
    }
#endif  // !ENABLE_ONEDNN_V2

    // Create memory using dummy data.
    context_.src_mem.reset(
        new memory(context_.fwd_pd.get()->src_desc(), cpu_engine_, DummyData));
    context_.filter_mem.reset(new memory(context_.fwd_pd.get()->weights_desc(),
                                         cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd.get()->dst_desc(), cpu_engine_, DummyData));

    // Create forward deconv primitive and add it to the net.
    context_.deconv_fwd.reset(new deconvolution_forward(*context_.fwd_pd));
    auto scratchpad_md = context_.fwd_pd->scratchpad_desc();
    context_.sp_mem.reset(
        new dnnl::memory(scratchpad_md, cpu_engine_, DummyData));

    if (!deconvFwdParams.bias_dims.empty()) {
      context_.bias_mem.reset(new memory({{deconvFwdParams.bias_dims},
                                          MklDnnType<Tbias>(),
                                          memory::format_tag::x},
                                         cpu_engine_, DummyData));
      context_.fwd_primitives_args.push_back(
          {{DNNL_ARG_SRC, *context_.src_mem},
           {DNNL_ARG_WEIGHTS, *context_.filter_mem},
           {DNNL_ARG_BIAS, *context_.bias_mem},
           {DNNL_ARG_SCRATCHPAD, *context_.sp_mem},
           {DNNL_ARG_DST, *context_.dst_mem}});
    } else if (!deconvFwdParams.fuse_bn_dims.empty()) {
      context_.bn_scale_mem.reset(
          new memory(*context_.bn_scale_md, cpu_engine_, DummyData));
      context_.bn_mean_mem.reset(
          new memory(*context_.bn_mean_md, cpu_engine_, DummyData));
      context_.bn_offset_mem.reset(
          new memory(*context_.bn_offset_md, cpu_engine_, DummyData));
      context_.bn_rsqrt_mem.reset(
          new memory(*context_.bn_rsqrt_md, cpu_engine_, DummyData));

      context_.fwd_primitives_args.push_back(
          {{DNNL_ARG_SRC, *context_.src_mem},
           {DNNL_ARG_WEIGHTS, *context_.filter_mem},
           {DNNL_ARG_DST, *context_.dst_mem},
           {DNNL_ARG_SCRATCHPAD, *context_.sp_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
            *context_.bn_mean_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
            *context_.bn_rsqrt_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
            *context_.bn_scale_mem},
           {DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | DNNL_ARG_SRC_1,
            *context_.bn_offset_mem}});
    } else {
      context_.fwd_primitives_args.push_back(
          {{DNNL_ARG_SRC, *context_.src_mem},
           {DNNL_ARG_WEIGHTS, *context_.filter_mem},
           {DNNL_ARG_SCRATCHPAD, *context_.sp_mem},
           {DNNL_ARG_DST, *context_.dst_mem}});
    }
    context_.fwd_primitives.push_back(*context_.deconv_fwd);
  }
  struct DeconvFwdContext context_;
};

template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput>
class MklDeconvFwdPrimitiveFactory : public MklPrimitiveFactory<float> {
 private:
  MklDeconvFwdPrimitiveFactory() {}
  ~MklDeconvFwdPrimitiveFactory() {}

 public:
  static MklDeconvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>* Get(
      const MklDeconvFwdParams& deconvFwdParams) {
    MklDeconvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>* deconv_fwd =
        nullptr;

    // Look into the pool for reusable primitive.
    deconv_fwd =
        dynamic_cast<MklDeconvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>*>(
            MklDeconvFwdPrimitiveFactory<Tinput, Tfilter, Tbias,
                                         Toutput>::GetInstance()
                .GetDeconvFwd(deconvFwdParams));
    if (deconv_fwd == nullptr) {
      deconv_fwd = new MklDeconvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>(
          deconvFwdParams);
      MklDeconvFwdPrimitiveFactory<Tinput, Tfilter, Tbias,
                                   Toutput>::GetInstance()
          .SetDeconvFwd(deconvFwdParams, deconv_fwd);
    }
    return deconv_fwd;
  }

 private:
  static MklDeconvFwdPrimitiveFactory& GetInstance() {
    static MklDeconvFwdPrimitiveFactory instance_;
    return instance_;
  }

  static string CreateKey(const MklDeconvFwdParams& deconvFwdParams) {
    string prefix = "deconv_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(deconvFwdParams.src_dims);
    key_creator.AddAsKey(deconvFwdParams.filter_dims);
    key_creator.AddAsKey(deconvFwdParams.bias_dims);
    key_creator.AddAsKey(deconvFwdParams.dst_dims);
    key_creator.AddAsKey(deconvFwdParams.strides);
    key_creator.AddAsKey(deconvFwdParams.dilations);
    key_creator.AddAsKey(deconvFwdParams.padding_left);
    key_creator.AddAsKey(deconvFwdParams.padding_right);
    key_creator.AddAsKey(deconvFwdParams.fmt_tag);
    key_creator.AddAsKey(deconvFwdParams.dtypes);

    // Generate keys for post-ops
    for (auto const& post_op_param : deconvFwdParams.post_op_params) {
      key_creator.AddAsKey(post_op_param.name);
      if (post_op_param.name == "activation") {
        key_creator.AddAsKey(post_op_param.alg);
        DCHECK_EQ(post_op_param.param.size(), 3);
        for (auto& param : post_op_param.param) {
          key_creator.AddAsKey(param);
        }
      } else if (post_op_param.name == "output_scale") {
        key_creator.AddAsKey(post_op_param.partial_key);
      } else if (post_op_param.name == "fuse_bn") {
        key_creator.AddAsKey(post_op_param.name);
        key_creator.AddAsKey(deconvFwdParams.fuse_bn_dims);
      } else {
        return string("not_a_key");
      }
    }
    return key_creator.GetKey();
  }

  MklPrimitive* GetDeconvFwd(const MklDeconvFwdParams& deconvFwdParams) {
    string key = CreateKey(deconvFwdParams);
    return this->GetOp(key);
  }

  void SetDeconvFwd(const MklDeconvFwdParams& deconvFwdParams,
                    MklPrimitive* op) {
    string key = CreateKey(deconvFwdParams);
    this->SetOp(key, op);
  }
};

template <typename Device, class Tinput, class Tfilter, class Tbias,
          class Toutput, bool is_depthwise>
class MklDeconvOp : public OpKernel {
 public:
  explicit MklDeconvOp(OpKernelConstruction* context) : OpKernel(context) {
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    if (context->HasAttr("explicit_paddings")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("explicit_paddings", &padding_list_));
    }
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    is_filter_const_ = false;
    if (AreWeightsFrozen()) {
      is_filter_const_ = true;
    } else if (context->HasAttr("is_filter_const")) {
      OP_REQUIRES_OK(context,
                     context->GetAttr("is_filter_const", &is_filter_const_));
    }

    if (strides_.size() == 4) {
      OP_REQUIRES(context, dilations_.size() == 4,
                  errors::InvalidArgument("Sliding window dilations field must "
                                          "specify 4 dimensions"));
      int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
      int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
      int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
      int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
      OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                  errors::InvalidArgument(
                      "Current implementation does not yet support "
                      "dilations in the batch and depth dimensions."));
      OP_REQUIRES(
          context, dilation_h > 0 && dilation_w > 0,
          errors::InvalidArgument("Dilated rates should be larger than 0."));
      OP_REQUIRES(context, !is_depthwise,
                  errors::InvalidArgument(
                      "DepthwiseDeconv is currently not supported."));
    }
  }

  ~MklDeconvOp() {}

  void Compute(OpKernelContext* context) {
    try {
      // Input tensors.
      const Tensor& input_tensor = context->input(kInputIndex);
      const Tensor& filter_tensor = context->input(kFilterIndex);
      const Tensor& src_tensor = context->input(kSrcIndex);

      auto input_shape = input_tensor.shape();
      auto filter_shape = filter_tensor.shape();
      auto src_shape = src_tensor.shape();
      auto filter_dims = TFShapeToMklDnnDims(filter_shape);

      memory::dims padding_left, padding_right, dilations, strides;
      TensorShape input_tf_shape = MakeInputTfShape(context, input_tensor);

      bool pad_attr_enabled = false;
      for (auto const& padding_val : padding_list_) {
        if (padding_val) {
          pad_attr_enabled = true;
          break;
        }
      }

      if (pad_attr_enabled) {
        PadWithDeconvFusion(context, padding_left, padding_right);
      }

      memory::dims src_dims, fwd_src_dims, fwd_filter_dims;

      // Get forward deconv sizes.
      DnnDeconvUtil deconv_util(context, strides_, padding_, data_format_,
                                dilations_);
      deconv_util.GetDeconvFwdSizes(input_tf_shape, filter_shape, &fwd_src_dims,
                                    &fwd_filter_dims, &strides, &dilations,
                                    &padding_left, &padding_right,
                                    pad_attr_enabled);
      if (!context->status().ok()) return;

      bool is_deconv2d = (this->strides_.size() == 4);

      deconv_util.GetInputSize(src_shape, &src_dims);
      if (!context->status().ok()) return;

      auto input_dims = TFShapeToMklDnnDims(input_tf_shape);
      TensorShape dst_shape(input_dims);

      Tensor* dst_tensor = nullptr;
      auto output_dst_dims = TFShapeToMklDnnDims(dst_shape);
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, dst_shape, &dst_tensor));

      if (input_shape.num_elements() == 0 || filter_shape.num_elements() == 0 ||
          src_shape.num_elements() == 0) {
        // If output tensor has more than 0 elements, we need to 0 them out.
        auto dst_data = dst_tensor->flat<Toutput>().data();
        for (size_t i = 0; i < input_shape.num_elements(); ++i) {
          dst_data[i] = static_cast<Toutput>(0);
        }
        return;
      }

      // Create memory for user data.
      // Describe how the inputs and outputs of Deconvolution look like.
      // Also specify buffers containing actual input and output data.
      auto fmt_tag = is_deconv2d
                         ? TFDataFormatToOneDnn2DDataFormat(data_format_)
                         : TFDataFormatToOneDnn3DDataFormat(data_format_);

      memory::dims bias_dims = {};
      if (fuse_biasadd_) {
        const Tensor& bias_tensor = context->input(kBiasIndex);
        bias_dims = {static_cast<int>(bias_tensor.dim_size(0))};
      }
      memory::dims fuse_bn_dims = {};
      TensorShape fuse_bn_shape;
      if (fuse_bn_) {
        // Inputs to FusedBatchNorm have same 1D shape
        fuse_bn_shape = MklGetInput(context, kInputIndex_BN_Mean).shape();
        OP_REQUIRES(context, fuse_bn_shape.dims() == 1,
                    errors::InvalidArgument("FusedBatchNorm must be 1D, not: ",
                                            fuse_bn_shape.DebugString()));

        // Note - MKL-DNN expects {1, C, 1, 1} for binary post-op even for NHWC
        fuse_bn_dims = {1, fuse_bn_shape.dim_size(0), 1, 1};
      }

      // oneDNN dilations start from 0.
      for (int i = 0; i < dilations.size(); ++i) --dilations[i];

      MklDeconvFwdParams deconvFwdParams(
          fwd_src_dims, fwd_filter_dims, fuse_biasadd_ ? bias_dims : NONE_DIMS,
          src_dims, strides, dilations, padding_left, padding_right,
          fuse_bn_dims, fmt_tag);

      this->ExtendDeconvFwdParams(context, deconvFwdParams);

      MklDeconvFwdPrimitive<Tinput, Tfilter, Tbias, Toutput>* deconv_fwd =
          MklDeconvFwdPrimitiveFactory<Tinput, Tfilter, Tbias, Toutput>::Get(
              deconvFwdParams);

      std::shared_ptr<DeconvFwdPd> deconv_fwd_pd =
          deconv_fwd->GetPrimitiveDesc();

      auto cpu_engine = deconv_fwd->GetEngine();

      auto src_md = memory::desc(src_dims, MklDnnType<Tinput>(), fmt_tag);
      MklDnnData<Tinput> src(&cpu_engine);
      Tinput* src_data = nullptr;
      if (src_md != deconv_fwd_pd.get()->src_desc()) {
        src.SetUsrMem(src_md, &src_tensor);
        src.CheckReorderToOpMem(deconv_fwd_pd->src_desc(), cpu_engine, context);
        src_data = static_cast<Tinput*>(src.GetOpMem().get_data_handle());
      } else {
        src_data = static_cast<Tinput*>(
            const_cast<Tinput*>(src_tensor.flat<Tinput>().data()));
      }

      memory::dims filter_strides;
      deconv_util.GetFilterStrides(fwd_filter_dims, &filter_strides);

      auto fwd_filter_md =
          memory::desc(fwd_filter_dims, MklDnnType<Tfilter>(), filter_strides);

      MklDnnData<Tfilter> filter(&cpu_engine);
      Tfilter* filter_data = nullptr;
      if (fwd_filter_md != deconv_fwd_pd.get()->weights_desc()) {
        bool is_filter_cached = false;
        if (is_filter_const_) {
          if (IsFilterCacheEmpty(context)) {
            // Cache filter if it is not already cached.
            CacheFilter(context, deconv_fwd_pd, filter_data, filter_tensor,
                        filter, fwd_filter_md);
          }
          filter_data = GetCachedFilter(context, deconv_fwd_pd->weights_desc());
          is_filter_cached = (filter_data != nullptr);
        }
        if (!is_filter_cached) {
          filter.SetUsrMem(fwd_filter_md, &filter_tensor);
          filter.CheckReorderToOpMem(deconv_fwd_pd.get()->weights_desc(),
                                     cpu_engine, context);
          filter_data =
              static_cast<Tfilter*>(filter.GetOpMem().get_data_handle());
        }
      } else {
        filter_data = static_cast<Tfilter*>(
            const_cast<Tfilter*>(filter_tensor.flat<Tfilter>().data()));
      }

      Toutput* dst_data = static_cast<Toutput*>(
          const_cast<Toutput*>(dst_tensor->flat<Toutput>().data()));

      UserScratchPad<unsigned char> scratch_pad;
      scratch_pad.AllocateSPTensor(deconv_fwd, context);

      // Execute deconvolution
      std::shared_ptr<stream> cpu_stream;
      MklDnnThreadPool eigen_tp(context);
      cpu_stream.reset(CreateStream(&eigen_tp, cpu_engine));

      if (fuse_biasadd_) {
        const Tensor& bias_tensor = context->input(kBiasIndex);
        Tbias* bias_data =
            this->GetBiasHandle(context, deconv_fwd_pd, bias_tensor);
        deconv_fwd->Execute(src_data, filter_data, bias_data, dst_data, nullptr,
                            nullptr, nullptr, nullptr, cpu_stream,
                            scratch_pad.Get());
      } else if (fuse_bn_) {
        const Tensor& bn_scale_tensor =
            MklGetInput(context, kInputIndex_BN_Scale);
        float* bn_scale_data = static_cast<float*>(
            const_cast<float*>(bn_scale_tensor.flat<float>().data()));
        const Tensor& bn_mean_tensor =
            MklGetInput(context, kInputIndex_BN_Mean);
        float* bn_mean_data = static_cast<float*>(
            const_cast<float*>(bn_mean_tensor.flat<float>().data()));
        const Tensor& bn_offset_tensor =
            MklGetInput(context, kInputIndex_BN_Offset);
        float* bn_offset_data = static_cast<float*>(
            const_cast<float*>(bn_offset_tensor.flat<float>().data()));

        Tensor bn_rsqrt_tensor;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataTypeToEnum<float>::v(),
                                              fuse_bn_shape, &bn_rsqrt_tensor));
        float* bn_rsqrt_data = static_cast<float*>(
            const_cast<float*>(bn_rsqrt_tensor.flat<float>().data()));
        this->ComputeBNScale(context, epsilon_, kInputIndex_BN_Variance,
                             bn_rsqrt_data);
        deconv_fwd->Execute(src_data, filter_data, nullptr, dst_data,
                            bn_scale_data, bn_mean_data, bn_offset_data,
                            bn_rsqrt_data, cpu_stream, scratch_pad.Get());
      } else {
        deconv_fwd->Execute(src_data, filter_data, nullptr, dst_data, nullptr,
                            nullptr, nullptr, nullptr, cpu_stream,
                            scratch_pad.Get());
      }
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    } catch (...) {
    }
  }

 protected:
  void PadWithDeconvFusion(OpKernelContext* context, memory::dims& padding_left,
                           memory::dims& padding_right) {
    int64* paddings = padding_list_.data();

    // If the data format is NHWC, indices 0, 1, 6 and 7 of paddings(_tf)
    // will be zero.
    // Example:
    // paddings_tf = [ [0, 0] [1, 2] [3, 4] [0, 0] ],
    // flat method = row-major, then:
    // paddings = {0, 0, 1, 2, 3, 4, 0, 0}.
    // Hence, the values are: top = 1, bottom = 2, left = 3, right = 4.
    //
    // Similarly, if the data format is NCHW, indices 0, 1, 2 and 3 of
    // paddings(_tf) will be zero.
    // i.e. for the above example, paddings = {0, 0, 0, 0, 1, 2, 3, 4}.
    int64 pad_top = 0, pad_left = 0;
    int64 pad_bottom = 0, pad_right = 0;
    string data_format = ToString(data_format_);
    if (data_format == "NHWC") {
      pad_top = paddings[2];
      pad_bottom = paddings[3];
      pad_left = paddings[4];
      pad_right = paddings[5];
    } else if (data_format == "NCHW") {
      pad_top = paddings[4];
      pad_bottom = paddings[5];
      pad_left = paddings[6];
      pad_right = paddings[7];
    }
    // Create padding arrays for oneDNN deconvolution.
    // oneDNN uses asymmetric padding.
    padding_left = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
    padding_right = {static_cast<int>(pad_bottom), static_cast<int>(pad_right)};
  }

  virtual Tbias* GetBiasHandle(OpKernelContext* context,
                               std::shared_ptr<DeconvFwdPd>& deconv_fwd_pd,
                               const Tensor& bias_tensor) {
    if (fuse_biasadd_) {
      return static_cast<Tbias*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
    }
    return nullptr;
  }

  TensorShape MakeInputTfShape(OpKernelContext* context,
                               const Tensor& input_tensor) {
    TensorShape input_tf_shape;
    CHECK_EQ(TensorShapeUtils::IsVector(input_tensor.shape()), true);
    // Deconv[2D|3D]BackpropInputV2 supports both DT_INT32 and DT_INT64
    // output_shape tensor::MakeShape is able to handle both DT_INT32 and
    // DT_INT64 for input_tensor.
    TF_CHECK_OK(tensor::MakeShape(input_tensor, &input_tf_shape));
    return input_tf_shape;
  }

 private:
  const int kInputIndex = 0, kFilterIndex = 1, kSrcIndex = 2, kBiasIndex = 3;
  // Input indices for FusedBatchNorm
  const int kInputIndex_BN_Scale = 3, kInputIndex_BN_Offset = 4;
  const int kInputIndex_BN_Mean = 5, kInputIndex_BN_Variance = 6;
  bool is_filter_const_;
  mutex mu_;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  Tensor cached_filter_md_ TF_GUARDED_BY(mu_);

  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  std::vector<int64> padding_list_;
  Padding padding_;
  TensorFormat data_format_;

  // Allocate tensors for cached filter data and cached filter memory
  // descriptor (data format)
  void AllocateTensor(OpKernelContext* context,
                      const DeconvFwdPd& deconv_prim_desc,
                      Tensor** filter_tensor) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    DCHECK(filter_tensor);
    TensorShape filter_tf_shape;
    filter_tf_shape.AddDim(
        (deconv_prim_desc.weights_desc().get_size() / sizeof(Tfilter)));
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<Tfilter>::value,
                                        filter_tf_shape, &cached_filter_data_));

    *filter_tensor = &cached_filter_data_;

    // There is no tensor format in DNNL 1.x. So we cache the complete filter
    // descriptor as flat byte array.
    TensorShape cached_filter_md_shape;
    memory::desc weights_desc = deconv_prim_desc.weights_desc();
    // We don't use .get_size() method of memory::desc since it returns size
    // required to store primitive's input memory. It is much more than size of
    // memory::desc itself.
    cached_filter_md_shape.AddDim(sizeof(weights_desc) / sizeof(uint8));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_UINT8, cached_filter_md_shape,
                                          &cached_filter_md_));
    *reinterpret_cast<memory::desc*>(cached_filter_md_.flat<uint8>().data()) =
        weights_desc;
  }

  // TF_LOCKS_EXCLUDED annotation ensures that the lock (mu_) cannot
  // be acquired before entering the function, since it is acquired
  // inside the function.
  inline bool IsFilterCacheEmpty(OpKernelContext* context)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& cached_filter_data_tensor = cached_filter_data_;
    return (cached_filter_data_tensor.NumElements() == 0);
  }

  // Cache the converted filter in a tensor.
  // Only one thread can execute this method at any given time.
  void CacheFilter(OpKernelContext* context,
                   const std::shared_ptr<DeconvFwdPd>& deconv_fwd_pd,
                   Tfilter* filter_data, const Tensor& filter_tensor,
                   MklDnnData<Tfilter>& filter, const memory::desc& filter_md)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    const Tensor& cached_filter_data_tensor = cached_filter_data_;

    // If filter is already cached, there's nothing to do.
    if (cached_filter_data_tensor.NumElements() > 0) {
      return;
    }

    // Otherwise, cache filter
    filter.SetUsrMem(filter_md, &filter_tensor);
    filter.CheckReorderToOpMem(deconv_fwd_pd.get()->weights_desc(),
                               this->cpu_engine_, context);
    filter_data = static_cast<Tfilter*>(filter.GetOpMem().get_data_handle());

    Tensor* filter_tensor_ptr = nullptr;
    AllocateTensor(context, *deconv_fwd_pd, &filter_tensor_ptr);
    void* cached_filter_data = filter.GetTensorBuffer(filter_tensor_ptr);
    size_t cached_filter_data_size = filter.GetOpMem().get_desc().get_size();
    memcpy(cached_filter_data, filter_data, cached_filter_data_size);
  }

  Tfilter* GetCachedFilter(OpKernelContext* context,
                           const memory::desc& filter_md)
      TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock lock(mu_);
    const Tensor& cached_filter_data = cached_filter_data_;
    const Tensor& cached_filter_md = cached_filter_md_;

    // Check if the memory descriptor of the cached weights is the same as
    // filter_md. If so, we can use the cached weights; otherwise
    // return nullptr.
    if (filter_md == *static_cast<memory::desc*>(cached_filter_md.data())) {
      return static_cast<Tfilter*>(
          const_cast<Tfilter*>(cached_filter_data.flat<Tfilter>().data()));
    }
    return nullptr;
  }

 protected:
  engine cpu_engine_ = engine(engine::kind::cpu, 0);
  bool fuse_biasadd_ = false;
  bool fuse_bn_ = false;
  float epsilon_ = 0.0001;
  bool fuse_activation_ = false;
  void set_fuse_bn(bool fuse_bn, float epsilon) {
    fuse_bn_ = fuse_bn;
    epsilon_ = epsilon;
  }
  float alpha_or_upbound_ = 0.0;
  dnnl::algorithm activation_alg_ = dnnl::algorithm::undef;
  void set_fuse_activation(bool fuse_activation, dnnl::algorithm activation_alg,
                           float alpha_or_upbound = 0.0) {
    fuse_activation_ = fuse_activation;
    activation_alg_ = activation_alg;
    // This variable is used for alpha in leakyrelu or upper bound in relu6
    // depending on the context
    alpha_or_upbound_ = alpha_or_upbound;
  }
  virtual void ExtendDeconvFwdParams(OpKernelContext* context,
                                     MklDeconvFwdParams& params) {
    // Create a string from data types of input, filter, bias, and output.
    params.dtypes.append(typeid(Tinput).name());
    params.dtypes.append(typeid(Tfilter).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());

    bool is_quantized_input = std::is_same<Tinput, quint8>::value ||
                              std::is_same<Tinput, qint8>::value;
    if (!is_quantized_input) {
      // Add fusions as post ops
      // NOTE - fuse_bn post_op entry must be before fuse_activation
      if (fuse_bn_) {
        params.post_op_params.push_back(
            {"fuse_bn", dnnl::algorithm::undef, {1.0}, ""});
      }
      if (fuse_activation_) {
        params.post_op_params.push_back(
            {"activation", activation_alg_, {1.0, alpha_or_upbound_, 0.0}, ""});
      }
    }
  }
  virtual void ComputeBNScale(OpKernelContext* context, float epsilon,
                              int bn_variance_index, float* scale_buf_ptr) {
    OP_REQUIRES(
        context, false,
        errors::Unimplemented("Compute BN scale not expected in base class"));
    return;
  }
};

template <typename Device, class Tinput, class Tfilter, class Tbias,
          class Toutput, bool is_depthwise>
class MklFusedDeconvOp : public MklDeconvOp<Device, Tinput, /*Tfilter*/ Tfilter,
                                            Tbias, Toutput, is_depthwise> {
 public:
  virtual ~MklFusedDeconvOp() {}

  explicit MklFusedDeconvOp(OpKernelConstruction* context)
      : MklDeconvOp<Device, Tinput, /*Tfilter*/ Tfilter, Tbias, Toutput,
                    is_depthwise>(context) {
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));

    int num_args;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    OP_REQUIRES(context, !fused_ops.empty(),
                errors::InvalidArgument(
                    "FusedDeconv2D must have at least one fused op."));

    if (fused_ops == std::vector<string>{"BiasAdd"}) {
      this->fuse_biasadd_ = true;
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "FusedDeconv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Relu"}) {
      this->fuse_biasadd_ = true;
      this->set_fuse_activation(true, dnnl::algorithm::eltwise_relu);
      OP_REQUIRES(context, num_args == 1,
                  errors::InvalidArgument(
                      "FusedDeconv2D must have one extra argument: bias."));
    } else if (fused_ops == std::vector<string>{"FusedBatchNorm"}) {
      float epsilon;
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument(
              "FusedDeconv2D with batchnorm must have 4 extra argument"));

      std::vector<DataType> TArgs;
      OP_REQUIRES_OK(context, context->GetAttr("TArgs", &TArgs));
      OP_REQUIRES(
          context, TArgs.size() == 4,
          errors::InvalidArgument("FusedDeconv2D TArgs size expected to be 4"));
      for (int i = 0; i < 4; ++i) {
        OP_REQUIRES(context, TArgs.at(i) == DT_FLOAT,
                    errors::InvalidArgument("TArgs must be DT_FLOAT"));
      }
      this->set_fuse_bn(true, epsilon);
    } else if (fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"}) {
      float epsilon;
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
      OP_REQUIRES(
          context, num_args == 4,
          errors::InvalidArgument(
              "FusedDeconv2D with batchnorm must have 4 extra argument"));

      std::vector<DataType> TArgs;
      OP_REQUIRES_OK(context, context->GetAttr("TArgs", &TArgs));
      OP_REQUIRES(
          context, TArgs.size() == 4,
          errors::InvalidArgument("FusedDeconv2D TArgs size expected to be 4"));
      for (int i = 0; i < 4; ++i) {
        OP_REQUIRES(context, TArgs.at(i) == DT_FLOAT,
                    errors::InvalidArgument("TArgs must be DT_FLOAT"));
      }
      this->set_fuse_bn(true, epsilon);
      this->set_fuse_activation(true, dnnl::algorithm::eltwise_relu);
    } else {
      OP_REQUIRES(context, false,
                  errors::Unimplemented("Fusion is not implemented: [",
                                        absl::StrJoin(fused_ops, ","), "]"));
    }
  }

  void ComputeBNScale(OpKernelContext* context, float epsilon,
                      int bn_variance_index, float* scale_buf_ptr) override {
    const Tensor& bn_var_tensor = MklGetInput(context, bn_variance_index);

    Eigen::Tensor<float, 1, Eigen::RowMajor> bn_rsqrt =
        (bn_var_tensor.flat<float>() + static_cast<float>(epsilon)).rsqrt();
    float* bn_rsqrt_data = bn_rsqrt.data();
    size_t num_elem = bn_var_tensor.shape().dim_size(0);
    for (size_t i = 0; i < num_elem; i++) {
      scale_buf_ptr[i] = bn_rsqrt_data[i];
    }
    return;
  }
};

#define REGISTER_MKL_CPU_KERNELS(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeConv2DBackpropInput")               \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklDeconvOp<CPUDevice, T, T, T, T, false>);         \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeConv3DBackpropInputV2")             \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklDeconvOp<CPUDevice, T, T, T, T, false>);

TF_CALL_float(REGISTER_MKL_CPU_KERNELS);
TF_CALL_bfloat16(REGISTER_MKL_CPU_KERNELS);

#undef REGISTER_MKL_CPU_KERNELS

}  // namespace tensorflow
#endif  // INTEL_MKL
