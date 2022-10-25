
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/kernels/mkl/mkl_rnn_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

namespace gru_scope {

typedef Eigen::ThreadPoolDevice CPUDevice;
/*=================================================================
  GRU  LBR Fwd Primitve for execution
==================================================================*/

template<typename T, typename Prim>
typename std::enable_if<std::is_same<Prim, dnnl::augru_forward>::value,
	                typename dnnl::augru_forward::desc*>::type
CreatePrimDesc(MklGRUParams<T> &params, dnnl::engine engine,
               dnnl::prop_kind prop_k, dnnl::rnn_direction direction) {
  params.show("AUGRUFwd");
  return new dnnl::augru_forward::desc(prop_k, direction, *params.src_layer.desc,
                          *params.src_iter.desc, *params.attention.desc,
                          *params.weights_layer.desc, *params.weights_iter.desc,
                          *params.bias.desc, *params.dst_layer.desc,
                          *params.dst_iter.desc);
}
template<typename T, typename Prim>
typename std::enable_if<std::is_same<Prim, dnnl::gru_forward>::value,
	                typename dnnl::gru_forward::desc*>::type
CreatePrimDesc(MklGRUParams<T> &params, dnnl::engine engine,
               dnnl::prop_kind prop_k, dnnl::rnn_direction direction) {
  params.show("GRUFwd");
  return new dnnl::gru_forward::desc(prop_k, direction, *params.src_layer.desc,
                          *params.src_iter.desc, *params.weights_layer.desc,
                          *params.weights_iter.desc, *params.bias.desc,
                          *params.dst_layer.desc, *params.dst_iter.desc);


}
/*=================================================================
  The class GRUPrimitive uses CreatePrimDesc to create specific
  primitives
==================================================================*/
template <typename T, typename Prim>
class GRUPrimitive {
 public:
  using Desc = typename Prim::desc;
  using primitiveDesc = typename Prim::primitive_desc;

  GRUPrimitive(dnnl::prop_kind pkind, dnnl::rnn_direction dir)
      : gru_desc(nullptr),
        prim_desc(nullptr),
        gru_prim(nullptr),
        prop_k(pkind),
        direction(dir) {}

  void CreatePrimitiveDesc(MklGRUParams<T> &params,
                                  dnnl::engine &engine) {
    auto descr = CreatePrimDesc<T, Prim>(params, engine,
		                              prop_k, direction);
    gru_desc.reset(descr);
    auto pdesc = new primitiveDesc(*gru_desc.get(), engine);
    prim_desc.reset(pdesc);
  }
  primitiveDesc *get_desc() { return prim_desc.get(); }
  inline void CreatePrimitive() {
    auto gprim = new Prim(*(prim_desc.get()));
    gru_prim.reset(gprim);
    // std::cout<<"Prim created:"<<std::endl;
  }
  inline Prim *get_prim() { return gru_prim.get(); }
  inline void CreateDescAndPrim(MklGRUParams<T> &params, dnnl::engine &engine) {
    CreatePrimitiveDesc(params, engine);
    CreatePrimitive();
  }
  static std::string GetPrimitivekey() {
    std::string prefix("plain");
    if (std::is_same<Desc, dnnl::lbr_gru_forward::desc>()) {
      prefix = "lbr";
    }
    return prefix + "_gru_forward";
  }

// Data members
  std::shared_ptr<Desc> gru_desc;
  std::shared_ptr<primitiveDesc> prim_desc;
  std::shared_ptr<Prim> gru_prim;
  dnnl::prop_kind prop_k;
  dnnl::rnn_direction direction;
};

/*=================================================================
  GRUContext related template arguments
   The FwdContext parameter
==================================================================*/
template <typename CntxtTraits>
struct GRUFwdPolicy {
  using PrimType = GRUPrimitive<CntxtTraits, dnnl::gru_forward>;
  using ParamType = MklGRUParams<CntxtTraits>;
  using Adapter= typename std::conditional<CntxtTraits::HasTimeDim,
                             MklGRUParamAdapter<CntxtTraits>,
                             GRUParamAdapter<CntxtTraits>>::type;
};
/*-------------------------------------------------
 The FwdContext parameter
---------------------------------------------------*/
template <typename CntxtTraits>
struct AUGRUFwdPolicy {
  using PrimType = GRUPrimitive<CntxtTraits, dnnl::augru_forward>;
  using ParamType = MklGRUParams<CntxtTraits>;
  using Adapter= typename std::conditional<CntxtTraits::HasTimeDim,
                             MklGRUParamAdapter<CntxtTraits>,
                             GRUParamAdapter<CntxtTraits>>::type;
};
/*=================================================================
  GRUContext The struct that is passed to MklGenericOp
==================================================================*/
template <typename CntxtTraits, typename ContextPolicy>
struct MklGRUContext {
  using ContextTraits = CntxtTraits;
  // using ContextPolicy = typename std::conditional<CntxtTraits::augru,
  //                           AUGRUFwdContext<CntxtTraits>,
  //                           GRUFwdContext<CntxtTraits>>::type;
  using ParamSign = typename ContextPolicy::ParamType;
  using ParamType = typename ContextPolicy::ParamType;
  using PrimType = typename ContextPolicy::PrimType;
  using Adapter = typename ContextPolicy::Adapter;

  MklGRUContext(Adapter *p_adapt)
      : prim(CntxtTraits::propkind, CntxtTraits::direction),
        cpu_engine(engine(engine::kind::cpu, 0)),
        pAdapt(p_adapt),
        engine_stream(cpu_engine) {
    CreateContext();
  }

  ~MklGRUContext() {}
  //-----------------------------------------------------
  // Creates only parms mem dims -used by getKey;
  //-----------------------------------------------------
  inline void CreateContext() {  // Same as Setup()
    params.GetKernelParams(*pAdapt, cpu_engine);
    params.CreateParamDescs(cpu_engine);
    prim.CreateDescAndPrim(params, cpu_engine);
    params.CreateParamMemory(prim, cpu_engine, engine_stream);
    params.CollectPrimitiveArgs(prim_args);
    primitives.push_back(*prim.gru_prim);
    // prim.CreatePrimitive();
  }
  inline dnnl::engine &get_engine() { return cpu_engine; }
  //-----------------------------------------------------
  inline void SetParamAdapter(Adapter *p_adapt) { pAdapt = p_adapt; }
  //-----------------------------------------------------
  // Can return a value based in input dimensions
  //-----------------------------------------------------
  static inline bool cache_primitive() { return true; }
  //-----------------------------------------------------
  static inline string GetKey() {
    std::string pkey = PrimType::GetPrimitivekey();
    return pkey;
  }
  inline void PreparePrimitiveExec(OpKernelContext *context,
                                   dnnl::stream &cpu_stream) {
    params.ReorderMemSetHandles(context, *pAdapt, prim, prim_args,
		                cpu_engine, cpu_stream);
  }
  inline void PostExecutionUpdates(OpKernelContext *context) {
    params.ResetMemHandles();
    //Only needed for gru ops with time dim
    if constexpr (CntxtTraits::HasTimeDim) {
      pAdapt->CopyOutput();
    }
  }

//Data Members
  ParamType params;
  PrimType prim;
  Adapter *pAdapt;

  std::vector<dnnl::primitive> primitives;
  std::vector<MemoryArgsMap> prim_args;
  const int src_idx = 0;
  const int weights_idx = 1;

  dnnl::engine cpu_engine;
  dnnl::stream engine_stream;
};
/*=================================================================
 The common template parameters for Context
 GRU and AUGRU Inf Context diretion Left to rightInf
 Only these are needed as of now for inference
==================================================================*/
template <typename T, bool TD=false>
  using GRUInfTrait = GRUTraits<float, T, false, TD>;

  //Trait & Poliy for AUGRU Ops
template <typename T, bool TD=false>
  using AUGRUInfTrait = GRUTraits<float, T, true, TD>;

  //Policy has Trait as template param
template <typename T, bool TD=false>
    using AUGRUInfPolicy = AUGRUFwdPolicy<AUGRUInfTrait<T, TD>>;
template <typename T, bool TD=false>
    using AUGRUInfContext = MklGRUContext<AUGRUInfTrait<T, TD>,
                                          AUGRUInfPolicy<T, TD>>;

/*=================================================================
  GRU Forward op
==================================================================*/
template <typename Device, typename T, bool TD=false>
class GRUForwardOp : public OpKernel{
  // using LbrGRUContext = MklGRUContext<LbrGRUFwdTrainLtR<T>>;
  using GRUInfPolicy = GRUFwdPolicy<GRUInfTrait<T,TD>>;
  using GRUInfContext = MklGRUContext<GRUInfTrait<T, TD>, GRUInfPolicy>;

  MklGenericOp<Device, GRUInfContext> *gru_inf;
  // GRUCellBlockOp<CPUDevice, T, false> eigen_gru;

 public:
  explicit GRUForwardOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    gru_inf = new MklGenericOp<Device, GRUInfContext>(ctx);
    if (std::is_same<T, float>::value) {
      (void)SetFPMathMode();
    }
  }
  ~GRUForwardOp() {
    if (gru_inf) delete gru_inf;
  }
  void Compute(OpKernelContext *context) override {
    if (gru_inf) gru_inf->Compute(context);
  }
};
template <typename Device, typename T, bool TD=false>
using AUGRUForwardOp =
       MklGenericOp<Device, AUGRUInfContext<T, TD>>;

// Register DNN kernels for supported operations and supported types - right now
// Register the Block GRU cell kernel for CPU.
#define REGISTER_KERNEL(NAME, T, CLASS, TD)                     \
  REGISTER_KERNEL_BUILDER(                                  \
      Name(NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CLASS<CPUDevice, T, TD>);

#define REGISTER_KERNEL_TYPES(NAME, CLASS, TD) \
  REGISTER_KERNEL(NAME, float, CLASS, TD);     \
  REGISTER_KERNEL(NAME, bfloat16, CLASS, TD);

REGISTER_KERNEL_TYPES("GRUBlockCell", GRUForwardOp, false);
REGISTER_KERNEL_TYPES("AUGRUBlockCell", AUGRUForwardOp, false);
REGISTER_KERNEL_TYPES("MklGRU", GRUForwardOp, true);
REGISTER_KERNEL_TYPES("MklAUGRU", AUGRUForwardOp, true);
#undef REGISTER_KERNEL
#undef REGISTER_KERNEL_TYPES

} //namespace gru_scope

}

#endif
