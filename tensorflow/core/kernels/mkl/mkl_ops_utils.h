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
#ifndef TENSORFLOW_CORE_KERNEL_MKL_OPS_H_
#define TENSORFLOW_CORE_KERNEL_MKL_OPS_H_

#ifdef INTEL_MKL
#include <type_traits>
#include "dnnl.hpp"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {
unsigned char* DummyData = nullptr;
typedef Eigen::ThreadPoolDevice CPUDevice;
using mkltag = memory::format_tag;
bool enable_rnn_debug = false;
/*=================================================================
 The common params used for all ops
  S-Src, W-Weights D-Dst Param
  This it a struct that is used to group together
  oneDNN memory related data and ops into a single class
==================================================================*/
template <typename T>
struct MklMem {

  MklMem()
      : type(MklDnnType<T>()),
        haveDesc(false),
        desc(nullptr),
        mem(nullptr),
        orig_mem(nullptr),
        allocated_buffer_(nullptr),
        data(nullptr) {}
  ~MklMem() {}
  //----------------------------------------------------------------
  void SetParams(dnnl::memory::dims pdim, memory::format_tag tag,
                 memory::format_tag ptag, dnnl::engine eng, std::string mname,
                 const T* pdata = nullptr, bool mkl = false) {
    dim = pdim;
    orig_tag = tag;
    prim_tag = ptag;
    isMkl = mkl;
    // mengine= eng;
    data = const_cast<T*>(pdata);
    name = mname;
    haveDesc = true;
  }
  void SetParams(dnnl::memory::dims pdim, memory::format_tag tag,
                 dnnl::engine eng, std::string mname, const T* pdata = nullptr,
                 bool mkl = false) {
    SetParams(pdim, tag, tag, eng, mname, pdata, mkl);
  }
  //----------------------------------------------------------------
  inline void CreateDesc(bool use_orig = true) {
    // memory::format_tag mtag = MklTensorFormatToMklDnnDataFormat(fmt);

    memory::format_tag ltag = use_orig ? orig_tag : prim_tag;
    auto mdesc =
        (haveDesc) ? new memory::desc(dim, type, ltag) : new memory::desc();
    desc.reset(mdesc);
  }
  //----------------------------------------------------------------
  inline void CreateMemInt(dnnl::engine eng, bool setdummy = true) {
    if (setdummy == false) {
      mem.reset(new memory(*desc.get(), eng));
    } else {
      mem.reset(new memory(*desc.get(), eng, DummyData));
    }
  }
  //----------------------------------------------------------------
  inline void CreatePrimMem(dnnl::engine& eng, bool setdummy = true) {
    CreateDesc(false);
    CreateMemInt(eng, setdummy);
  }
  //----------------------------------------------------------------
  inline void CreateMem(const memory::desc& pdesc, dnnl::engine& eng,
                        bool setdummy) {
    if (setdummy) {
      mem.reset(new memory(pdesc, eng, DummyData));
    } else {
      mem.reset(new memory(pdesc, eng));
    }
  }
  //----------------------------------------------------------------
  inline dnnl::memory& get_mem() { return *(mem.get()); }
  //----------------------------------------------------------------
  inline dnnl::memory::desc& get_desc() { return *desc.get(); }
  inline void set_mem(dnnl::memory* nmem) { mem.reset(nmem); }
  //----------------------------------------------------------------
  inline void set_data_handle(const T* pdata) {
    data = const_cast<T*>(pdata);
    mem->set_data_handle(data);
  }
  //----------------------------------------------------------------
  inline void set_data_handle(stream& fwd_stream, const Tensor** ptensor) {
    const T* ldata = reinterpret_cast<const T*>((*ptensor)->flat<T>().data());
    data = const_cast<T*>(ldata);
    set_data_handle(fwd_stream);
  }
  //----------------------------------------------------------------
  inline void set_data_handle(stream& fwd_stream, Tensor** ptensor) {
    data = reinterpret_cast<T*>((*ptensor)->flat<T>().data());
    set_data_handle(fwd_stream);
  }
  //----------------------------------------------------------------
  inline void set_data_handle(stream& fwd_stream, Tensor& ptensor) {
    data = reinterpret_cast<T*>(ptensor.flat<T>().data());
    set_data_handle(fwd_stream);
  }
  //----------------------------------------------------------------
  inline void reset_data_handle() {
    mem->set_data_handle(tensorflow::DummyData);
  }
  //----------------------------------------------------------------
  inline void set_data_handle(stream& fwd_stream) {
#ifndef ENABLE_ONEDNN_OPENMP
    mem->set_data_handle(static_cast<void*>(data), fwd_stream);
#else
    mem->set_data_handle(static_cast<void*>(data));
#endif
  }
  //----------------------------------------------------------------
  inline bool IsReorderNeeded(const memory::desc& op_pd) const {
    DCHECK(mem);
    bool rval = op_pd != (*desc.get());
    return rval;
  }
  //----------------------------------------------------------------
  bool Reorder(const memory::desc& op_md, const dnnl::engine& eng,
               OpKernelContext* context = nullptr) {
    DCHECK(mem);
    if (IsReorderNeeded(op_md)) {
      if (enable_rnn_debug) {
        VLOG(2) << "Reordering :" << name << std::endl;
      }
      dnnl::memory* reord_mem = new memory(op_md, eng);
      // reord_mem->set_data_handle(data);
      // dnnl::memory *reord_mem= mem.get();
      auto* prim = FindOrCreateReorder<T>(mem.get(), reord_mem);
      std::shared_ptr<stream> cpu_stream;
      MklDnnThreadPool eigen_tp;
      if (context != nullptr) {
        eigen_tp = MklDnnThreadPool(context);
        cpu_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));
      } else {
        cpu_stream.reset(CreateStream(nullptr, prim->GetEngine()));
      }
      std::vector<primitive> net;
      net.push_back(*(prim->GetPrimitive()));
      std::vector<MemoryArgsMap> net_args;
      net_args.push_back({{DNNL_ARG_FROM, *mem}, {DNNL_ARG_TO, *reord_mem}});
      execute_primitives(net, cpu_stream, net_args);
      // Sawp memory pointers to use in arguments.
      orig_mem.reset(mem.get());
      mem.reset(reord_mem);
      return true;
    }
    return false;
  }
  //----------------------------------------------------------------
  void show(std::string msg) {
    if (!enable_rnn_debug) {
      return;
    }
    VLOG(2) << "----------------" << msg << "--------------" << std::endl;
    VLOG(2) << "\t Dim : {";
    for (int i = 0; i < dim.size(); ++i) {
      VLOG(2) << dim[i] << ",";
    }
    VLOG(2) << "}" << std::endl;
    VLOG(2) << "Data :" << data << std::endl;
    if (std::is_same<T, bfloat16>::value) {
      VLOG(2) << "Bfloat16" << std::endl;
    } else {
      VLOG(2) << "float32" << std::endl;
    }
    if (mem.get())
      VLOG(2) << "Mem:" << mem.get() << std::endl;
    else
      VLOG(2) << "Null Mem:" << std::endl;
  }

  //Data Members
  memory::dims dim;
  memory::format_tag orig_tag;
  memory::format_tag prim_tag;
  memory::data_type type;
  // dnnl::engine mengine;
  bool isMkl;
  bool haveDesc;

  std::shared_ptr<dnnl::memory::desc> desc;
  std::shared_ptr<dnnl::memory::desc> old_desc;
  std::shared_ptr<dnnl::memory> mem;
  std::shared_ptr<dnnl::memory> orig_mem;
  std::shared_ptr<void*> allocated_buffer_;

  std::string name;
  T* data;

};
/*=================================================================
  Generic ObejctType Cache for any type of Object
==================================================================*/
template <typename T, int CAP = 1024>
class MklObjectCache {
 public:
  MklObjectCache() {}

  ~MklObjectCache() {}

  T* Get(const string& key) {
    auto& lru_cache = MklObjectCache<T>::GetLRUCache();
    return lru_cache.GetOp(key);
  }

  void Set(const string& key, T* ob) {
    auto& lru_cache = MklObjectCache<T>::GetLRUCache();
    lru_cache.SetOp(key, ob);
  }

 private:
  static inline LRUCache<T>& GetLRUCache() {
    static const int kCapacity = CAP;  // cache capacity
    static thread_local LRUCache<T> lru_cache_(kCapacity);
    return lru_cache_;
  }
};
/*=================================================================
  Generic Primitive.
  A new Primitve that holds an OpContext, specific to an Op
  This is the object stored in the Primitive Cache for each Op
==================================================================*/
template <typename Tin, typename MklOpContext>
class MklGenericPrimitive : public MklPrimitive {
  using ParamAdapter = typename MklOpContext::Adapter;

 public:
  explicit MklGenericPrimitive(ParamAdapter* p_adapter) : OpContext_(nullptr) {
    MklOpContext* opcontext = new MklOpContext(p_adapter);
    OpContext_.reset(opcontext);
  }
  ~MklGenericPrimitive() {}

  /*-------------------------------------------------------
    Execute the primitve
  -------------------------------------------------------*/
  inline void Execute(OpKernelContext* kcontext) {
    std::shared_ptr<stream> cpu_stream;
    MklDnnThreadPool eigen_tp(kcontext);
    cpu_stream.reset(CreateStream(&eigen_tp, OpContext_->get_engine()));
    OpContext_->PreparePrimitiveExec(kcontext, *cpu_stream);

    execute_primitives(OpContext_->primitives, cpu_stream,
                       OpContext_->prim_args);
    OpContext_->PostExecutionUpdates(kcontext);
  }
  /*-------------------------------------------------------
    SetParamAdapter that is created newly for currentop
  -------------------------------------------------------*/
  inline void SetParamAdapter(ParamAdapter* p_adapter) {
    OpContext_->SetParamAdapter(p_adapter);
  }

 private:
  // Should the context be copied
  std::shared_ptr<MklOpContext> OpContext_;
};
/*=================================================================
  Generic template Factory
  A new Primtive Factory that holds the MklGenericPrimitive
  Can be used for any Op that uses MklGenericPrimitive
==================================================================*/
template <typename T, typename MklOpContext>
class MklGenericPrimitiveFactory : public MklPrimitiveFactory<T> {
  using ParamSign = typename MklOpContext::ParamSign;
  using Adapter = typename MklOpContext::Adapter;
  using MklGenPrim = MklGenericPrimitive<T, MklOpContext>;
  using MklGenPrimFactory = MklGenericPrimitiveFactory<T, MklOpContext>;

 public:
  static MklGenPrim* Get(ParamSign& param, Adapter& p_adapter) {
    // Get primitive from the cached pool.
    if (!MklOpContext::cache_primitive()) {
      return new MklGenPrim(&p_adapter);
    }
    MklGenPrim* primitive = nullptr;
    primitive = static_cast<MklGenPrim*>(
        MklGenPrimFactory::GetInstance().GetPrimitive(param));
    if (primitive == nullptr) {
      primitive = new MklGenPrim(&p_adapter);
      MklGenPrimFactory::GetInstance().SetPrimitive(param, primitive);
    }
    return primitive;
  }

  static MklGenericPrimitiveFactory& GetInstance() {
    static MklGenericPrimitiveFactory instance_;
    return instance_;
  }

 private:
  MklGenericPrimitiveFactory() {}
  ~MklGenericPrimitiveFactory() {}

  /*----------------------------------------------------------------
   * Note: Key should be computed in the context of
   *       Primitive & Param, not in the cache obj
   *---------------------------------------------------------------*/
  inline MklPrimitive* GetPrimitive(ParamSign& param) {
    string context_key = MklOpContext::GetKey();
    string key = param.GetKey(context_key);
    if (enable_rnn_debug) {
      VLOG(2) << "Key :" << key << std::endl;
    }
    MklPrimitive* prim = this->GetOp(key);
    return prim;
  }

  //----------------------------------------------------------------
  inline void SetPrimitive(ParamSign& param, MklGenPrim* op) {
    string context_key = MklOpContext::GetKey();
    string key = param.GetKey(context_key);
    this->SetOp(key, op);
  }
};
/*=================================================================
  Generic Op : The template that can be used to instantiate
               any Op that has a class for
                 1)Primitive, 2)Parameters,
                 3) A signature  (similar to param) to get Key
                 4) An Adpater to the TF framework
		 All four can be same class or different

   The compute does the following
      I.   Read Params from TF Adapter
      II.  Gets Signature for Op from Signature Objects
      III. Check Primtive Cache for the primtive.
      IV.  Allocate Outputs
      V.   Execute the Primitive
==================================================================*/
template <typename Device, typename MklOpContext>
class MklGenericOp : public OpKernel {
  using CntxtTraits = typename MklOpContext::ContextTraits;
  using Ti = typename CntxtTraits::Ti;
  using MklGenPrim = MklGenericPrimitive<Ti, MklOpContext>;
  typename MklOpContext::Adapter p_adapter;
  using KeyType = typename MklOpContext::Adapter::KeyType;

 public:
  explicit MklGenericOp(OpKernelConstruction* context)
      : OpKernel(context), p_adapter(context), MklOpPrim(nullptr) {}
  void Compute(OpKernelContext* context) override {
    try {
      p_adapter.ReadParams(context);
      if (MklOpContext::cache_primitive() || !GotLocalCached()) {
        typename MklOpContext::ParamSign psign;
        psign.GetParamSignature(p_adapter);
        MklOpPrim =
            MklGenericPrimitiveFactory<Ti, MklOpContext>::Get(psign, p_adapter);
        CacheLocal(MklOpPrim);
      }

      MklOpPrim->SetParamAdapter(&p_adapter);
      p_adapter.AllocateOutputs(context);
      MklOpPrim->Execute(context);

    } catch (dnnl::error& e) {
      ShowErrorMsg(context, e);
    }
  }
  ~MklGenericOp() {
    // The prim is not cached to delete it
    if (!MklOpContext::cache_primitive() && MklOpPrim) {
      // delete MklOpPrim;
      CleanUpCache();
    }
  }
  inline void ShowErrorMsg(OpKernelContext* context, dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                       string(e.message) + ", in file " + string(__FILE__) +
                       ":" + std::to_string(__LINE__);
    OP_REQUIRES_OK(context, errors::Aborted("Operation received an exception:",
                                            error_msg));
  }
  inline bool GotLocalCached() {
    KeyType key_code = p_adapter.GetSimpleKey();
    if (LocalCache.find(key_code) != LocalCache.end()) {
      MklOpPrim = LocalCache[key_code];
      return true;
    }
    return false;
  }
  inline void CacheLocal(MklGenPrim* Prim) {
    // If primtive is cached globally return
    if (MklOpContext::cache_primitive()) return;
    KeyType key_code = p_adapter.GetSimpleKey();
    LocalCache[key_code] = Prim;
  }
  inline void CleanUpCache() {
    for (auto it = LocalCache.begin(); it != LocalCache.end(); ++it) {
      MklGenPrim* Op = it->second;
      delete Op;
    }
  }

 private:
  MklGenPrim* MklOpPrim;
  std::map<KeyType, MklGenPrim*> LocalCache;
};
}
#endif
#endif
