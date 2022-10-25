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
#ifndef TENSORFLOW_CORE_KERNEL_MKL_RNN_OPS_H_
#define TENSORFLOW_CORE_KERNEL_MKL_RNN_OPS_H_

#ifdef INTEL_MKL
#include <type_traits>
#include "dnnl.hpp"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_ops_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

using PrimArgs = std::vector<std::unordered_map<int, memory>>;
typedef Eigen::ThreadPoolDevice CPUDevice;

/*-------------------------------------------------
 The common template parameters for Context Traits
---------------------------------------------------*/
template <typename PrimaryType, typename InputType, bool AUGRU = false,
	  bool HTD=false, bool LBR=false,
	  dnnl::prop_kind pkind = dnnl::prop_kind::forward_inference,
	  dnnl::rnn_direction dir =
		  dnnl::rnn_direction::unidirectional_left2right>
struct GRUTraits {
  using T = PrimaryType;
  using Ti = InputType;
  static const bool augru = AUGRU;
  static const bool HasTimeDim=HTD;

  static const bool lbr = LBR;
  static const dnnl::prop_kind propkind = pkind;
  static const dnnl::rnn_direction direction = dir;
  static const bool inference = (pkind == dnnl::prop_kind::forward_inference);
};
/*=================================================================
  Cellbound used for weight inputs. This is needed as the
  parameters need to be sliced up to be used for oneDNN primitive
==================================================================*/
struct CellBounds {
  CellBounds(const int batch_size, const int input_size, const int cell_size)
      : batch_size_(batch_size),
        input_size_(input_size),
        cell_size_(cell_size) {}

  // W(0,0)      -> W(ip, cs) = W_ru(0,0) -> (ip, cs)
  // W(ip,0)     -> W(cs, cs) = W_ru(0,cs)-> (cs, cs)
  // W(ip+cs,0)  -> W(ip, cs) = W_c (0,0) -> (ip, cs)

  // U(0,0)      -> W(cs, cs) = W_ru(ip,0) -> (cs, cs)
  // U(cs,0)     -> W(cs, cs) = W_ru(ip,cs)-> (cs, cs)
  // U(cs+cs,0)  -> W(cs, cs) = W_c (ip,0) -> (cs, cs)

  inline Eigen::array<Eigen::DenseIndex, 2> start1() const { return {0, 0}; }

  inline Eigen::array<Eigen::DenseIndex, 2> start2() const {
    return {0, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> start3() const {
    return {input_size_, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> start4() const {
    return {input_size_, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> start5() const {
    return {input_size_ + cell_size_, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> start6() const {
    return {cell_size_, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> start7() const {
    return {cell_size_ + cell_size_, 0};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> dim1() const {
    return {input_size_, cell_size_};
  }

  inline Eigen::array<Eigen::DenseIndex, 2> dimx() const {
    return {cell_size_, cell_size_};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> dim_1cell() const {
    return {0, cell_size_};
  }
  inline Eigen::array<Eigen::DenseIndex, 2> dim_2cell() const {
    return {0, cell_size_ * 2};
  }
  inline Eigen::array<Eigen::Index, 3> hn_start(const int TimeDim) const {
    return {TimeDim-1, 0, 0};
  }
  inline Eigen::array<Eigen::Index, 3> hn_dim() const {
    return {0, batch_size_, cell_size_};
  }

 protected:
  const int batch_size_;
  const int input_size_;
  const int cell_size_;
};
/*---------------------------------------------------------------
 * Reorder used for bias if bias need to be fp32
 * Maybe this is not essential. Another fn can be used.
 *---------------------------------------------------------------*/
template <typename Tx, typename Ty>
void ReorderTensor(Tensor &src_tx, Tensor &dst_ty, memory::dims &dim,
                   stream &fwd_stream, dnnl::engine &cpu_engine) {
  MklMem<Tx> tx_src;
  MklMem<Ty> ty_dst;
  tx_src.SetParams(dim, mkltag::ldgo, cpu_engine, "");
  tx_src.CreatePrimMem(cpu_engine);
  tx_src.set_data_handle(fwd_stream, src_tx);

  ty_dst.SetParams(dim, mkltag::ldgo, cpu_engine, "");
  ty_dst.CreatePrimMem(cpu_engine);
  ty_dst.set_data_handle(fwd_stream, dst_ty);

  auto reorder_desc = dnnl::reorder::primitive_desc(
      cpu_engine, tx_src.get_desc(), cpu_engine, ty_dst.get_desc());
  // dnnl::memory *ro_mem = new memory(dst_desc, cpu_engine);
  CreateAndExecuteReorder(reorder_desc, tx_src.get_mem(), ty_dst.get_mem(),
                          cpu_engine);
  tx_src.reset_data_handle();
  ty_dst.reset_data_handle();
}
/*--------------------------------------------
 * This is used to cache parameters of GRU and other
 * temp parameters/vars
--------------------------------------------*/
template <typename T>
struct GRUWeights {
  using Type = T;
  GRUWeights() {
    // VLOG(2)<<"\n\t\tCreated GRUWeights\n";
  }
  ~GRUWeights() {}
  inline void set_reorder_bias(bool val) { reorder_bias = val; }
  inline void SetBZeros(int b_size) {
    auto tensor = reinterpret_cast<T *>(b_zeros.flat<T>().data());
    for (int i = 0; i < b_size; ++i) {
      tensor[i] = T(0.0);
    }
  }
  inline void SetHPrevZeros(int batch_size, int cell_size) {
    long size = batch_size*cell_size;
    auto tensor = reinterpret_cast<T *>(h_prev_tensor.flat<T>().data());
    for (long i = 0; i < size; ++i) {
      tensor[i] = T(0.0);
    }
  }
  //----------------------------------------
  inline void ReorderBias(memory::dims &bias_dims, stream &fwd_stream,
                          dnnl::engine cpu_engine) {
    if (reorder_bias) {
      ReorderTensor<bfloat16, float>(temp_bias, bias, bias_dims, fwd_stream,
                                     cpu_engine);
      reorder_bias = false;
    }
  }

  inline void ReorderAndCache(OpKernelContext *context, Tensor &src_tensor,
                              Tensor &dst_tensor, memory::desc &src_desc,
                              memory::desc &dst_desc, dnnl::engine &cpu_engine_,
                              bool cache = true) {
    MklDnnData<T> src(&cpu_engine_);
    src.SetUsrMem(src_desc, &src_tensor);
    src.CheckReorderToOpMem(dst_desc, cpu_engine_, context);
    T *src_data = static_cast<T *>(src.GetOpMem().get_data_handle());

    // dst should be already allocated
    if (cache) {
      void *cached_data = static_cast<void *>(dst_tensor.flat<T>().data());
      size_t cached_data_size = src_desc.get_size();
      cached_data_size = src.GetOpMem().get_desc().get_size();
      memcpy(cached_data, src_data, cached_data_size);
    }
  }

  inline void ReorderWeights(OpKernelContext *ctx, memory::desc &src_desc,
                             memory::desc &dst_desc, dnnl::engine &cpu_engine_,
                             bool inference = true) {
    if (!w_reordered || !inference) {
      if (w_cache_tensor.flat<float>().size() != dst_desc.get_size()) {
        TensorShape tf_shape = MklDnnDimsToTFShape(dst_desc.dims());
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(
                     DataTypeToEnum<T>::v(),
                     // tf_shape, //TensorShape({dst_desc.get_size()}),
                     TensorShape({dst_desc.get_size()}), &w_cache_tensor));
      }
      ReorderAndCache(ctx, w_tensor, w_cache_tensor, src_desc, dst_desc,
                      cpu_engine_, inference);
      w_reordered = true;
    }
  }
  inline void ReorderWeightsIter(OpKernelContext *ctx, memory::desc &src_desc,
                                 memory::desc &dst_desc,
                                 dnnl::engine &cpu_engine_,
                                 bool inference = true) {
    if (!w_i_reordered || !inference) {
      if (w_icache_tensor.flat<float>().size() != dst_desc.get_size()) {
        TensorShape tf_shape = MklDnnDimsToTFShape(dst_desc.dims());
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(
                     DataTypeToEnum<T>::v(),
                     // tf_shape, //TensorShape({dst_desc.get_size()}),
                     TensorShape({dst_desc.get_size()}), &w_icache_tensor));
      }
      ReorderAndCache(ctx, w_i_tensor, w_icache_tensor, src_desc, dst_desc,
                      cpu_engine_, inference);
      w_i_reordered = true;
    }
  }
  inline bool WeightsIterNeedReorder() { return (!w_i_reordered); }
  inline bool WeightsNeedReorder() { return (!w_reordered); }

  Tensor w_tensor;
  Tensor w_i_tensor;
  Tensor w_cache_tensor;
  Tensor w_icache_tensor;
  Tensor bias;
  Tensor temp_bias;
  Tensor b_zeros;
  // Used only when Time Dimension >0
  Tensor h_prev_tensor;
  bool cached = false;
  bool reorder_bias = false;
  bool w_reordered = false;
  bool w_i_reordered = false;
};
/*------------------------------------------
class MklParamTranspose : public MklTransposeCpuOp {
Not able to use MklTransposeCpuOp as it fails to link
for test. Need to figure out how to do that.
Till then copying Transpose op as it is
------------------------------------------*/
template <typename T>
class MklParamTranspose {
  public :
    inline void Transpose(OpKernelContext *ctx,
		          const Tensor* weights_in,
		          Tensor &weights_out) {
      int dim1=weights_in->dim_size(0);
      int dim2=weights_in->dim_size(1);
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                TensorShape({dim2,dim1}), &weights_out));
      //ParamTranspose.Transpose(ctx, *weights_in, {1,0}, weights_out);
      TF_CHECK_OK(TransposeND(ctx, *weights_in, &weights_out, {1,0}));
    }
    inline memory::dims ReorderStrides(const memory::dims& strides,
                                          const gtl::ArraySlice<int32>& perm) {
      memory::dims reordered_strides;
      reordered_strides.resize(strides.size());
      for (size_t i = 0; i < strides.size(); ++i) {
        reordered_strides[perm[i]] = strides[i];
      }
      return reordered_strides;
    }
    // Transpose of N-dimensional tensor using oneDNN
    Status TransposeND(OpKernelContext* context, const Tensor& in_tensor,
                       Tensor* out_tensor, const gtl::ArraySlice<int32>& perm) {
      try {
        engine cpu_engine = engine(engine::kind::cpu, 0);
        MklDnnData<T> in(&cpu_engine);
        MklDnnData<T> out(&cpu_engine);

        memory::dims in_dims = TFShapeToMklDnnDims(in_tensor.shape());
        memory::dims out_dims = TFShapeToMklDnnDims(out_tensor->shape());
        memory::dims in_strides = CalculateTFStrides(in_dims);
        // Reorder output strides based on permutation requested.
        memory::dims out_strides =
            ReorderStrides(CalculateTFStrides(out_dims), perm);

        std::shared_ptr<stream> transpose_stream;
        in.SetUsrMem(in_dims, in_strides, &in_tensor);
        // Output dimensions are same as input dimensions. We adjust the layout
        // using strides.
        out.SetUsrMem(in_dims, out_strides, out_tensor);

        std::vector<primitive> net;
        auto* prim = FindOrCreateReorder<T>(in.GetUsrMem(), out.GetUsrMem());
        MklDnnThreadPool eigen_tp(context);
        transpose_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));
        in.SetUsrMemDataHandle(&in_tensor, transpose_stream);
        out.SetUsrMemDataHandle(out_tensor, transpose_stream);
        net.push_back(*(prim->GetPrimitive()));
        std::vector<MemoryArgsMap> net_args;
        net_args.push_back(
            {{DNNL_ARG_FROM, *in.GetUsrMem()}, {DNNL_ARG_TO, *out.GetUsrMem()}});
        execute_primitives(net, transpose_stream, net_args);

        return Status::OK();
      } catch (dnnl::error& e) {
        string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + std::string(e.message) + ", in file " +
                       std::string(__FILE__) + ":" + std::to_string(__LINE__);
        return errors::Aborted("Operation received an exception:", error_msg);
      }
    }
};
/*----------------------------------------
 * The primary interface between TF and oneDNN
 * API Calls
 *----------------------------------------*/
template <typename CntxtParams>
struct GRUParamAdapter {
 public:
  using T = typename CntxtParams::T;
  using Ti = typename CntxtParams::Ti;
  using KeyType = typename std::string;

  explicit GRUParamAdapter(OpKernelConstruction *ctx)
      : batch_size(0), input_size(0), cell_size(0), Wb(nullptr) {
    TimeDim = 1;
    // Check for conditions specific to GRU
    bool lbr_gru =false;
    bool training =false;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lbr", &lbr_gru));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training));
    OP_REQUIRES(ctx, !lbr_gru,
                errors::Unimplemented("Lbr GRU is not implemented: "));
    OP_REQUIRES(ctx, !training,
                errors::Unimplemented("GRU training is not implemented: ["));
  }
  //-----------------------------------------------
  Tensor *get_weights() { return Wb->w_tensor; }
  Tensor *get_weights_iter() { return Wb->w_i_tensor; }
  /*--------------------------------------------
   * generates a Simple key for caching
   * For GRU Time, batch_size are only possible
   * variables
   --------------------------------------------*/
  inline KeyType GetSimpleKey() { return SimpleKey; }
  //-----------------------------------------------
  void AllocateParamTensors(OpKernelContext *ctx, GRUWeights<Ti> *W) {
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(
                       DataTypeToEnum<Ti>::v(),
                       TensorShape({input_size + (cell_size * 2), cell_size}),
                       &W->w_tensor));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(
                       DataTypeToEnum<Ti>::v(),
                       TensorShape({input_size + (cell_size * 2), cell_size}),
                       &W->w_i_tensor));
    int bsize = (lbr) ? 4 : 3;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::v(),
                                TensorShape({cell_size * bsize}), &W->bias));
    if (std::is_same<T, Ti>::value) {
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<Ti>::v(),
                                             TensorShape({cell_size * bsize}),
                                             &W->temp_bias));
    }
    if (lbr) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<Ti>::v(),
                                        TensorShape({cell_size}), &W->b_zeros));
      W->SetBZeros(cell_size);
    }
  }
  //-----------------------------------------------
  inline GRUWeights<Ti> *GetWeightsFromCache(int batch_size, int input_size,
                                             int cell_size) {
    //uintptr_t wru_loc = reinterpret_cast<std::uintptr_t>(w_ru_tensor);
    //uintptr_t wc_loc = reinterpret_cast<std::uintptr_t>(w_c_tensor);
    std::string gru_type ="";
    if constexpr (CntxtParams::augru) {
      gru_type += "Augru";
    } else {
      gru_type += "Gru";
    }
    if constexpr (CntxtParams::HasTimeDim) {
      gru_type += "_TimDim";
    }
    SimpleKey= gru_type+"_"+std::to_string(batch_size);
    std::string key = SimpleKey + "_" +
                      std::to_string(input_size) + "_" +
                      std::to_string(cell_size) + "_";
    if (lbr) {
      key += "_lbr";
    }
    Wb = WeightsCache.Get(key);
    if (!Wb) {
      Wb = new GRUWeights<Ti>();
      WeightsCache.Set(key, Wb);
    }
    return Wb;
  }
  inline GRUWeights<Ti> *GetWeightsFromCache() {
    return GetWeightsFromCache(batch_size, input_size, cell_size);
  }
  /*-----------------------------------------------
    just a wrapper
  /-----------------------------------------------*/
  inline void ReadParams(OpKernelContext *ctx) {
    ReadInputs(ctx);
    CheckInputShapes(ctx);
    ReadParamsCommon(ctx, !CntxtParams::inference);
  }
  inline void ReadInputs(OpKernelContext *ctx) {
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));
    if constexpr (CntxtParams::augru) {
      OP_REQUIRES_OK(ctx, ctx->input("au_x", &attn_tensor));
    }
    batch_size = x_tensor->dim_size(0);
    input_size = x_tensor->dim_size(1);
    cell_size = h_prev_tensor->dim_size(1);
  }
  /*------------------------------------------------
    Fn that reads params and slices them for oneDNN
  -----------------------------------------------*/
  inline void ReadParamsCommon(OpKernelContext *ctx, bool training = false) {
    const CPUDevice &de = ctx->eigen_device<CPUDevice>();

    /* check for contraints of the params */
    Wb = GetWeightsFromCache(batch_size, input_size, cell_size);
    if (!Wb->cached || training) {
      // std::cout<<"\t\tREADING PARAMS"<<std::endl;
      // Read only once for inference
      Tensor w_ru_tensor_t;
      Tensor w_c_tensor_t;
      OP_REQUIRES_OK(ctx, ctx->input("w_ru", &w_ru_tensor));
      OP_REQUIRES_OK(ctx, ctx->input("w_c", &w_c_tensor));
      OP_REQUIRES_OK(ctx, ctx->input("b_ru", &b_ru_tensor));
      OP_REQUIRES_OK(ctx, ctx->input("b_c", &b_c_tensor));

      MklParamTranspose<Ti> ParamTranspose;
      ParamTranspose.Transpose(ctx, w_ru_tensor, w_ru_tensor_t);
      ParamTranspose.Transpose(ctx, w_c_tensor, w_c_tensor_t);

      AllocateParamTensors(ctx, Wb);
      CheckParamShapes(ctx);
      if (enable_rnn_debug) {
        showDims();
      }
      typename TTypes<Ti>::Matrix W = Wb->w_tensor.template matrix<Ti>();
      typename TTypes<Ti>::Matrix U = Wb->w_i_tensor.template matrix<Ti>();

      typename TTypes<Ti>::Matrix W_ru = w_ru_tensor_t.matrix<Ti>();
      typename TTypes<Ti>::Matrix W_c = w_c_tensor_t.matrix<Ti>();
    
      CellBounds cb(batch_size, input_size, cell_size);
      // ip --> input_size,   cs --> cell_size
      //------------------------------------------
      // W(0,0)      -> W(ip, cs) = W_ru(ip,cs) -> (cs, cs)
      // W(cs,0)     -> W(cs, cs) = W_ru(0,cs)-> (ip, cs)
      // W(ip+cs,0)  -> W(ip, cs) = W_c (0,0) -> (ip, cs)
      // W.slice(cb.start1(), cb.dim1()) = (W_ru.slice(cb.start4(), cb.dimx()));
      // W.slice(cb.start3(), cb.dimx()) = (W_ru.slice(cb.start2(), cb.dim1()));

      W.slice(cb.start1(), cb.dim1()) = (W_ru.slice(cb.start3(), cb.dimx()));
      W.slice(cb.start3(), cb.dimx()) = (W_ru.slice(cb.start1(), cb.dim1()));
      W.slice(cb.start5(), cb.dim1()) = (W_c.slice(cb.start1(), cb.dim1()));

      // U(0,0)      -> W(cs, cs) = W_ru(ip,0) -> (cs, cs)
      // W(cs,0)     -> W(cs, cs) = W_ru(0,0)  -> (ip, cs)
      // W(cs+cs,0)  -> W(cs, cs) = W_c (ip,0) -> (cs, cs)
      // U.slice(cb.start1(), cb.dimx()) = (W_ru.slice(cb.start3(), cb.dimx()));
      // U.slice(cb.start6(), cb.dimx()) = (W_ru.slice(cb.start1(), cb.dim1()));

      U.slice(cb.start1(), cb.dimx()) = (W_ru.slice(cb.start4(), cb.dimx()));
      U.slice(cb.start6(), cb.dimx()) = (W_ru.slice(cb.start2(), cb.dimx()));
      U.slice(cb.start7(), cb.dimx()) = (W_c.slice(cb.start2(), cb.dimx()));

      if (b_ru_tensor != nullptr && b_c_tensor != nullptr) {
        int size1 = b_ru_tensor->dim_size(0);
        int size2 = b_c_tensor->dim_size(0);
        int total_size = size1 + size2;
        if (lbr == false && total_size == (3 * cell_size)) {
          gtl::ArraySlice<Tensor> B{*b_ru_tensor, *b_c_tensor};
          if (std::is_same<Ti, T>::value) {
            TF_CHECK_OK(tensor::Concat(B, &Wb->bias));
          } else {
            TF_CHECK_OK(tensor::Concat(B, &Wb->temp_bias));
            // Enables reorder to Wb->bias, before setting data handle :(Ti->T)
            Wb->set_reorder_bias(true);
          }
        } else {
          gtl::ArraySlice<Tensor> B{*b_ru_tensor, Wb->b_zeros, *b_c_tensor};
          if (std::is_same<Ti, T>::value) {
            TF_CHECK_OK(tensor::Concat(B, &Wb->bias));
          } else {
            TF_CHECK_OK(tensor::Concat(B, &Wb->temp_bias));
            // Enables reorder to Wb->bias, before setting data handle :(Ti->T)
            Wb->set_reorder_bias(true);
          }
        }
      }
      Wb->cached = (training == false);
    }
  }
  //-----------------------------------------------
  void showDims() {
    if (!enable_rnn_debug) {
      return;
    }
    Wb = GetWeightsFromCache();
    VLOG(2) << "Splitting Weights:" << std::endl;
    VLOG(2) << "Input Size :" << input_size << std::endl;
    VLOG(2) << "Cell Size :" << cell_size << std::endl;
    VLOG(2) << "W_ru :" << w_ru_tensor->dim_size(0) << "x"
            << w_ru_tensor->dim_size(1) << std::endl;
    VLOG(2) << "W_c  :" << w_c_tensor->dim_size(0) << "x"
            << w_c_tensor->dim_size(1) << std::endl;
    VLOG(2) << "W    :" << Wb->w_tensor.dim_size(0) << "x"
            << Wb->w_tensor.dim_size(1) << std::endl;
    VLOG(2) << "W_i  :" << Wb->w_i_tensor.dim_size(0) << "x"
            << Wb->w_i_tensor.dim_size(1) << std::endl;
  }
  void showTensors(std::string msg) {
    if (!enable_rnn_debug) return;
    Wb = GetWeightsFromCache();
    VLOG(2) << " From" << msg << std::endl;
    VLOG(2) << "X Tensor:" << x_tensor->DebugString() << std::endl;
    VLOG(2) << "H Prev  :" << h_prev_tensor->DebugString() << std::endl;
    if constexpr (CntxtParams::augru) {
      VLOG(2) << "Attn :" << attn_tensor->DebugString() << std::endl;
    }
    VLOG(2) << "W_ru    :" << w_ru_tensor->DebugString() << std::endl;
    VLOG(2) << "W_c     :" << w_c_tensor->DebugString() << std::endl;
    VLOG(2) << "B_ru    :" << b_ru_tensor->DebugString() << std::endl;
    VLOG(2) << "B_c     :" << b_c_tensor->DebugString() << std::endl;
    VLOG(2) << "W Tensor:" << Wb->w_tensor.DebugString() << std::endl;
    VLOG(2) << "W I Tensor:" << Wb->w_i_tensor.DebugString() << std::endl;
    VLOG(2) << "Bias    :" << Wb->bias.DebugString() << std::endl;
  }
  //--------------------------------------------------
  inline void AllocateOutputs(OpKernelContext *ctx) {
    /*
    if (CntxtParams::inference ==false) {
      OP_REQUIRES_OK(
          ctx, ctx->allocate_output("r", TensorShape({batch_size, cell_size}),
                                  &r_tensor));

      OP_REQUIRES_OK(
          ctx, ctx->allocate_output("u", TensorShape({batch_size, cell_size}),
                                  &u_tensor));

      OP_REQUIRES_OK(
          ctx, ctx->allocate_output("c", TensorShape({batch_size, cell_size}),
                                  &c_tensor));
    }
    */
    OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                            {"h_prev"}, "h",
                            TensorShape({batch_size, cell_size}), 
			    &h_tensor));
  }
  //--------------------------------------------------
  inline void GetDimValues(memory::dim &TimeSteps, memory::dim &L,
                           memory::dim &N, memory::dim &C, memory::dim &D) {
    // As of now only N & C are updates
    TimeSteps = TimeDim;
    N = batch_size;
    C = cell_size;
  }
  //--------------------------------------------------
  inline void CheckInputShapes(OpKernelContext *ctx){
    // Shape of 'h' must be [batch_size, cell_size]
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(0) == batch_size,
                errors::InvalidArgument("h_prev.dims(0) != batch_size: ",
                                        h_prev_tensor->dim_size(0), " vs. ",
                                        batch_size));
    OP_REQUIRES(ctx, h_prev_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "h_prev.dims(1) != cell_size: ", h_prev_tensor->dim_size(1),
                    " vs. ", cell_size));
  }
  inline void CheckParamShapes(OpKernelContext *ctx){
    // Shape of 'w_ru' must be [input_size+cell_size, 2*cell_size]
    OP_REQUIRES(ctx, w_ru_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_ru.dim_size(0) != input_size + cell_size: ",
                    w_ru_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_ru_tensor->dim_size(1) == cell_size * 2,
                errors::InvalidArgument("w_ru.dim_size(1) != cell_size * 2: ",
                                        w_ru_tensor->dim_size(1), " vs. ",
                                        cell_size * 2));

    // Shape of 'w_c' must be [input_size+cell_size, cell_size]
    OP_REQUIRES(ctx, w_c_tensor->dim_size(0) == input_size + cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(0) != input_size + cell_size: ",
                    w_c_tensor->dim_size(0), " vs. ", input_size + cell_size));

    OP_REQUIRES(ctx, w_c_tensor->dim_size(1) == cell_size,
                errors::InvalidArgument(
                    "w_c.dim_size(1) != cell_size: ", w_c_tensor->dim_size(1),
                    " vs. ", cell_size));

    // Shape of 'b_ru' must be [2*cell_size]
    OP_REQUIRES(ctx, b_ru_tensor->dim_size(0) == cell_size * 2,
                errors::InvalidArgument("b_ru.dim_size(0) != cell_size * 2: ",
                                        b_ru_tensor->dim_size(0), " vs. ",
                                        cell_size * 2));

    OP_REQUIRES(ctx, b_ru_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_ru must be 1",
                                        b_ru_tensor->dims(), " vs. 1", 1));
    // Shape of 'b_c' must be [cell_size]
    OP_REQUIRES(ctx, b_c_tensor->dim_size(0) == cell_size,
                errors::InvalidArgument(
                    "b_c.dim_size(0) != cell_size: ", b_c_tensor->dim_size(0),
                    " vs. ", cell_size));
    OP_REQUIRES(ctx, b_c_tensor->dims() == 1,
                errors::InvalidArgument("Rank of b_c must be 1",
                                        b_c_tensor->dims(), " vs. 1"));
  }
  inline void ReorderInputs(OpKernelContext *context) { }
  inline void CopyOutput() {}

// Data Members all public
  const Tensor *x_tensor = nullptr;
  const Tensor *h_prev_tensor = nullptr;
  const Tensor *w_ru_tensor = nullptr;
  const Tensor *w_c_tensor = nullptr;
  const Tensor *b_ru_tensor = nullptr;
  const Tensor *b_c_tensor = nullptr;
  const Tensor *attn_tensor = nullptr;

  Tensor *r_tensor = nullptr;
  Tensor *u_tensor = nullptr;
  Tensor *c_tensor = nullptr;
  Tensor *h_tensor = nullptr;

  // Forward weights may need caching, so use global
  GRUWeights<Ti> *Wb;
  MklObjectCache<GRUWeights<Ti>> WeightsCache;

  bool lbr = CntxtParams::lbr;
  int TimeDim;
  // outputs

  int batch_size;
  int input_size;
  int cell_size;
  KeyType SimpleKey="";
};
/*-------------------------------------------------------------
 * Defining a new ParamAdapter for MKlGRU ops
 * Handles specific handling needed with TimeDim >1
 * ------------------------------------------------------------*/
template <typename CntxtParams>
struct MklGRUParamAdapter : public GRUParamAdapter<CntxtParams> {
  using T = typename CntxtParams::T;
  using Ti = typename CntxtParams::Ti;
  using Base= GRUParamAdapter<CntxtParams>;
  using Base::TimeDim;
  using Base::x_tensor;
  using Base::h_prev_tensor;
  using Base::h_tensor;
  using Base::attn_tensor;
  using Base::batch_size;
  using Base::cell_size;
  using Base::input_size;

  explicit MklGRUParamAdapter(OpKernelConstruction *context) :
	     GRUParamAdapter<CntxtParams>(context) {

    int TDim=1;
    if (context->HasAttr("TimeDim")) {
      OP_REQUIRES_OK(context, context->GetAttr("TimeDim", &TDim));
      TimeDim=TDim;
    }
    std::string format="";
    if (context->HasAttr("x_format")) {
      OP_REQUIRES_OK(context, context->GetAttr("x_format", &format));
      X_format_tnc=(format=="TNC");
    }
    format="";
    if (context->HasAttr("au_format")) {
      OP_REQUIRES_OK(context, context->GetAttr("au_format", &format));
      AUX_format_tnc=(format=="TNC");
    }
  }
  inline void ReadParams(OpKernelContext *ctx) {
    ReadInputs(ctx);
    CheckInputShapes(ctx);
    Base::ReadParamsCommon(ctx, !CntxtParams::inference);
    //CheckParamShapes(ctx);
  }
  inline void ReadInputs(OpKernelContext *ctx) {
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));
    OP_REQUIRES_OK(ctx, ctx->input("h_prev", &h_prev_tensor));
    if constexpr (CntxtParams::augru) {
      OP_REQUIRES_OK(ctx, ctx->input("au_x", &attn_tensor));
    }
    if (X_format_tnc) {
      TimeDim = x_tensor->dim_size(0);
      batch_size = x_tensor->dim_size(1);
    } else {
      TimeDim = x_tensor->dim_size(1);
      batch_size = x_tensor->dim_size(0);
    }
    input_size = x_tensor->dim_size(2);
    cell_size = x_tensor->dim_size(2);
  }
  inline void CheckInputShapes(OpKernelContext *ctx) {
  }
  inline void CheckParamShapes(OpKernelContext *ctx) {
  }
  inline void AllocateOutputs(OpKernelContext *ctx) {
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
                            "h_out",
                            TensorShape({TimeDim, batch_size, cell_size}), 
			    &h_tensor));
    OP_REQUIRES_OK(
          ctx, ctx->allocate_output("h_n", 
		                    TensorShape({batch_size, cell_size}),
                                    &h_n_tensor));
  }
  void Slice(const Tensor& t, Tensor& res, int pos) {
    // CHECK should never fail here, since the number of elements must match
    //CHECK(res.CopyFrom(t.Slice(pos, pos + 1), {t.dim_size(1), t.dim_size(2)}));
    auto src_ptr = t.flat<Ti>().data();
    auto copy_count = t.dim_size(1)*t.dim_size(2);
    auto spos = t.shape().num_elements() - copy_count;
    std::copy_n(src_ptr+spos, copy_count, res.flat<Ti>().data());
  }
  inline void CopyOutput() {
    Tensor& H_tensor = *h_tensor;
    Slice(H_tensor, *h_n_tensor, TimeDim-1);
  }
  inline void ReorderInplace(OpKernelContext *context,
		             const Tensor *reorder_tensor, Tensor &reordered_tensor,
			     dnnl::memory::desc &src_desc,
			     dnnl::memory::desc &dst_desc) {
    //MklDnnData<Ti> src(&engine);
    //src.SetUsrMem(src_desc, reorder_tensor);
    int first_dim = reorder_tensor->dim_size(0);
    int secon_dim = reorder_tensor->dim_size(1);
    int third_dim = reorder_tensor->dim_size(2);
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                   DataTypeToEnum<Ti>::v(),
                   TensorShape({secon_dim, first_dim, third_dim}),
	  	   &reordered_tensor));
    //src.CheckReorderToOpMem(dst_desc, &reordered_tensor, context);
    MklParamTranspose<Ti> ParamTranspose;
    TF_CHECK_OK(ParamTranspose.TransposeND(
	    context, *reorder_tensor, &reordered_tensor, {1,0,2}));
  }
  inline void ReorderInput(OpKernelContext *context, 
		           const Tensor *reorder_tensor, 
			   Tensor &reordered_tensor) {
    auto type = MklDnnType<Ti>();
    int first_dim = reorder_tensor->dim_size(0);
    int secon_dim = reorder_tensor->dim_size(1);
    int third_dim = reorder_tensor->dim_size(2);
    //dnnl::memory::dims src_dims{batch_size, TimeDim, cell_size};
    //dnnl::memory::dims dst_dims{TimeDim, batch_size, cell_size};
    dnnl::memory::dims src_dims{first_dim, secon_dim, third_dim};
    dnnl::memory::dims dst_dims{secon_dim, first_dim, third_dim};
    auto src_md = memory::desc(src_dims, type, mkltag::ntc);
    auto dst_md = memory::desc(dst_dims, type, mkltag::tnc);
    ReorderInplace(context, reorder_tensor, reordered_tensor, src_md, dst_md);
  }
  inline void ReorderInputs(OpKernelContext *context){
    if (!X_format_tnc) {
      ReorderInput(context, x_tensor, td_x_tensor);
      x_tensor=&td_x_tensor;
    }
    if constexpr (CntxtParams::augru) {
      if (!AUX_format_tnc) {
        ReorderInput(context, attn_tensor, td_attn_tensor);
	attn_tensor=&td_attn_tensor;
      }
    }
  }

// Data Members
  Tensor *h_n_tensor;
  bool X_format_tnc=true;
  bool AUX_format_tnc=true;
  Tensor td_x_tensor;
  Tensor td_attn_tensor;
};
/*-------------------------------------------------
 The common params used for all GRU ops
 Holds all the arguments for GRU Ops in MklMem Type
---------------------------------------------------*/
template <typename CntxtParams>
struct MklGRUParams {
  using T = typename CntxtParams::T;
  using Ti = typename CntxtParams::Ti;

  // Tensor dimensions default values.
  memory::dim N = 0,  // batch size
      TimeS = 1,      // time steps
      Channels = 0,   // channels
      Gates = 3,      // gates
      Layers = 1,     // layers
      Dir = 1,        // direction
      biasG = Gates;

 public:
  MklGRUParams() {}
  void show(std::string prefix) {
    if (!enable_rnn_debug) {
      return;
    }
    src_layer.show(prefix + "Src Layer :");
    src_iter.show(prefix + "Src Iter:");
    if constexpr (CntxtParams::augru) {
      attention.show(prefix + "Attention");
    }
    weights_layer.show(prefix + "Weights :");
    weights_iter.show(prefix + "Weights Iter:");
    bias.show(prefix + "Bias :");
    dst_layer.show(prefix + "Dst Layer:");
    dst_iter.show(prefix + "Dst iter:");
  }
  //----------------------------------------------------------------
  inline void CollectPrimitiveArgs(PrimArgs &prim_args) {
    std::unordered_map<int, memory> args;
    args.insert({DNNL_ARG_SRC_LAYER, *src_layer.mem});
    args.insert({DNNL_ARG_SRC_ITER, *src_iter.mem});
    args.insert({DNNL_ARG_WEIGHTS_LAYER, *weights_layer.mem});
    args.insert({DNNL_ARG_WEIGHTS_ITER, *weights_iter.mem});
    args.insert({DNNL_ARG_BIAS, *bias.mem});
    args.insert({DNNL_ARG_DST_LAYER, *dst_layer.mem});
    args.insert({DNNL_ARG_DST_ITER, *dst_iter.mem});
    args.insert({DNNL_ARG_WORKSPACE, *workspace.mem});
    if constexpr (CntxtParams::augru) {
      args.insert({DNNL_ARG_AUGRU_ATTENTION, *attention.mem});
    }
    prim_args.push_back(args);
  }
  //----------------------------------------------------------------
  inline void CreateParamDescs(dnnl::engine &eng) {
    src_layer.CreateDesc();
    src_iter.CreateDesc();
    weights_layer.CreateDesc();
    weights_iter.CreateDesc();
    bias.CreateDesc();
    dst_layer.CreateDesc();
    dst_iter.CreateDesc();
    if constexpr (CntxtParams::augru) {
      attention.CreateDesc();
    }
  }
  //----------------------------------------------------------------
  template <typename PRIM, typename PAdapter>
  inline void ReorderMemIfNeeded(OpKernelContext *context,
                                 PAdapter &Adapter, PRIM &prim,
                                 dnnl::engine &eng, dnnl::stream &estream) {
    auto pd = prim.get_desc();
    // reorder_W = false;
    // reorder_W_iter=false;
    if (reorder_W) {
      if (!inference || Adapter.Wb->WeightsNeedReorder()) {
        memory::desc w_desc = pd->weights_desc();
        Adapter.Wb->ReorderWeights(context, weights_layer.get_desc(), w_desc,
                                   eng, inference);
      }
    }
    if (reorder_W_iter) {
      if (!inference || Adapter.Wb->WeightsIterNeedReorder()) {
        memory::desc wi_desc = pd->weights_iter_desc();
        Adapter.Wb->ReorderWeightsIter(context, weights_iter.get_desc(),
                                       wi_desc, eng, inference);
      }
    }
    if constexpr(CntxtParams::HasTimeDim)  {
      Adapter.ReorderInputs(context);
    }
  }
  //----------------------------------------------------------------
  template <typename PRIM>
  inline void SetReorderFlags(PRIM &prim) {
    auto pd = prim.get_desc();
    reorder_W = !(weights_layer.get_desc() == pd->weights_desc());
    reorder_W_iter = !(weights_iter.get_desc() == pd->weights_iter_desc());
  }
  //----------------------------------------------------------------
  template <typename PRIM>
  void CreateParamMemory(PRIM &prim, dnnl::engine &eng, dnnl::stream &estream) {
    auto pd = prim.get_desc();

    src_layer.CreatePrimMem(eng);
    src_iter.CreatePrimMem(eng, false);
    weights_layer.CreatePrimMem(eng);
    weights_iter.CreatePrimMem(eng);
    bias.CreatePrimMem(eng);
    dst_layer.CreatePrimMem(eng);
    dst_iter.CreatePrimMem(eng, false);
    workspace.CreateMem(pd->workspace_desc(), eng, false);
    // Should try reorder here itself
    SetReorderFlags(prim);
    if constexpr (CntxtParams::augru) {
      attention.CreatePrimMem(eng);
    }
  }
  //----------------------------------------------------------------
  template <typename PAdapter>
  inline void SetMemHandles(PAdapter &p_extr,
                            dnnl::engine &eng, stream &data_stream) {
    if (enable_rnn_debug) {
      p_extr.showTensors("Before Setting Handles");
    }
    src_layer.set_data_handle(data_stream, &p_extr.x_tensor);
    src_iter.set_data_handle(data_stream, &p_extr.h_prev_tensor);

    // weights_layer.set_data_handle(data_stream, p_extr.Wb->w_tensor);
    // weights_iter.set_data_handle(data_stream, p_extr.Wb->w_i_tensor);
    weights_layer.set_data_handle(data_stream, (inference && reorder_W)
                                                   ? p_extr.Wb->w_cache_tensor
                                                   : p_extr.Wb->w_tensor);
    weights_iter.set_data_handle(data_stream, (inference && reorder_W_iter)
                                                  ? p_extr.Wb->w_icache_tensor
                                                  : p_extr.Wb->w_i_tensor);
    if (std::is_same<T, Ti>::value == false) {
      // Reorder done only if types are different
      // oneDNN BF16 prim needs bias in fp32
      p_extr.Wb->ReorderBias(bias.dim, data_stream, eng);
    }
    bias.set_data_handle(data_stream, p_extr.Wb->bias);

    // dst_layer.set_data_handle(data_stream, p_extr.Wb->temp_output);
    dst_layer.set_data_handle(data_stream, &p_extr.h_tensor);
    if constexpr (CntxtParams::augru) {
      attention.set_data_handle(data_stream, &p_extr.attn_tensor);
    }
  }
  //----------------------------------------------------------------
  inline void ResetMemHandles() {
    src_layer.reset_data_handle();
    // src_iter.reset_data_handle();
    weights_layer.reset_data_handle();
    weights_iter.reset_data_handle();
    bias.reset_data_handle();
    dst_layer.reset_data_handle();
    // dst_iter.reset_data_handle();
    // workspace.reset_data_handle();
    if constexpr (CntxtParams::augru) {
      attention.reset_data_handle();
    }
  }
  //----------------------------------------------------------------
  template <typename Tx>
  inline void AddKey(FactoryKeyCreator &key_creator, MklMem<Tx> &mem) {
    key_creator.AddAsKey(mem.dim);
  }
  //----------------------------------------------------------------
  inline void AddKeys(FactoryKeyCreator &key_creator) {
    /*
    AddKey(key_creator, src_layer);
    AddKey(key_creator, weights_layer);
    */
    // Only Uniq Dims for Keys TimeS, L, N, C, D, G, biasG;
    key_creator.AddAsKey(TimeS);
    key_creator.AddAsKey(Layers);
    key_creator.AddAsKey(N);
    key_creator.AddAsKey(Channels);
    key_creator.AddAsKey(Dir);
    key_creator.AddAsKey(Gates);
    key_creator.AddAsKey(biasG);
    // This may not be needed and GRU and AUGRU are diff primtives
    if constexpr (CntxtParams::augru) {
      key_creator.AddAsKey(TimeS);
    }
  }
  /*-------------------------------------------------------
   * Key contruction for primitive cache
   *-------------------------------------------------------*/
  inline std::string GetKey(std::string prefix) {
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    AddKeys(key_creator);
    return key_creator.GetKey();
  }
  template <typename PAdapter>
  inline void GetParamSignature(PAdapter &p_extr) {
    p_extr.GetDimValues(TimeS, Layers, N, Channels, Dir);
    biasG = Gates + (CntxtParams::lbr ? 1 : 0);
  }
  /*-------------------------------------------------------
   * This reads and constructs the essntial MklMem params
   *-------------------------------------------------------*/
  template <typename PAdapter>
  inline void GetKernelParams(PAdapter &p_extr,
                              dnnl::engine &eng) {
    p_extr.GetDimValues(TimeS, Layers, N, Channels, Dir);
    biasG = Gates + (CntxtParams::lbr ? 1 : 0);
    // L-Layers, D-Directions,
    // C-Inputs, G-Gates, C-OutputActivations
    memory::dims src_dims = {TimeS, N, Channels};
    memory::dims attention_dims = {TimeS, N, 1};
    memory::dims weights_dims = {Layers, Dir, Channels, Gates, Channels};
    memory::dims bias_dims = {Layers, Dir, biasG, Channels};
    memory::dims src_iter_dims = {Layers, Dir, N, Channels};

    if (enable_rnn_debug) {
      VLOG(2) << "-------------------------------" << std::endl;
      VLOG(2) << "TimeS   :" << TimeS << std::endl;
      VLOG(2) << "Layers :" << Layers << std::endl;
      VLOG(2) << "N      :" << N << std::endl;
      VLOG(2) << "Dir    :" << Dir << std::endl;
      VLOG(2) << "Gates  :" << Gates << std::endl;
      VLOG(2) << "biasG  :" << biasG << std::endl;
      VLOG(2) << "Channels :" << Channels << std::endl;
    }

    src_layer.SetParams(src_dims, mkltag::tnc, eng, "src_layer");
    src_iter.SetParams(src_iter_dims, mkltag::ldnc, eng, "src_iter");

    if constexpr (CntxtParams::augru) {
      attention.SetParams(attention_dims, mkltag::tnc, eng, "attn_layer");
    }
    weights_layer.SetParams(weights_dims, memory::format_tag::any,
                            memory::format_tag::ldgoi, eng, "weights_layer");
    weights_iter.SetParams(weights_dims, memory::format_tag::any,
                           memory::format_tag::ldgoi, eng, "weights_iter");
    bias.SetParams(bias_dims, mkltag::ldgo, eng, "bias");

    dst_layer.SetParams(src_dims, mkltag::tnc, eng, "dst_layer");
    dst_iter.SetParams(src_iter_dims, mkltag::ldnc, eng, "dst_iter");
    // workspace.SetParams(weights_dims, mkltag::any, 4, eng, "workspace");
  }
  template <typename PRIM, typename PAdapter>
  inline void ReorderMemSetHandles(OpKernelContext *context,
                                   PAdapter &pAdapt, PRIM &prim,
                                   PrimArgs &prim_args, dnnl::engine &cpu_engine,
                                   stream &cpu_stream) {
    ReorderMemIfNeeded(context, pAdapt, prim, cpu_engine, cpu_stream);
    SetMemHandles(pAdapt, cpu_engine, cpu_stream);
    // CollectPrimitiveArgs(prim_args);
  }

// Data Members
  bool inference = CntxtParams::inference;

  MklMem<Ti> src_layer;
  MklMem<Ti> src_iter;
  MklMem<Ti> attention;
  MklMem<Ti> weights_layer;
  MklMem<Ti> weights_iter;
  MklMem<T> bias;
  MklMem<Ti> dst_layer;
  MklMem<Ti> dst_iter;
  MklMem<T> workspace;
  bool reorder_W = false;
  bool reorder_W_iter = false;
};
}
#endif
#endif
