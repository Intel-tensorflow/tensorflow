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

#if defined(INTEL_MKL) && defined(ENABLE_MKL)
#define EIGEN_USE_THREADS

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

template <typename T>
static Tensor GetRandomTensor(const TensorShape& shape) {
  DataType type = DataTypeToEnum<T>::v();
  Tensor tensor(type, TensorShape(shape));
  tensor.flat<T>() = tensor.flat<T>().setRandom();
  return tensor;
}

enum gru_type {
  NONE=0,
  GRUBlockCell,
  AUGRUBlockCell,
  MklGRU,
  MklAUGRU
};
struct gru_data{
  gru_type gtype;
  std::string node_name;
  std::string op_name;
};
template <typename T>
struct GRUParams {
  int TimeDim;
  int BatchSize;
  int Channels;
  Tensor X;
  Tensor h_prev;
  Tensor attn; 
  Tensor w_ru;
  Tensor w_c;
  Tensor b_ru;
  Tensor b_c;
  gru_type gtype;
  const gru_data gru_info[5] ={
     {NONE, "", ""},
     {GRUBlockCell, "gru_block_cell", "GRUBlockCell"},
     {AUGRUBlockCell, "augru_block_cell", "AUGRUBlockCell"},
     {MklGRU, "mkl_gru", "MklGRU"},
     {MklAUGRU, "mkl_augru", "MklAUGRU"},
  };
  explicit GRUParams(int Time, int N, int C, gru_type g):
                     TimeDim(Time), BatchSize(N), Channels(C), gtype(g) {

    X = GetRandomTensor<T>({TimeDim, BatchSize, Channels});
    h_prev = GetRandomTensor<T>({TimeDim, BatchSize, Channels});
    if (g==AUGRUBlockCell || MklAUGRU) {
      attn = GetRandomTensor<T>({TimeDim, BatchSize});
    }

    // Multiplied Challes by 2 for 2 gates r & u
    w_ru = GetRandomTensor<T>({Channels*2, Channels*2}); 
    w_c = GetRandomTensor<T>({Channels*2, Channels}); 
    b_ru = GetRandomTensor<T>({Channels*2}); 
    // For lbr we need *2 for c also
    b_c = GetRandomTensor<T>({Channels}); 
  }
  bool HasAugumentedInput() {
    return ((gtype==AUGRUBlockCell) || (gtype==MklAUGRU));
  }
  std::string GRUOp() {
    return gru_info[gtype].op_name;
  }
  std::string GRUNode() {
    return gru_info[gtype].node_name;
  }
};
template <typename T>
struct GRUOpTest : public OpsTestBase {

  void RunMklGRUOp(GRUParams<T>& params, Tensor* output) {
    DataType dtype = DataTypeToEnum<T>::v();
    std::string gru_op = params.GRUOp();
    std::string gru_node = params.GRUNode();
    if (params.HasAugumentedInput()) {
      TF_EXPECT_OK(NodeDefBuilder(gru_node, gru_op)
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Attr("T", dtype)
                       .Attr("lbr", false)
                       .Attr("training", false)
                       .Finalize(node_def()));
    } else {
      TF_EXPECT_OK(NodeDefBuilder(gru_node, gru_op)
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Attr("T", dtype)
                       .Attr("lbr", false)
                       .Attr("training", false)
                       .Finalize(node_def()));
    }
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(params.X.shape(), params.X.template flat<T>());
    AddInputFromArray<T>(params.h_prev.shape(), params.h_prev.template flat<T>());
    if (params.HasAugumentedInput()) {
      AddInputFromArray<T>(params.attn.shape(), params.attn.template flat<T>());
    }
    AddInputFromArray<T>(params.w_ru.shape(), params.w_ru.template flat<T>());
    AddInputFromArray<T>(params.w_c.shape(), params.w_c.template flat<T>());

    AddInputFromArray<T>(params.b_ru.shape(), params.b_ru.template flat<T>());
    AddInputFromArray<T>(params.b_c.shape(), params.b_c.template flat<T>());

    TF_ASSERT_OK(RunOpKernel());
    const Tensor& output_tensor = *GetOutput(3);
    *output= output_tensor;
  }
  inline auto GenConst(const tensorflow::Scope& root,
		std::string name, Tensor &tensor) {
    return ops::Const(root.WithOpName(name), Input::Initializer(tensor));
  }
  Output GRU(GRUParams<T>& params,
	     const tensorflow::Scope& root,
	     Output& inp_h_prev) {
    //auto root = tensorflow::Scope::NewRootScope();
    auto inp_x =ops::Const(root.WithOpName("x"), 
		       Input::Initializer(params.X));
    //auto inp_h_prev =GenConst("h_prev", params.h_prev);
    auto axis =ops::Const(root.WithOpName("concat_axis"), {1}, {});

    Output next_op = 
      ops::Concat(root.WithOpName("concat"), {inp_x, inp_h_prev}, axis);
    auto inp_w_ru=GenConst(root, "w_ru", params.w_ru);
    auto mmul_op = ops::MatMul(root.WithOpName("matmul_ru"), next_op, inp_w_ru);

    auto inp_b_ru=ops::Const(root.WithOpName("b_ru"), 
		          Input::Initializer(params.b_ru));
    auto mmba = ops::BiasAdd(root.WithOpName("bias"), mmul_op, inp_b_ru);
    auto mmba_sig= ops::Sigmoid(root.WithOpName("sigmoid"), mmba);

    auto saxis =ops::Const(root.WithOpName("split_dim"), {1}, {});
    auto split = ops::Split(root.WithOpName("split"), saxis, mmba_sig, 2);

    auto r_out= split[0];
    auto u_out= split[1];
    if (params.HasAugumentedInput()) {
      u_out = ops::Mul(root.WithOpName("mula"), params.attn, u_out);
    }
    auto s1 =ops::Const(root.WithOpName("C1"), {1}, {});
    auto sub = ops::Sub(root.WithOpName("sub"), s1, u_out);
    auto mul1 = ops::Mul(root.WithOpName("mul1"), r_out, inp_h_prev);
    auto mul2 = ops::Mul(root.WithOpName("mu21"), u_out, inp_h_prev);
   
    Input mul_inp=mul1;
    //tensorflow::InputList ccat_values = {inp_x, mul_inp};
    Output ccat_op_2= 
	    ops::Concat(root.WithOpName("Concat_2"), {inp_x, mul_inp}, axis);
    auto inp_w_c =GenConst(root, "w_c", params.w_c);
    auto mmul_c = ops::MatMul(root.WithOpName("matmul_c"), ccat_op_2, inp_w_c);

    auto inp_b_c=GenConst(root, "b_c", params.b_c);
    auto bias_c = ops::BiasAdd(root.WithOpName("bias_c"), mmul_c, inp_b_c);
    auto tanh_c= ops::Tanh(root.WithOpName("tanh_c"), bias_c);

    auto mul3 = ops::Mul(root.WithOpName("mul_o"), sub, tanh_c);
    auto add_v2 = ops::AddV2(root.WithOpName("addv2"), mul2, mul3);
    return add_v2;
  }
  static void RunAndFetch(const tensorflow::Scope& root, 
	                  const string& fetch,
                          Tensor* output) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }
  void RunGRU(GRUParams<T>& params, Tensor *output) {
    auto root = tensorflow::Scope::NewRootScope();
    auto inp_h_prev =GenConst(root, "h_prev", params.h_prev);
    auto addv2 = GRU(params, root, inp_h_prev);
    RunAndFetch(root, addv2.name(), output);
  }
  void RunGRUTimeDim(GRUParams<T>& params, Tensor *output) {
    auto root = tensorflow::Scope::NewRootScope();
    auto inp_h_prev =GenConst(root, "h_prev", params.h_prev);

    Output addv2 = GRU(params, root, inp_h_prev);
    for (int i=0; i<params.TimeDim-1; ++i) {
      addv2 = GRU(params, root, addv2);
    }
    RunAndFetch(root, addv2.name(), output);
  }
  void VerifyGRU(gru_type gtype, int Time, int N, int C){
    Tensor gru_output;
    Tensor graph_output;
    GRUParams<T> params(Time,N,C,gtype);

    if (Time >1) {
      RunGRUTimeDim(params, &graph_output);
    } else {
      RunGRU(params, &graph_output);
    }
    RunMklGRUOp(params, &gru_output);
    ASSERT_EQ(gru_output.dtype(), graph_output.dtype());
    ASSERT_EQ(gru_output.shape(), graph_output.shape());
    test::ExpectClose(gru_output, graph_output, 1e-2);
  }
  void RunGRU(gru_type gtype, int TimeDim) {
    // Some Random batch_size and cell size
    std::vector<std::pair<int,int>> bc= 
        {{3,4}, {12, 16}, {16,16}, {32,32}}; 
    for (int i=0; i<bc.size(); ++i) {
      VerifyGRU(gtype, TimeDim, bc[i].first, bc[i].second);
    }
  }
  void RunGRUTimeDim(gru_type gtype) {
    for (int i=3; i<14; ++i) {
      RunGRU(gtype, i);
    }
  }
  void RunGRUBlockCell() {
    RunGRU(GRUBlockCell, 1);
  }
  void RunAUGRUBlockCell() {
    RunGRU(AUGRUBlockCell, 1);
  }
  void RunMklGRU() {
    RunGRUTimeDim(MklGRU);
  }
  void RunMklAUGRU() {
    RunGRUTimeDim(MklAUGRU);
  }
};

TYPED_TEST_SUITE_P(GRUOpTest);

TYPED_TEST_P(GRUOpTest, GRUBlockCell_tests) {
  this->RunGRUBlockCell();
}
TYPED_TEST_P(GRUOpTest, AUGRUBlockCell_tests) {
  this->RunAUGRUBlockCell();
}
TYPED_TEST_P(GRUOpTest, MklGRU_tests) {
  this->RunMklGRU();
}
TYPED_TEST_P(GRUOpTest, MklAUGRU_tests) {
  this->RunMklAUGRU();
}

REGISTER_TYPED_TEST_SUITE_P(GRUOpTest,  //
                            GRUBlockCell_tests,    //
                            AUGRUBlockCell_tests,  //
                            MklGRU_tests,      //
                            MklAUGRU_tests);

using GRUTypes = ::testing::Types<float, bfloat16>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, GRUOpTest, GRUTypes);

}  // namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
