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
#define EIGEN_USE_THREADS

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/mkl_graph_util.h"
#include "tensorflow/core/kernels/mkl/mkl_kernel_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

class QuantizedFusedInstanceNormTest : public OpsTestBase {
 protected:
  void RunFloatFusedInstanceNorm(const Tensor& input_float,
                                 const Tensor& scale_float,
                                 const Tensor& offset_float,
                                 const Tensor& mean_float,
                                 const Tensor& variance_float,
                                 const bool use_mean_variance,
                                 const string activation_mode,
                                 const float leakyrelu_alpha, Tensor* output) {
    Graph* graph = new Graph(OpRegistry::Global());

    Node* input_node = test::graph::Constant(graph, input_float, "input");
    Node* scale_node = test::graph::Constant(graph, scale_float, "scale");
    Node* offset_node = test::graph::Constant(graph, offset_float, "offset");
    Node* mean_node = test::graph::Constant(graph, mean_float, "mean");
    Node* variance_node =
        test::graph::Constant(graph, variance_float, "variance");

    Node* float_fused_instance_norm_op;
    TF_CHECK_OK(
        NodeBuilder("float_fused_instance_norm", "_MklFusedInstanceNorm")
            .Input(input_node)
            .Input(scale_node)
            .Input(offset_node)
            .Input(mean_node)
            .Input(variance_node)
            .Attr("T", DT_FLOAT)
            .Attr("activation_mode", activation_mode)
            .Attr("leakyrelu_alpha", leakyrelu_alpha)
            .Attr("epsilon", 0.0001)
            .Attr("reduction_axes", {1, 2})  // data_format = NHWC
            .Attr("use_mean_variance", use_mean_variance)
            .Finalize(graph, &float_fused_instance_norm_op));

    GraphDef graph_def;
    graph->ToGraphDef(&graph_def);

    MklTestingUtil::RunGraph(graph_def, "float_fused_instance_norm", output);
  }

  template <typename Tinput, typename Toutput>
  void RunQuantizedFusedInstanceNorm(const Tensor& input_float,
                                     const Tensor& scale_float,
                                     const Tensor& offset_float,
                                     const Tensor& mean_float,
                                     const Tensor& variance_float,
                                     const Tensor& expected_out_float) {
    float input_min, input_max;
    MklTestingUtil::ComputeMinMax<float>(input_float, &input_min, &input_max);
    const float input_max_abs =
        std::max(std::abs(input_min), std::abs(input_max));

    Tensor input_quantized;
    MklTestingUtil::RunMklQuantizeOp(input_float, -input_max_abs, input_max_abs,
                                     DataTypeToEnum<Tinput>::v(), "SCALED",
                                     &input_quantized);

    AddInputFromArray<Tinput>(input_quantized.shape(),
                              input_quantized.flat<Tinput>());
    AddInputFromArray<float>(scale_float.shape(), scale_float.flat<float>());
    AddInputFromArray<float>(offset_float.shape(), offset_float.flat<float>());
    AddInputFromArray<float>(mean_float.shape(), mean_float.flat<float>());
    AddInputFromArray<float>(variance_float.shape(),
                             variance_float.flat<float>());
    AddInputFromArray<float>(TensorShape({}), {-input_max_abs});
    AddInputFromArray<float>(TensorShape({}), {input_max_abs});

    if (std::is_same<Toutput, qint8>::value) {
      float expected_output_min, expected_output_max;
      MklTestingUtil::ComputeMinMax<float>(
          expected_out_float, &expected_output_min, &expected_output_max);
      const float output_max_abs = std::max(std::abs(expected_output_min),
                                            std::abs(expected_output_max));
      AddInputFromArray<float>(TensorShape({}), {-output_max_abs});
      AddInputFromArray<float>(TensorShape({}), {output_max_abs});
    }

    TF_ASSERT_OK(RunOpKernel());

    const Tensor& output = *GetOutput(0);
    if (std::is_same<Toutput, qint8>::value) {
      const Tensor& output_min = *GetOutput(1);
      const Tensor& output_max = *GetOutput(2);
      Tensor output_float;
      MklTestingUtil::RunDequantizeOp(output, output_min, output_max, "SCALED",
                                      &output_float);
      test::ExpectTensorNear<float>(expected_out_float, output_float, 1.1);
    } else {
      test::ExpectTensorNear<float>(expected_out_float, output, 1.1);
    }
  }

  template <typename Tinput, typename Toutput>
  void TestQuantizedFusedInstanceNorm(string activation_mode = "Identity",
                                      float leakyrelu_alpha = 0.0f) {
    DataType input_dt = DataTypeToEnum<Tinput>::v();
    DataType out_type = DataTypeToEnum<Toutput>::v();

    std::vector<DataType> input_types = {
        input_dt,  // x
        DT_FLOAT,  // scale
        DT_FLOAT,  // offset
        DT_FLOAT,  // mean
        DT_FLOAT,  // variance
        DT_FLOAT,  // x_min
        DT_FLOAT   // x_max
    };
    std::vector<DataType> output_types = {out_type};

    if (out_type != DT_FLOAT) {
      input_types.push_back(DT_FLOAT);  // output_min
      input_types.push_back(DT_FLOAT);  // output_max
      output_types.push_back(DT_FLOAT);
      output_types.push_back(DT_FLOAT);
    }

    TF_EXPECT_OK(
        NodeDefBuilder("quantized_fused_IN", "_QuantizedFusedInstanceNorm")
            .Attr("input_types", input_types)
            .Attr("out_types", output_types)
            .Attr("T", input_dt)
            .Attr("U", DT_FLOAT)
            .Attr("Tout", out_type)
            .Attr("epsilon", 0.0001)
            .Attr("activation_mode", activation_mode)
            .Attr("leakyrelu_alpha", leakyrelu_alpha)
            .Attr("reduction_axes", {1, 2})  // data_format = NHWC
            .Input(FakeInput())
            .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());

    const int batch = 1;
    const int height = 2;
    const int width = 2;
    const int channels = 4;

    Tensor input_float(DT_FLOAT, {batch, height, width, channels});
    test::FillValues<float>(&input_float,
                            {2, 3, 4, 1, 2, 3, 2, 3, 2, 2, 1, 2, 2, 3, 2, 3});

    Tensor scale_float(DT_FLOAT, {channels});
    test::FillValues<float>(&scale_float, {2, 3, 1, 3});

    Tensor offset_float(DT_FLOAT, {channels});
    test::FillValues<float>(&offset_float, {-5, -3, -4, -4});

    Tensor mean_float(DT_FLOAT, {channels});
    test::FillValues<float>(&mean_float, {0.1, 0.2, 0.2, 0.3});

    Tensor variance_float(DT_FLOAT, {channels});
    test::FillValues<float>(&variance_float, {1.1, 1.1, 1.2, 1.2});

    constexpr bool use_mean_variance = std::is_same<Toutput, qint8>::value;

    Tensor expected_float;
    RunFloatFusedInstanceNorm(
        input_float, scale_float, offset_float, mean_float, variance_float,
        use_mean_variance, activation_mode, leakyrelu_alpha, &expected_float);

    RunQuantizedFusedInstanceNorm<Tinput, Toutput>(
        input_float, scale_float, offset_float, mean_float, variance_float,
        expected_float);
  }
};

TEST_F(QuantizedFusedInstanceNormTest, QuantizedFusedIN_No_Activation) {
  TestQuantizedFusedInstanceNorm<qint8, qint8>();
}

TEST_F(QuantizedFusedInstanceNormTest, QuantizedFusedIN_With_Relu) {
  TestQuantizedFusedInstanceNorm<qint8, qint8>("Relu");
}

TEST_F(QuantizedFusedInstanceNormTest, QuantizedFusedIN_With_LeakyRelu) {
  TestQuantizedFusedInstanceNorm<qint8, qint8>("LeakyRelu", 0.2);
}

}  // namespace tensorflow

#endif  // INTEL_MKL
