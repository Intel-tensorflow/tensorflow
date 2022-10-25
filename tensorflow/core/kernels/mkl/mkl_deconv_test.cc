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

#if defined(INTEL_MKL)
#define EIGEN_USE_THREADS

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/mkl/mkl_kernel_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

class DeConv2DTest : public OpsTestBase {
 protected:
  void ConfigureDeConv2D(string padding = "SAME", const int& stride = 1) {
    TF_EXPECT_OK(NodeDefBuilder("deconv_op", "_MklNativeConv2DBackpropInput")
                     .Attr("T", DT_FLOAT)
                     .Attr("strides", {1, stride, stride, 1})
                     .Attr("data_format", "NHWC")
                     .Attr("padding", padding)
                     .Attr("dilations", {1, 1, 1, 1})
                     .Attr("_kernel", "MklNameChangeOp")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// This test is commented out until op registration for FP32/BF16 is fixed.
// Currently it fails becasue of Multiple OpKernel registrations for the op
// "_MklNativeConv2DBackpropInput".
// TEST_F(DeConv2DTest, Small) {
//   const int stride = 1;
//   string padding = "SAME";
//   ConfigureDeConv2D(padding, stride);
//   Tensor input_sizes(DT_INT32, {4});
//   test::FillValues<int>(&input_sizes, {2, 6, 4, 2});

//   Tensor image_float(DT_FLOAT, {2, 6, 4, 3});
//   test::FillValues<float>(
//       &image_float,
//       {10,  20,  30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,
//        30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,
//        50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,
//        70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,  70,  80,
//        90,  100, 110, 120, 10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
//        110, 120, 10,  20,  30,  40,  50,  60,  70,  80,  90,  100, 110, 120,
//        10,  20,  30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,
//        30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,
//        50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,
//        70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,  70,  80,
//        90,  100, 110, 120});

//   const int filter_size = 3;
//   const int filter_count = 3;

//   Tensor filter_float(DT_FLOAT, {3, 3, 2, 3});
//   test::FillValues<float>(
//       &filter_float,
//       {10, 40, 70, 20, 50, 80, 30, 60, 90, 10, 40, 70, 20, 50, 80, 30, 60,
//       90,
//        10, 40, 70, 20, 50, 80, 30, 60, 90, 10, 40, 70, 20, 50, 80, 30, 60,
//        90, 10, 40, 70, 20, 50, 80, 30, 60, 90, 10, 40, 70, 20, 50, 80, 30,
//        60, 90});
//   AddInputFromArray<int>(input_sizes.shape(), input_sizes.flat<int>());
//   AddInputFromArray<float>(filter_float.shape(), filter_float.flat<float>());
//   AddInputFromArray<float>(image_float.shape(), image_float.flat<float>());

//   TF_ASSERT_OK(RunOpKernel());
//   const Tensor& output_deconv = *GetOutput(0);

//   Tensor expected_output_float(DT_FLOAT, {2, 6, 4, 2});
//   test::FillValues<float>(
//       &expected_output_float,
//       {21600,  22200,  46800,  46800,  73800,  73800,  66000,  57600,  32400,
//        33300,  70200,  70200,  110700, 110700, 99000,  86400,  32400,  33300,
//        70200,  70200,  110700, 110700, 99000,  86400,  32400,  33300,  70200,
//        70200,  110700, 110700, 99000,  86400,  32400,  33300,  70200,  70200,
//        110700, 110700, 99000,  86400,  21600,  22200,  46800,  46800,  73800,
//        73800,  66000,  57600,  21600,  22200,  46800,  46800,  73800,  73800,
//        66000,  57600,  32400,  33300,  70200,  70200,  110700, 110700, 99000,
//        86400,  32400,  33300,  70200,  70200,  110700, 110700, 99000,  86400,
//        32400,  33300,  70200,  70200,  110700, 110700, 99000,  86400,  32400,
//        33300,  70200,  70200,  110700, 110700, 99000,  86400,  21600,  22200,
//        46800,  46800,  73800,  73800,  66000,  57600});

//   test::ExpectTensorNear<float>(output_deconv, expected_output_float, 1e-6);
// }

class QuantizedDeconvTest : public OpsTestBase {
 protected:
  using NormalRandGen = Eigen::internal::NormalRandomGenerator<float>;
  template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput>
  void RunQuantizedDeconv(const Tensor& input_sizes_data, Tensor& input_float,
                          Tensor& filter_float, Tensor& bias_float,
                          Tensor& expected_out_float,
                          std::vector<string>& fused_ops,
                          const float tol = 1.0) {
    bool fuse_bias = std::find(fused_ops.begin(), fused_ops.end(), "BiasAdd") !=
                     fused_ops.end();
    bool fuse_requantize = std::find(fused_ops.begin(), fused_ops.end(),
                                     "Requantize") != fused_ops.end();
    float input_min, input_max;
    MklTestingUtil::ComputeMinMax<float>(input_float, &input_min, &input_max);
    const float input_max_abs =
        std::max(std::abs(input_min), std::abs(input_max));
    Tensor input_quantized;
    MklTestingUtil::RunMklQuantizeOp(input_float, -input_max_abs, input_max_abs,
                                     DataTypeToEnum<Tinput>::v(), "SCALED",
                                     &input_quantized);

    float filter_min, filter_max;
    MklTestingUtil::ComputeMinMax<float>(filter_float, &filter_min,
                                         &filter_max);
    const float filter_max_abs =
        std::max(std::abs(filter_min), std::abs(filter_max));
    Tensor filter_quantized;
    MklTestingUtil::RunMklQuantizeOp(
        filter_float, -filter_max_abs, filter_max_abs,
        DataTypeToEnum<Tfilter>::v(), "SCALED", &filter_quantized);
    AddInputFromArray<int32>(input_sizes_data.shape(),
                             input_sizes_data.flat<int32>());
    AddInputFromArray<Tfilter>(filter_quantized.shape(),
                               filter_quantized.flat<Tfilter>());
    AddInputFromArray<Tinput>(input_quantized.shape(),
                              input_quantized.flat<Tinput>());
    if (fuse_bias) {
      if (std::is_same<Tbias, float>::value) {
        AddInputFromArray<Tbias>(bias_float.shape(), bias_float.flat<Tbias>());
      } else {
        // Tbias needs to be INT32
        float bias_min, bias_max;
        MklTestingUtil::ComputeMinMax<float>(bias_float, &bias_min, &bias_max);
        const float bias_max_abs =
            std::max(std::abs(bias_min), std::abs(bias_max));
        Tensor bias_quantized;
        MklTestingUtil::RunMklQuantizeOp(
            bias_float, -bias_max_abs, bias_max_abs, DataTypeToEnum<Tbias>::v(),
            "SCALED", &bias_quantized);
        AddInputFromArray<Tbias>(bias_quantized.shape(),
                                 bias_quantized.flat<Tbias>());
      }
    }
    AddInputFromArray<float>(TensorShape({}), {-filter_max_abs});
    AddInputFromArray<float>(TensorShape({}), {filter_max_abs});
    AddInputFromArray<float>(TensorShape({}), {-input_max_abs});
    AddInputFromArray<float>(TensorShape({}), {input_max_abs});

    if (fuse_requantize) {
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
    Tensor output_float = output;
    if (!std::is_same<Toutput, float>::value &&
        !std::is_same<Toutput, bfloat16>::value) {
      const Tensor& output_min = *GetOutput(1);
      const Tensor& output_max = *GetOutput(2);
      MklTestingUtil::RunDequantizeOp(output, output_min, output_max, "SCALED",
                                      &output_float);
    }
    test::ExpectTensorNear<float>(expected_out_float, output_float, tol);
  }

  void RunFloatDeconv(const Tensor& input_sizes_data, const Tensor& input_data,
                      const Tensor& filter_data, const Tensor& bias_data,
                      Tensor* output, const bool is_depthwise,
                      const bool is_deconv3d,
                      const std::vector<string>& fused_ops,
                      const string padding, const int stride) {
    auto root = tensorflow::Scope::NewRootScope();
    auto input_data_op =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));

    auto input_sizes_data_op =
        ops::Const(root.WithOpName("input_sizes"), input_sizes_data);

    Output out_op;
    string last_op = "";
    if (is_depthwise) {
      last_op = "DepthwiseConv2dNativeBackpropInput";
      out_op = ops::DepthwiseConv2dNativeBackpropInput(
          root.WithOpName(last_op), input_sizes_data_op,
          ops::Const(root.WithOpName("filter"),
                     Input::Initializer(filter_data)),
          input_data_op, {1, stride, stride, 1}, padding);
    } else if (is_deconv3d) {
      last_op = "Conv3DBackpropInputV2";
      out_op = ops::Conv3DBackpropInputV2(
          root.WithOpName(last_op), input_sizes_data_op,
          ops::Const(root.WithOpName("filter"),
                     Input::Initializer(filter_data)),
          input_data_op, {1, stride, stride, stride, 1}, padding);
    } else {
      last_op = "Conv2DBackpropInput";
      out_op = ops::Conv2DBackpropInput(
          root.WithOpName(last_op), input_sizes_data_op,
          ops::Const(root.WithOpName("filter"),
                     Input::Initializer(filter_data)),
          input_data_op, {1, stride, stride, 1}, padding);
    }

    for (int i = 0; i < fused_ops.size(); ++i) {
      if (fused_ops[i] == "BiasAdd") {
        last_op = "with_bias";
        out_op = ops::BiasAdd(
            root.WithOpName(last_op), out_op,
            ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));
      }
    }

    tensorflow::GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    MklTestingUtil::RunGraph(graph_def, last_op, output);
  }

  template <typename Tinput, typename Tbias, typename Toutput>
  void RunDeconvTest(std::vector<string> fused_ops, bool is_deconv3d = false,
                     const float tol = 1.0, const float alpha = 0.0) {
    const int stride = 2;
    const string padding = "VALID";
    std::map<string, DataType> data_types = {
        {"Tinput", DataTypeToEnum<Tinput>::v()},
        {"Tfilter", DT_QINT8},
        {"Tbias", DataTypeToEnum<Tbias>::v()},
        {"out_type", DataTypeToEnum<Toutput>::v()}};
    std::vector<DataType> input_types = {DT_INT32, data_types["Tfilter"],
                                         data_types["Tinput"]};
    if (std::find(fused_ops.begin(), fused_ops.end(), "BiasAdd") !=
        fused_ops.end()) {
      input_types.push_back(data_types["Tbias"]);
    }
    input_types.insert(input_types.end(), {DT_FLOAT,    // min_input
                                           DT_FLOAT,    // max_input
                                           DT_FLOAT,    // min_filter
                                           DT_FLOAT});  // max_filter

    if (std::find(fused_ops.begin(), fused_ops.end(), "Requantize") !=
        fused_ops.end()) {
      input_types.push_back(DT_FLOAT);  // min_freezed_output
      input_types.push_back(DT_FLOAT);  // max_freezed_output
    }

    TF_EXPECT_OK(
        NodeDefBuilder("quantized_conv_op", is_deconv3d
                                                ? "_FusedQuantizedDeconv3D"
                                                : "_FusedQuantizedDeconv2D")
            .Attr("Thost_inputs", input_types)
            .Attr("Thost_outputs", {data_types["out_type"], DT_FLOAT, DT_FLOAT})
            .Attr("Tdevice_inputs", std::vector<DataType>())
            .Attr("Tdevice_outputs", std::vector<DataType>())
            .Attr("Tinput", data_types["Tinput"])
            .Attr("Tfilter", data_types["Tfilter"])
            .Attr("Tbias", data_types["Tbias"])
            .Attr("out_type", data_types["out_type"])
            .Attr("strides",
                  is_deconv3d ? std::vector<int>({1, stride, stride, stride, 1})
                              : std::vector<int>({1, stride, stride, 1}))
            .Attr("padding", padding)
            .Attr("fused_ops", fused_ops)
            .Input(FakeInput())
            .Input(FakeInput())
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    const int dims = is_deconv3d ? 5 : 4;
    TensorShape input_shape(is_deconv3d ? TensorShape({2, 5, 6, 4, 3})
                                        : TensorShape({2, 6, 4, 3}));
    TensorShape filter_shape(is_deconv3d ? TensorShape({3, 3, 3, 2, 3})
                                         : TensorShape({3, 3, 2, 3}));

    Tensor input_sizes(DT_INT32, {dims});
    test::FillValues<int32>(
        &input_sizes, is_deconv3d ? gtl::ArraySlice<int32>({2, 11, 13, 9, 2})
                                  : gtl::ArraySlice<int32>({2, 13, 9, 2}));

    Tensor input_float(DT_FLOAT, input_shape);
    input_float.flat<float>().setRandom<NormalRandGen>();

    Tensor filter_float(DT_FLOAT, filter_shape);
    filter_float.flat<float>().setRandom<NormalRandGen>();

    Tensor bias_float(DataTypeToEnum<Tbias>::v(), {2});
    test::FillValues<float>(&bias_float, {2, 1});

    Tensor expected_float;
    RunFloatDeconv(input_sizes, input_float, filter_float, bias_float,
                   &expected_float, /*is_depthwise*/ false, is_deconv3d,
                   fused_ops, padding, stride);
    RunQuantizedDeconv<Tinput, qint8, float, Toutput>(
        input_sizes, input_float, filter_float, bias_float, expected_float,
        fused_ops, tol);
  }
};

TEST_F(QuantizedDeconvTest, RequantizeFusion) {
  RunDeconvTest<qint8, float, qint8>({"Requantize"});
}

TEST_F(QuantizedDeconvTest, DequantizeFusion) {
  RunDeconvTest<qint8, float, float>({"Dequantize"});
}

TEST_F(QuantizedDeconvTest, 3DRequantizeFusion) {
  RunDeconvTest<qint8, float, qint8>({"Requantize"}, true);
}

TEST_F(QuantizedDeconvTest, 3DDequantizeFusion) {
  RunDeconvTest<qint8, float, float>({"Dequantize"}, true);
}

TEST_F(QuantizedDeconvTest, BiasAddRequantizeFusion) {
  RunDeconvTest<qint8, float, qint8>({"BiasAdd", "Requantize"});
}

TEST_F(QuantizedDeconvTest, 3DBiasAddRequantizeFusion) {
  RunDeconvTest<qint8, float, qint8>({"BiasAdd", "Requantize"}, true);
}
}  // namespace tensorflow
#endif  // INTEL_MKL && ENABLE_MKL
