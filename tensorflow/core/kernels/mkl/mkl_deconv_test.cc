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

TEST_F(DeConv2DTest, Small) {
  const int stride = 1;
  string padding = "SAME";
  ConfigureDeConv2D(padding, stride);
  Tensor input_sizes(DT_INT32, {4});
  test::FillValues<int>(&input_sizes, {2, 6, 4, 2});

  Tensor image_float(DT_FLOAT, {2, 6, 4, 3});
  test::FillValues<float>(
      &image_float,
      {10,  20,  30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,
       30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,
       50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,
       70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,  70,  80,
       90,  100, 110, 120, 10,  20,  30,  40,  50,  60,  70,  80,  90,  100,
       110, 120, 10,  20,  30,  40,  50,  60,  70,  80,  90,  100, 110, 120,
       10,  20,  30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,
       30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,
       50,  60,  70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,
       70,  80,  90,  100, 110, 120, 10,  20,  30,  40,  50,  60,  70,  80,
       90,  100, 110, 120});

  const int filter_size = 3;
  const int filter_count = 3;

  Tensor filter_float(DT_FLOAT, {3, 3, 2, 3});
  test::FillValues<float>(
      &filter_float,
      {10, 40, 70, 20, 50, 80, 30, 60, 90, 10, 40, 70, 20, 50, 80, 30, 60, 90,
       10, 40, 70, 20, 50, 80, 30, 60, 90, 10, 40, 70, 20, 50, 80, 30, 60, 90,
       10, 40, 70, 20, 50, 80, 30, 60, 90, 10, 40, 70, 20, 50, 80, 30, 60, 90});
  AddInputFromArray<int>(input_sizes.shape(), input_sizes.flat<int>());
  AddInputFromArray<float>(filter_float.shape(), filter_float.flat<float>());
  AddInputFromArray<float>(image_float.shape(), image_float.flat<float>());

  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_deconv = *GetOutput(0);

  Tensor expected_output_float(DT_FLOAT, {2, 6, 4, 2});
  test::FillValues<float>(
      &expected_output_float,
      {21600,  22200,  46800,  46800,  73800,  73800,  66000,  57600,  32400,
       33300,  70200,  70200,  110700, 110700, 99000,  86400,  32400,  33300,
       70200,  70200,  110700, 110700, 99000,  86400,  32400,  33300,  70200,
       70200,  110700, 110700, 99000,  86400,  32400,  33300,  70200,  70200,
       110700, 110700, 99000,  86400,  21600,  22200,  46800,  46800,  73800,
       73800,  66000,  57600,  21600,  22200,  46800,  46800,  73800,  73800,
       66000,  57600,  32400,  33300,  70200,  70200,  110700, 110700, 99000,
       86400,  32400,  33300,  70200,  70200,  110700, 110700, 99000,  86400,
       32400,  33300,  70200,  70200,  110700, 110700, 99000,  86400,  32400,
       33300,  70200,  70200,  110700, 110700, 99000,  86400,  21600,  22200,
       46800,  46800,  73800,  73800,  66000,  57600});

  test::ExpectTensorNear<float>(output_deconv, expected_output_float, 1e-6);
}

}  // namespace tensorflow
#endif  // INTEL_MKL
