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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <utility>

#include "tsl/platform/cpu_info.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace cpu {

class MatmulTest : public HloTestBase {};

TEST_F(MatmulTest, SimpleTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[2,8,4,16]{3,2,1,0},f32[2,8,16,32]{3,2,1,0})->f32[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.f32 {
    arg.0 = f32[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestBF16) {
  // TODO(penporn): Refactor IsBF16SupportedByOneDNNOnThisCPU() from
  // tensorflow/core/graph/mkl_graph_util.h and call the function instead.
  using tsl::port::TestCPUFeature;
  if (!TestCPUFeature(tsl::port::CPUFeature::AVX512_BF16) &&
      !TestCPUFeature(tsl::port::CPUFeature::AMX_BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule matmul.test.bf16, entry_computation_layout={(bf16[2,8,4,16]{3,2,1,0},bf16[2,8,16,32]{3,2,1,0})->bf16[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.bf16 {
    arg.0 = bf16[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = bf16[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = bf16[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32TransposeB) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.1, entry_computation_layout={(f32[2,8,4,16]{3,1,2,0},f32[2,8,4,16]{3,1,2,0})->f32[2,8,4,4]{3,2,1,0}}

  ENTRY matmul.test.1 {
    arg.0 = f32[2,8,4,16]{3,1,2,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,4,16]{3,1,2,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[2,8,4,4]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAddFusion1) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[2,2,40,30]{3,2,1,0})->f32[2,2,40,40]{3,2,1,0}}

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[2,2,40,30]{3,2,1,0} parameter(0), parameter_replication={false}
    reshape.2 = f32[2,2,40,30]{3,2,1,0} reshape(arg0.1)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[2,2,30,40]{3,2,1,0} broadcast(constant.3), dimensions={}
    dot.7 = f32[2,2,40,40]{3,2,1,0} dot(reshape.2, broadcast.4), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[40]{0} broadcast(constant.5), dimensions={}
    broadcast.9 = f32[2,2,40,40]{3,2,1,0} broadcast(broadcast.6), dimensions={3}
    add.10 = f32[2,2,40,40]{3,2,1,0} add(dot.7, broadcast.9)
    reshape.11 = f32[2,2,40,40]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[2,2,40,40]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[2,2,40,40]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAddFusion2) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[40,30]{1,0})->f32[40,1,40]{2,1,0}}

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[40,30]{1,0} parameter(0), parameter_replication={false}
    reshape.2 = f32[40,30]{1,0} reshape(arg0.1)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[30,40]{1,0} broadcast(constant.3), dimensions={}
    dot.7 = f32[40,40]{1,0} dot(reshape.2, broadcast.4), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0}
    reshape.1 = f32[40,1,40]{2,1,0} reshape(dot.7)
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[40]{0} broadcast(constant.5), dimensions={}
    broadcast.9 = f32[40,1,40]{2,1,0} broadcast(broadcast.6), dimensions={2}
    add.10 = f32[40,1,40]{2,1,0} add(reshape.1, broadcast.9)
    tuple.12 = (f32[40,1,40]{2,1,0}) tuple(add.10)
    ROOT get-tuple-element.13 = f32[40,1,40]{2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter1) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[2,2,40,30]{3,2,1,0}, f32[2,2,30,40]{3,2,1,0}, f32[2,2,40,40]{3,2,1,0})->f32[2,2,40,40]{3,2,1,0}}

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[2,2,40,30]{3,2,1,0} parameter(0), parameter_replication={false}
    arg0.2 = f32[2,2,30,40]{3,2,1,0} parameter(1), parameter_replication={false}
    arg0.3 = f32[2,2,40,40]{3,2,1,0} parameter(2), parameter_replication={false}
    dot.7 = f32[2,2,40,40]{3,2,1,0} dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    add.10 = f32[2,2,40,40]{3,2,1,0} add(dot.7, arg0.3)
    reshape.11 = f32[2,2,40,40]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[2,2,40,40]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[2,2,40,40]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter2) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[2,2,40,30]{3,2,1,0}, f32[2,2,30,40]{3,2,1,0}, f32[40]{0})->f32[2,2,40,40]{3,2,1,0}}

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[2,2,40,30]{3,2,1,0} parameter(0), parameter_replication={false}
    arg0.2 = f32[2,2,30,40]{3,2,1,0} parameter(1), parameter_replication={false}
    arg0.3 = f32[40]{0} parameter(2), parameter_replication={false}
    dot.7 = f32[2,2,40,40]{3,2,1,0} dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    broad.1 = f32[2,2,40,40]{3,2,1,0} broadcast(arg0.3), dimensions={3}
    add.10 = f32[2,2,40,40]{3,2,1,0} add(dot.7, broad.1)
    reshape.11 = f32[2,2,40,40]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[2,2,40,40]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[2,2,40,40]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter3) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[16,128,768]{2,1,0}, f32[768,768]{1,0}, f32[768]{0})->f32[16,128,768]{2,1,0}}

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[16,128,768]{2,1,0} parameter(0), sharding={replicated}
    arg0.2 = f32[768,768]{1,0} parameter(1), sharding={replicated}
    dot.84 = f32[16,128,768]{2,1,0} dot(arg0.1, arg0.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    arg0.3 = f32[768]{0} parameter(2), sharding={replicated}
    reshape.85 = f32[1,1,768]{2,1,0} reshape(arg0.3)
    broadcast.86 = f32[1,1,768]{2,1,0} broadcast(reshape.85), dimensions={0,1,2}
    reshape.87 = f32[768]{0} reshape(broadcast.86)
    broadcast.88 = f32[16,128,768]{2,1,0} broadcast(reshape.87), dimensions={2}
    ROOT add.89 = f32[16,128,768]{2,1,0} add(dot.84, broadcast.88)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32TransposeBWithBiasAddFusion) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.1, entry_computation_layout={(f32[2,8,4,16]{3,1,2,0},f32[2,8,4,16]{3,1,2,0})->f32[2,8,4,4]{3,2,1,0}}

  ENTRY matmul.test.1 {
    arg.0 = f32[2,8,4,16]{3,1,2,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,4,16]{3,1,2,0} parameter(1), parameter_replication={false}
    dot.7 = f32[2,8,4,4]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[4]{0} broadcast(constant.5), dimensions={}
    broadcast.9 = f32[2,8,4,4]{3,2,1,0} broadcast(broadcast.6), dimensions={3}
    add.10 = f32[2,8,4,4]{3,2,1,0} add(dot.7, broadcast.9)
    reshape.11 = f32[2,8,4,4]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[2,8,4,4]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[2,8,4,4]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, ApproxGELUTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[2,8,4,16]{3,2,1,0},f32[2,8,16,32]{3,2,1,0})->f32[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.f32 {
    arg.0 = f32[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    onednn.matmul.0 = f32[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    mul.0 = f32[2,8,4,32]{3,2,1,0} multiply(onednn.matmul.0, onednn.matmul.0)
    mul.1 = f32[2,8,4,32]{3,2,1,0} multiply(onednn.matmul.0, mul.0)
    const.0 = f32[] constant(0.044715)
    bcast.0 = f32[2,8,4,32]{3,2,1,0} broadcast(const.0), dimensions={}
    mul.2 = f32[2,8,4,32]{3,2,1,0} multiply(mul.1, bcast.0)
    add.0 = f32[2,8,4,32]{3,2,1,0} add(onednn.matmul.0, mul.2)
    const.1 = f32[] constant(0.797884583)
    bcast.1 = f32[2,8,4,32]{3,2,1,0} broadcast(const.1), dimensions={}
    mul.3 = f32[2,8,4,32]{3,2,1,0} multiply(add.0, bcast.1)
    tanh = f32[2,8,4,32]{3,2,1,0} tanh(mul.3)
    const.2 = f32[] constant(1)
    bcast.2 = f32[2,8,4,32]{3,2,1,0} broadcast(const.2), dimensions={}
    add.2 = f32[2,8,4,32]{3,2,1,0} add(tanh, bcast.2)
    const.3 = f32[] constant(0.5)
    bcast.3 = f32[2,8,4,32]{3,2,1,0} broadcast(const.3), dimensions={}
    mul.4 = f32[2,8,4,32]{3,2,1,0} multiply(add.2, bcast.3)
    ROOT out = f32[2,8,4,32]{3,2,1,0} multiply(onednn.matmul.0, mul.4)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["GELU_TANH"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

// GPTJ BIAS+GELU pattern with reduced sizes for test time:
// batch=2; seq_len=16; hidden_size=64; intermediate_size=256
TEST_F(MatmulTest, BiasAndApproxGELUTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[2,16,64]{2,1,0}, f32[64,256]{1,0}, f32[256]{0})->f32[2,16,256]{2,1,0}}

  ENTRY matmul.test.f32 {
  Arg_5.6 = f32[2,16,64]{2,1,0} parameter(0), sharding={replicated}
  Arg_7.8 = f32[64,256]{1,0} parameter(1), sharding={replicated}
  dot.232 = f32[2,16,256]{2,1,0} dot(Arg_5.6, Arg_7.8), lhs_contracting_dims={2}, rhs_contracting_dims={0}, metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/fc_in/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/flax/linen/linear.py" source_line=206}
  Arg_6.7 = f32[256]{0} parameter(2), sharding={replicated}
  reshape.233 = f32[1,1,256]{2,1,0} reshape(Arg_6.7), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/fc_in/reshape[new_sizes=(1, 1, 16384) dimensions=None]" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/flax/linen/linear.py" source_line=213}
  broadcast.234 = f32[1,1,256]{2,1,0} broadcast(reshape.233), dimensions={0,1,2}, metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/fc_in/add" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/flax/linen/linear.py" source_line=213}
  reshape.235 = f32[256]{0} reshape(broadcast.234), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/fc_in/add" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/flax/linen/linear.py" source_line=213}
  broadcast.236 = f32[2,16,256]{2,1,0} broadcast(reshape.235), dimensions={2}, metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/fc_in/add" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/flax/linen/linear.py" source_line=213}
  add.237 = f32[2,16,256]{2,1,0} add(dot.232, broadcast.236), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/fc_in/add" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/flax/linen/linear.py" source_line=213}
  multiply.238 = f32[2,16,256]{2,1,0} multiply(add.237, add.237), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  multiply.239 = f32[2,16,256]{2,1,0} multiply(add.237, multiply.238), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  constant.20 = f32[] constant(0.044715)
  broadcast.21 = f32[2,16,256]{2,1,0} broadcast(constant.20), dimensions={}, metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  multiply.240 = f32[2,16,256]{2,1,0} multiply(multiply.239, broadcast.21), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  add.241 = f32[2,16,256]{2,1,0} add(add.237, multiply.240), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/add" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  constant.18 = f32[] constant(0.797884583)
  broadcast.19 = f32[2,16,256]{2,1,0} broadcast(constant.18), dimensions={}, metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  multiply.242 = f32[2,16,256]{2,1,0} multiply(add.241, broadcast.19), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  tanh.243 = f32[2,16,256]{2,1,0} tanh(multiply.242), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/tanh" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  constant.16 = f32[] constant(1)
  broadcast.17 = f32[2,16,256]{2,1,0} broadcast(constant.16), dimensions={}, metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/add" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  add.244 = f32[2,16,256]{2,1,0} add(tanh.243, broadcast.17), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/add" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  constant.14 = f32[] constant(0.5)
  broadcast.15 = f32[2,16,256]{2,1,0} broadcast(constant.14), dimensions={}, metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  multiply.245 = f32[2,16,256]{2,1,0} multiply(add.244, broadcast.15), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  ROOT out = f32[2,16,256]{2,1,0} multiply(add.237, multiply.245), metadata={op_name="jit(apply)/jit(main)/GPTJLayer/mlp/mul" source_file="/home/akaczynsk/.venvs/jax-onednn/lib/python3.10/site-packages/transformers/models/gptj/modeling_flax_gptj.py" source_line=312}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["BIAS","GELU_TANH"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

// FIXME(intel-tf): Resolve covert instructions bf16->fp32->bf16
/*
TEST_F(MatmulTest, ApproxGELUTestBF16) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.bf16,
entry_computation_layout={(bf16[2,8,4,16]{3,2,1,0},bf16[2,8,16,32]{3,2,1,0})->bf16[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.bf16 {
    arg.0 = bf16[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = bf16[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    onednn.matmul.0 = bf16[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1),
lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1},
rhs_contracting_dims={2} mul.0 = bf16[2,8,4,32]{3,2,1,0}
multiply(onednn.matmul.0, onednn.matmul.0) mul.1 = bf16[2,8,4,32]{3,2,1,0}
multiply(onednn.matmul.0, mul.0) const.0 = bf16[] constant(0.044715) bcast.0 =
bf16[2,8,4,32]{3,2,1,0} broadcast(const.0), dimensions={} mul.2 =
bf16[2,8,4,32]{3,2,1,0} multiply(mul.1, bcast.0) add.0 = bf16[2,8,4,32]{3,2,1,0}
add(onednn.matmul.0, mul.2) const.1 = bf16[] constant(0.797884583) bcast.1 =
bf16[2,8,4,32]{3,2,1,0} broadcast(const.1), dimensions={} mul.3 =
bf16[2,8,4,32]{3,2,1,0} multiply(add.0, bcast.1) tanh = bf16[2,8,4,32]{3,2,1,0}
tanh(mul.3) const.2 = bf16[] constant(1) bcast.2 = bf16[2,8,4,32]{3,2,1,0}
broadcast(const.2), dimensions={} add.2 = bf16[2,8,4,32]{3,2,1,0} add(tanh,
bcast.2) const.3 = bf16[] constant(0.5) bcast.3 = bf16[2,8,4,32]{3,2,1,0}
broadcast(const.3), dimensions={} mul.4 = bf16[2,8,4,32]{3,2,1,0}
multiply(add.2, bcast.3) ROOT out = bf16[2,8,4,32]{3,2,1,0}
multiply(onednn.matmul.0, mul.4)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{5e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["GELU_TANH"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}
*/

TEST_F(MatmulTest, ReLUTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[2,8,4,16]{3,2,1,0},f32[2,8,16,32]{3,2,1,0})->f32[2,8,4,32]{3,2,1,0}}

  relu.1 {
    Arg_0.3 = f32[2,8,4,32]{3,2,1,0} parameter(0)
    constant.4 = f32[] constant(0)
    broadcast.5 = f32[2,8,4,32]{3,2,1,0} broadcast(constant.4), dimensions={}
    ROOT maximum.6 = f32[2,8,4,32]{3,2,1,0} maximum(Arg_0.3, broadcast.5)
  }

  ENTRY matmul.test.f32 {
    arg.0 = f32[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    onednn.matmul.0 = f32[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    ROOT call.7 = f32[2,8,4,32]{3,2,1,0} call(onednn.matmul.0), to_apply=relu.1
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["RELU"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

TEST_F(MatmulTest, DivisionByConstantWithEltwiseLinearF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.divide.test.1, entry_computation_layout={(f32[16,128,768]{2,1,0}, f32[768,12,64]{2,1,0})->f32[16,128,12,64]{3,2,1,0}}

  ENTRY matmul.divide.test.f32 {
    Arg_4.5 = f32[16,128,768]{2,1,0} parameter(0), sharding={replicated}
    Arg_2.3 = f32[768,12,64]{2,1,0} parameter(1), sharding={replicated}
    onednn.matmul.0 = f32[16,128,12,64]{3,2,1,0} dot(Arg_4.5, Arg_2.3), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    constant.8 = f32[] constant(8)
    broadcast.9 = f32[16,128,12,64]{3,2,1,0} broadcast(constant.8), dimensions={}
    ROOT divide.16 = f32[16,128,12,64]{3,2,1,0} divide(onednn.matmul.0, broadcast.9)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec(1e-4, 1e-4)));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["LINEAR"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}
TEST_F(MatmulTest, TestF32NonConstantWeights) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[8,4,16]{2,1,0},f32[16,32]{1,0})->f32[8,4,32]{2,1,0}}

  ENTRY matmul.test.f32 {
    arg.0 = f32[8,4,16]{2,1,0} parameter(0), parameter_replication={false}
    arg.1 = f32[16,32]{1,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[8,4,32]{2,1,0} dot(arg.0, arg.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     %matmul.test.f32
  ; CHECK-NOT: custom_call_target="__onednn$matmul_reorder",
  ; CHECK:     custom-call(%{{[a-z,A-Z,0-9,\.]*}}, %arg.1), custom_call_target="__onednn$matmul",
  )");
}

TEST_F(MatmulTest, TestF32ConstantWeights) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[8,4,16]{2,1,0})->f32[8,4,32]{2,1,0}}

  ENTRY matmul.test.f32 {
    arg.0 = f32[8,4,16]{2,1,0} parameter(0), parameter_replication={false}
    constant = f32[] constant(1)
    arg.1 = f32[16,32]{1,0} broadcast(constant), dimensions={}
    ROOT onednn.matmul.0 = f32[8,4,32]{2,1,0} dot(arg.0, arg.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     %matmul.test.f32
  ; CHECK-NOT: custom_call_target="__onednn$matmul_reorder",
  ; CHECK:     custom-call(%{{[a-z,A-Z,0-9,\.]*}}, %constant{{[a-z,A-Z,0-9,\.]*}}), custom_call_target="__onednn$matmul",
  )");
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
