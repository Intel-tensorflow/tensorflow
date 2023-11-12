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

#include <utility>

#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

class LayerNormTest : public HloTestBase {};

TEST_F(LayerNormTest, SimpleTest) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test, entry_computation_layout={(f32[4,1,256]{2,1,0}, f32[1,1,256]{2,1,0}, f32[1,1,256]{2,1,0})->f32[4,1,256]{2,1,0}}

  region_add {
    Arg_0.7555 = f32[] parameter(0)
    Arg_1.7556 = f32[] parameter(1)
    ROOT add.7557 = f32[] add(Arg_0.7555, Arg_1.7556)
  }

  ENTRY main {
    Arg_0.1 = f32[4,1,256]{2,1,0} parameter(0), sharding={replicated}
    Arg_0.2 = f32[1,1,256]{2,1,0} parameter(1), sharding={replicated}
    Arg_0.3 = f32[1,1,256]{2,1,0} parameter(2), sharding={replicated}
    reshape.9744 = f32[1,4,1,256]{3,2,1,0} reshape(Arg_0.1)
    multiply.9743 = f32[4,1,256]{2,1,0} multiply(Arg_0.1, Arg_0.1)
    reshape.9745 = f32[1,4,1,256]{3,2,1,0} reshape(multiply.9743)
    concatenate.9746 = f32[2,4,1,256]{3,2,1,0} concatenate(reshape.9744, reshape.9745), dimensions={0}
    constant.9731 = f32[] constant(0)
    reduce.9747 = f32[2,4,1]{2,1,0} reduce(concatenate.9746, constant.9731), dimensions={3}, to_apply=region_add
    constant.9729 = f32[] constant(256)
    broadcast.9730 = f32[2,4,1]{2,1,0} broadcast(constant.9729), dimensions={}
    divide.9748 = f32[2,4,1]{2,1,0} divide(reduce.9747, broadcast.9730)
    slice.9749 = f32[1,4,1]{2,1,0} slice(divide.9748), slice={[0:1], [0:4], [0:1]}
    reshape.9756 = f32[4,1,1]{2,1,0} reshape(slice.9749)
    broadcast.9758 = f32[4,1,1]{2,1,0} broadcast(reshape.9756), dimensions={0,1,2}
    reshape.9759 = f32[4,1]{1,0} reshape(broadcast.9758)
    broadcast.9760 = f32[4,1,256]{2,1,0} broadcast(reshape.9759), dimensions={0,1}
    subtract.9761 = f32[4,1,256]{2,1,0} subtract(Arg_0.1, broadcast.9760)
    slice.9751 = f32[1,4,1]{2,1,0} slice(divide.9748), slice={[1:2], [0:4], [0:1]}
    reshape.9752 = f32[4,1]{1,0} reshape(slice.9751)
    reshape.9750 = f32[4,1]{1,0} reshape(slice.9749)
    multiply.9753 = f32[4,1]{1,0} multiply(reshape.9750, reshape.9750)
    subtract.9754 = f32[4,1]{1,0} subtract(reshape.9752, multiply.9753)
    constant.9727 = f32[] constant(0)
    broadcast.9728 = f32[4,1]{1,0} broadcast(constant.9727), dimensions={}
    maximum.9755 = f32[4,1]{1,0} maximum(subtract.9754, broadcast.9728)
    reshape.9757 = f32[4,1,1]{2,1,0} reshape(maximum.9755)
    constant.9725 = f32[] constant(1e-05)
    broadcast.9726 = f32[4,1,1]{2,1,0} broadcast(constant.9725), dimensions={}
    add.9762 = f32[4,1,1]{2,1,0} add(reshape.9757, broadcast.9726)
    rsqrt.9763 = f32[4,1,1]{2,1,0} rsqrt(add.9762)
    broadcast.9764 = f32[4,1,1]{2,1,0} broadcast(rsqrt.9763), dimensions={0,1,2}
    reshape.9765 = f32[4,1]{1,0} reshape(broadcast.9764)
    broadcast.9766 = f32[4,1,256]{2,1,0} broadcast(reshape.9765), dimensions={0,1}
    broadcast.9767 = f32[1,1,256]{2,1,0} broadcast(Arg_0.2), dimensions={0,1,2}
    reshape.9768 = f32[1,256]{1,0} reshape(broadcast.9767)
    broadcast.9769 = f32[4,1,256]{2,1,0} broadcast(reshape.9768), dimensions={1,2}
    multiply.9770 = f32[4,1,256]{2,1,0} multiply(broadcast.9766, broadcast.9769)
    multiply.9771 = f32[4,1,256]{2,1,0} multiply(subtract.9761, multiply.9770)
    broadcast.9772 = f32[1,1,256]{2,1,0} broadcast(Arg_0.3), dimensions={0,1,2}
    reshape.9773 = f32[1,256]{1,0} reshape(broadcast.9772)
    broadcast.9774 = f32[4,1,256]{2,1,0} broadcast(reshape.9773), dimensions={1,2}
    ROOT add.9775 = f32[4,1,256]{2,1,0} add(multiply.9771, broadcast.9774)
  }
)";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(LayerNormTest, SimpleTestBF16) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm_bf16.test, entry_computation_layout={(f32[768]{0}, f32[768]{0}, bf16[16,128,768]{2,1,0})->bf16[16,128,768]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

  region_0.16 {
    Arg_0.17 = f32[] parameter(0)
    Arg_1.18 = f32[] parameter(1)
    ROOT add.19 = f32[] add(Arg_0.17, Arg_1.18), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/reduce_sum[axes=(3,)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=92}
  }

  ENTRY main.53 {
    Arg_2.3 = bf16[16,128,768]{2,1,0} parameter(2), sharding={replicated}
    convert.31 = f32[16,128,768]{2,1,0} convert(Arg_2.3), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=160}
    convert.11 = f32[16,128,768]{2,1,0} convert(Arg_2.3), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=89}
    reshape.13 = f32[1,16,128,768]{3,2,1,0} reshape(convert.11), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/broadcast_in_dim[shape=(1, 16, 128, 768) broadcast_dimensions=(1, 2, 3)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=100}
    multiply.12 = f32[16,128,768]{2,1,0} multiply(convert.11, convert.11), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=44}
    reshape.14 = f32[1,16,128,768]{3,2,1,0} reshape(multiply.12), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/broadcast_in_dim[shape=(1, 16, 128, 768) broadcast_dimensions=(1, 2, 3)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=100}
    concatenate.15 = f32[2,16,128,768]{3,2,1,0} concatenate(reshape.13, reshape.14), dimensions={0}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/concatenate[dimension=0]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=100}
    constant.10 = f32[] constant(0)
    reduce.20 = f32[2,16,128]{2,1,0} reduce(concatenate.15, constant.10), dimensions={3}, to_apply=region_0.16, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/reduce_sum[axes=(3,)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=92}
    constant.8 = f32[] constant(768)
    broadcast.9 = f32[2,16,128]{2,1,0} broadcast(constant.8), dimensions={}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/div" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=92}
    divide.21 = f32[2,16,128]{2,1,0} divide(reduce.20, broadcast.9), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/div" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=92}
    slice.22 = f32[1,16,128]{2,1,0} slice(divide.21), slice={[0:1], [0:16], [0:128]}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/slice[start_indices=(0, 0, 0) limit_indices=(1, 16, 128) strides=(1, 1, 1)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=100}
    reshape.29 = f32[16,128,1]{2,1,0} reshape(slice.22), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/broadcast_in_dim[shape=(16, 128, 1) broadcast_dimensions=(0, 1)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=158}
    broadcast.32 = f32[16,128,1]{2,1,0} broadcast(reshape.29), dimensions={0,1,2}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/sub" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=160}
    reshape.33 = f32[16,128]{1,0} reshape(broadcast.32), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/sub" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=160}
    broadcast.34 = f32[16,128,768]{2,1,0} broadcast(reshape.33), dimensions={0,1}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/sub" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=160}
    subtract.35 = f32[16,128,768]{2,1,0} subtract(convert.31, broadcast.34), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/sub" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=160}
    slice.24 = f32[1,16,128]{2,1,0} slice(divide.21), slice={[1:2], [0:16], [0:128]}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/slice[start_indices=(1, 0, 0) limit_indices=(2, 16, 128) strides=(1, 1, 1)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=100}
    reshape.25 = f32[16,128]{1,0} reshape(slice.24), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/squeeze[dimensions=(0,)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=100}
    reshape.23 = f32[16,128]{1,0} reshape(slice.22), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/squeeze[dimensions=(0,)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=100}
    multiply.26 = f32[16,128]{1,0} multiply(reshape.23, reshape.23), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=44}
    subtract.27 = f32[16,128]{1,0} subtract(reshape.25, multiply.26), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/sub" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=103}
    constant.6 = f32[] constant(0)
    broadcast.7 = f32[16,128]{1,0} broadcast(constant.6), dimensions={}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/max" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=103}
    maximum.28 = f32[16,128]{1,0} maximum(subtract.27, broadcast.7), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/max" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=103}
    reshape.30 = f32[16,128,1]{2,1,0} reshape(maximum.28), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/broadcast_in_dim[shape=(16, 128, 1) broadcast_dimensions=(0, 1)]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=159}
    constant.4 = f32[] constant(1e-06)
    broadcast.5 = f32[16,128,1]{2,1,0} broadcast(constant.4), dimensions={}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/add" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=161}
    add.36 = f32[16,128,1]{2,1,0} add(reshape.30, broadcast.5), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/add" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=161}
    rsqrt.37 = f32[16,128,1]{2,1,0} rsqrt(add.36), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/rsqrt" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=161}
    broadcast.39 = f32[16,128,1]{2,1,0} broadcast(rsqrt.37), dimensions={0,1,2}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=167}
    reshape.40 = f32[16,128]{1,0} reshape(broadcast.39), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=167}
    broadcast.41 = f32[16,128,768]{2,1,0} broadcast(reshape.40), dimensions={0,1}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=167}
    Arg_1.2 = f32[768]{0} parameter(1), sharding={replicated}
    reshape.38 = f32[1,1,768]{2,1,0} reshape(Arg_1.2), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/reshape[new_sizes=(1, 1, 768) dimensions=None]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=166}
    broadcast.42 = f32[1,1,768]{2,1,0} broadcast(reshape.38), dimensions={0,1,2}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=167}
    reshape.43 = f32[768]{0} reshape(broadcast.42), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=167}
    broadcast.44 = f32[16,128,768]{2,1,0} broadcast(reshape.43), dimensions={2}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=167}
    multiply.45 = f32[16,128,768]{2,1,0} multiply(broadcast.41, broadcast.44), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=167}
    multiply.46 = f32[16,128,768]{2,1,0} multiply(subtract.35, multiply.45), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/mul" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=169}
    Arg_0.1 = f32[768]{0} parameter(0), sharding={replicated}
    reshape.47 = f32[1,1,768]{2,1,0} reshape(Arg_0.1), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/reshape[new_sizes=(1, 1, 768) dimensions=None]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=173}
    broadcast.48 = f32[1,1,768]{2,1,0} broadcast(reshape.47), dimensions={0,1,2}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/add" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=174}
    reshape.49 = f32[768]{0} reshape(broadcast.48), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/add" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=174}
    broadcast.50 = f32[16,128,768]{2,1,0} broadcast(reshape.49), dimensions={2}, metadata={op_name="jit(apply)/jit(main)/LayerNormOp/add" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=174}
    add.51 = f32[16,128,768]{2,1,0} add(multiply.46, broadcast.50), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/add" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=174}
    ROOT convert.52 = bf16[16,128,768]{2,1,0} convert(add.51), metadata={op_name="jit(apply)/jit(main)/LayerNormOp/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/home/mabuzain/.local/lib/python3.10/site-packages/flax/linen/normalization.py" source_line=177}
  }
)";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
}

}  // namespace
}  // namespace xla
