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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_CONV_OPS_H_

#ifdef INTEL_MKL
#include <limits>
#include <memory>
#include <vector>

#include "dnnl.hpp"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/onednn_env_vars.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

using dnnl::deconvolution_forward;
using dnnl::prop_kind;
using dnnl::stream;

namespace tensorflow {
class DnnDeconvUtil {
 protected:
  OpKernelContext* context_;
  std::vector<int32> strides_;
  std::vector<int32> dilations_;
  Padding padding_;
  TensorFormat data_format_;

 public:
  DnnDeconvUtil(OpKernelContext* context, const std::vector<int32>& strides,
                Padding pad, TensorFormat fm,
                const std::vector<int32>& dilations)
      : context_(context),
        strides_(strides),
        dilations_(dilations),
        padding_(pad),
        data_format_(fm) {}

  virtual ~DnnDeconvUtil() { context_ = nullptr; }

  // Calculate Deconvolution strides
  virtual inline void GetStrides(memory::dims* strides) {
    // Take the stride from the second and third dimensions only
    // Striding is not supported on the batch or depth dimension.
    DCHECK(strides);
    if (strides_.size() == 4) {
      int stride_rows = GetTensorDim(strides_, data_format_, 'H');
      int stride_cols = GetTensorDim(strides_, data_format_, 'W');
      *strides = {stride_rows, stride_cols};
    } else if (strides_.size() == 5) {
      int stride_planes = GetTensorDim(strides_, data_format_, '0');
      int stride_rows = GetTensorDim(strides_, data_format_, '1');
      int stride_cols = GetTensorDim(strides_, data_format_, '2');
      *strides = {stride_planes, stride_rows, stride_cols};
    }
  }

  // Calculate Deconvolution dilations
  virtual inline void GetDilations(memory::dims* dilations) {
    // Take the dilation from the second and third dimensions only.
    // Dilation is not supported on the batch or depth dimension.
    DCHECK(dilations);
    if (dilations_.size() == 4) {
      int dilations_rows = GetTensorDim(dilations_, data_format_, 'H');
      int dilations_cols = GetTensorDim(dilations_, data_format_, 'W');
      *dilations = {dilations_rows, dilations_cols};
    } else if (dilations_.size() == 5) {
      int dilations_planes = GetTensorDim(dilations_, data_format_, '0');
      int dilations_rows = GetTensorDim(dilations_, data_format_, '1');
      int dilations_cols = GetTensorDim(dilations_, data_format_, '2');
      *dilations = {dilations_planes, dilations_rows, dilations_cols};
    }
  }

  // Calculate Deconvolution input size in oneDNN order. oneDNN
  // requires input in NCHW format. Function does not return anything.
  // But errors arising from sanity checks are returned in context's status.
  virtual inline void GetInputSize(const TensorShape& input_shape,
                                   memory::dims* input_dims) {
#define CHECK_BOUNDS(val, err_msg)                                     \
  do {                                                                 \
    OP_REQUIRES(context_,                                              \
                FastBoundsCheck(val, std::numeric_limits<int>::max()), \
                errors::InvalidArgument(err_msg));                     \
  } while (0)

    DCHECK(input_dims);

    // Input channel
    int64 input_depth_raw = GetTensorDim(input_shape, data_format_, 'C');
    int input_depth = static_cast<int>(input_depth_raw);

    // Input batch
    int64 input_batch_raw = GetTensorDim(input_shape, data_format_, 'N');
    CHECK_BOUNDS(input_batch_raw, "Input batch too large");
    int input_batch = static_cast<int>(input_batch_raw);

    if (strides_.size() == 4) {  // NCHW format for Deconv2D
      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, 'H');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, 'W');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // oneDNN always requires input in NCHW format Deconv2D.
      std::vector<memory::dim> input_sizes(4, -1);
      input_sizes[MklDnnDims::Dim_N] = input_batch;
      input_sizes[MklDnnDims::Dim_C] = input_depth;
      input_sizes[MklDnnDims::Dim_H] = input_rows;
      input_sizes[MklDnnDims::Dim_W] = input_cols;
      *input_dims = input_sizes;
    } else if (strides_.size() == 5) {  // NCDHW format for Deconv3D
      // Input planes/third-dimension
      int64 input_planes_raw = GetTensorDim(input_shape, data_format_, '0');
      CHECK_BOUNDS(input_planes_raw, "Input depth too large");
      int input_planes = static_cast<int>(input_planes_raw);

      // Input rows/height
      int64 input_rows_raw = GetTensorDim(input_shape, data_format_, '1');
      CHECK_BOUNDS(input_rows_raw, "Input rows too large");
      int input_rows = static_cast<int>(input_rows_raw);

      // Input columns/width
      int64 input_cols_raw = GetTensorDim(input_shape, data_format_, '2');
      CHECK_BOUNDS(input_cols_raw, "Input cols too large");
      int input_cols = static_cast<int>(input_cols_raw);

      // oneDNN always requires input in NCDHW format for Deconv3D.
      std::vector<memory::dim> input_sizes(5, -1);
      input_sizes[MklDnnDims3D::Dim3d_N] = input_batch;
      input_sizes[MklDnnDims3D::Dim3d_C] = input_depth;
      input_sizes[MklDnnDims3D::Dim3d_D] = input_planes;
      input_sizes[MklDnnDims3D::Dim3d_H] = input_rows;
      input_sizes[MklDnnDims3D::Dim3d_W] = input_cols;
      *input_dims = input_sizes;
    }
#undef CHECK_BOUNDS
  }

  // Calculate Deconvolution filter size in oneDNN order.
  // oneDNN requires filter in OIHW (Deconv2D) format.
  // Function does not return anything.
  // But errors arising from sanity checks are returned in context's status.
  virtual inline void GetFilterSize(const TensorShape& input_shape,
                                    const TensorShape& filter_shape,
                                    memory::dims* filter_dims) {
    DCHECK(filter_dims);

    OP_REQUIRES(context_, filter_shape.dims() == strides_.size(),
                errors::InvalidArgument((strides_.size() == 4)
                                            ? "filter must be 4-dimensional: "
                                            : "filter must be 5-dimensional: ",
                                        filter_shape.DebugString()));

    for (int i = 0; i < ((strides_.size() == 4) ? 3 : 5); i++) {
      OP_REQUIRES(context_, FastBoundsCheck(filter_shape.dim_size(i),
                                            std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    int input_depth = GetTensorDim(input_shape, data_format_, 'C');

    if (strides_.size() == 4) {  // Deconv2D
      // TF filter is always in (rows, cols, in_depth, out_depth) order.
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_O));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_2DFILTER_DIM_I));

      // oneDNN always needs filter in OIHW format for regular convolutions
      // OIHW = (out_depth, in_depth, rows, cols)
      std::vector<memory::dim> filter_sizes(4, -1);
      filter_sizes[MklDnnDims::Dim_O] = filter_out_depth;
      filter_sizes[MklDnnDims::Dim_I] = filter_in_depth;
      filter_sizes[MklDnnDims::Dim_H] = filter_rows;
      filter_sizes[MklDnnDims::Dim_W] = filter_cols;
      *filter_dims = filter_sizes;
    } else {  // Deconv3D
      OP_REQUIRES(context_, input_depth == filter_shape.dim_size(3),
                  errors::InvalidArgument(
                      "input and filter must have the same depth: ",
                      input_depth, " vs ", filter_shape.dim_size(3)));
      // TF filter is always in (planes, rows, cols, in_depth, out_depth) order
      int filter_planes =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_P));
      int filter_rows =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_H));
      int filter_cols =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_W));
      int filter_in_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_O));
      int filter_out_depth =
          static_cast<int>(filter_shape.dim_size(TF_3DFILTER_DIM_I));

      // oneDNN always needs filter in OIDHW format.
      // OIDHW = (out_depth, in_depth, planes, rows, cols)
      std::vector<memory::dim> filter_sizes(5, -1);
      filter_sizes[MklDnnDims3D::Dim3d_O] = filter_out_depth;
      filter_sizes[MklDnnDims3D::Dim3d_I] = filter_in_depth;
      filter_sizes[MklDnnDims3D::Dim3d_D] = filter_planes;
      filter_sizes[MklDnnDims3D::Dim3d_H] = filter_rows;
      filter_sizes[MklDnnDims3D::Dim3d_W] = filter_cols;
      *filter_dims = filter_sizes;
    }
  }

  // oneDNN format : OIHW for 2D and OIDHW for 3D
  // TF format : HWOI for 2D and DHWOI for 3D
  // Stride should be {I, 1, W*O*I, O*I} for 2D
  // and {I, 1, H*W*O*I, W*O*I, O*I} for 3D
  // Note: oneDNN does not directly support HWOI and DHWOI formats.
  //       So the user memory descriptor needs to be created with strides.
  virtual inline void GetFilterStrides(const memory::dims filter_dims,
                                       memory::dims* filter_strides) {
    OP_REQUIRES(context_, (filter_dims.size() == 4 || filter_dims.size() == 5),
                errors::InvalidArgument("filter must be 4- or 5-dimensional"));

    // For developers: keep the stride element computation order.
    if (filter_dims.size() == 4) {  // Deconv2D
      std::vector<memory::dim> strides(4, -1);
      strides[0] = filter_dims[1];
      strides[1] = 1;
      strides[3] = filter_dims[0] * filter_dims[1];
      strides[2] = filter_dims[3] * strides[3];
      *filter_strides = strides;
    } else {  // Deconv3D
      std::vector<memory::dim> strides(5, -1);
      strides[0] = filter_dims[1];
      strides[1] = 1;
      strides[4] = filter_dims[0] * filter_dims[1];
      strides[3] = filter_dims[4] * strides[4];
      strides[2] = filter_dims[3] * strides[3];
      *filter_strides = strides;
    }
  }

  // Calculate Deconvolution filter size in oneDNN order.
  // oneDNN requires filter in OIHW (Deconv2D).
  // Function does not return anything. But errors arising from sanity
  // checks are returned in context's status.
  virtual inline void GetFilterSize(size_t src_index, size_t filter_index,
                                    memory::dims* filter_dims) {
    DCHECK(filter_dims);
    GetFilterSize(GetTfShape(context_, src_index),
                  GetTfShape(context_, filter_index), filter_dims);
  }

  // Calculate Bias size for 2D/3D Deconvolution. Function does not
  // return anything, but may set an error in context status.
  virtual inline void GetBiasSize(size_t bias_index, memory::dims* bias_dims) {
    const Tensor& bias = MklGetInput(context_, bias_index);
    if (bias.dims() > 1) {
      if (strides_.size() == 4) {
        OP_REQUIRES(
            context_, bias.dims() <= 4,
            errors::InvalidArgument("For NHWC format, bias should have  "
                                    "4 or less dimensions",
                                    bias.shape().DebugString()));
      } else if (strides_.size() == 5) {
        OP_REQUIRES(
            context_, bias.dims() <= 5,
            errors::InvalidArgument("For NDHWC format, bias should have  "
                                    "5 or less dimensions",
                                    bias.shape().DebugString()));
      }
      // Make sure all the dims except channel(last) is 1
      for (int i = 0; i < bias.dims() - 1; i++) {
        OP_REQUIRES(
            context_, bias.dim_size(i) == 1,
            errors::InvalidArgument("For bias_dims > 1, all except the last "
                                    "dimension (channel) must be 1: ",
                                    bias.shape().DebugString()));
      }
      *bias_dims = {static_cast<int>(bias.dim_size(bias.dims() - 1))};
    } else {
      *bias_dims = {static_cast<int>(bias.dim_size(0))};
    }
  }

  // Function to calculate output and padding size for 2D/3D deconvolution.
  //
  // Calculate output shape of Deconvolution in oneDNN and TensorFlow order.
  // oneDNN uses NCHW(Deconv2D) output order.
  // But TensorFlow output will be in NHWC||NCHW depending on data format.
  // Function also calculates left, right, top and bottom pads.
  // Function does not return any status which is set with context status.
  //
  virtual inline void GetOutputAndPadSize(
      const TensorShape& input_shape, const TensorShape& filter_shape,
      const memory::dims& strides, const memory::dims& dilations,
      memory::dims* pad_l, memory::dims* pad_r, bool pad_enabled = false) {
    DCHECK(pad_l);
    DCHECK(pad_r);

    bool is_deconv2d = (strides_.size() == 4);

    int input_rows, input_cols, input_planes;
    if (is_deconv2d) {
      input_rows = GetTensorDim(input_shape, data_format_, 'H');
      input_cols = GetTensorDim(input_shape, data_format_, 'W');
    } else {
      input_rows = GetTensorDim(input_shape, data_format_, '1');
      input_cols = GetTensorDim(input_shape, data_format_, '2');
      input_planes = GetTensorDim(input_shape, data_format_, '0');
    }

    // Filter dimension
    // Deconv2D
    //    First dimension: rows/height.
    //    Second dimension: cols/width.
    // Deconv3D
    //    First dimension: planes/depth.
    //    Second dimension: rows/height.
    //    Third dimension: cols/width.

    int filter_rows, filter_cols, filter_planes;
    if (is_deconv2d) {
      filter_rows = filter_shape.dim_size(TF_2DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_2DFILTER_DIM_W);
    } else {
      filter_rows = filter_shape.dim_size(TF_3DFILTER_DIM_H);
      filter_cols = filter_shape.dim_size(TF_3DFILTER_DIM_W);
      filter_planes = filter_shape.dim_size(TF_3DFILTER_DIM_P);
    }

    int stride_planes, stride_rows, stride_cols;
    int dilation_planes, dilation_rows, dilation_cols;
    if (is_deconv2d) {
      // Deconv2D stride is a vector of 2 elements: {s_r, s_c}
      stride_rows = strides[0];
      stride_cols = strides[1];
      dilation_rows = dilations[0];
      dilation_cols = dilations[1];
    } else {
      // Deconv3D stride is a vector of 3 elements: {s_d, s_r, s_c}
      stride_planes = strides[0];
      stride_rows = strides[1];
      stride_cols = strides[2];
      dilation_planes = dilations[0];
      dilation_rows = dilations[1];
      dilation_cols = dilations[2];
    }

    // Output batch is same as input batch.
    int out_batch = GetTensorDim(input_shape, data_format_, 'N');

    // Output depth is same as the last dimension for filters for deonvolutions.
    int out_depth = filter_shape.dim_size(
        is_deconv2d ? static_cast<int>(TF_2DFILTER_DIM_O)
                    : static_cast<int>(TF_3DFILTER_DIM_O));

    int64 out_rows = 0, out_cols = 0, out_planes = 0;
    int64 pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    int64 pad_front, pad_back;

    if (is_deconv2d) {
      Padding padding_type;
      if (pad_enabled) {
        padding_type = Padding::EXPLICIT;
        pad_top = static_cast<int64_t>((*pad_l)[0]);
        pad_left = static_cast<int64_t>((*pad_l)[1]);
        pad_bottom = static_cast<int64_t>((*pad_r)[0]);
        pad_right = static_cast<int64_t>((*pad_r)[1]);
      } else {
        padding_type = padding_;
      }
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_rows, filter_rows, dilation_rows, stride_rows,
                         padding_type, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_cols, filter_cols, dilation_cols, stride_cols,
                         padding_type, &out_cols, &pad_left, &pad_right));
    } else {
      Padding padding_type;
      if (pad_enabled) {
        padding_type = Padding::EXPLICIT;
        pad_front = static_cast<int64>((*pad_l)[0]);
        pad_top = static_cast<int64>((*pad_l)[1]);
        pad_left = static_cast<int64>((*pad_l)[2]);
        pad_back = static_cast<int64>((*pad_r)[0]);
        pad_bottom = static_cast<int64>((*pad_r)[1]);
        pad_right = static_cast<int64>((*pad_r)[2]);
      } else {
        padding_type = padding_;
      }
      OP_REQUIRES_OK(context_, GetWindowedOutputSizeVerboseV2(
                                   input_planes, filter_planes, dilation_planes,
                                   stride_planes, padding_type, &out_planes,
                                   &pad_front, &pad_back));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_rows, filter_rows, dilation_rows, stride_rows,
                         padding_type, &out_rows, &pad_top, &pad_bottom));
      OP_REQUIRES_OK(context_,
                     GetWindowedOutputSizeVerboseV2(
                         input_cols, filter_cols, dilation_cols, stride_cols,
                         padding_type, &out_cols, &pad_left, &pad_right));
    }

    if (!pad_enabled) {
      // If pad_enabled, i.e., pad and deconv op are fused, then
      // all pads are already passed from pad op through
      // *pad_l and *pad_r and they don't need to be set here.
      if (is_deconv2d) {
        *pad_l = {static_cast<int>(pad_top), static_cast<int>(pad_left)};
        *pad_r = {static_cast<int>(pad_bottom), static_cast<int>(pad_right)};
      } else {
        *pad_l = {static_cast<int>(pad_front), static_cast<int>(pad_top),
                  static_cast<int>(pad_left)};
        *pad_r = {static_cast<int>(pad_back), static_cast<int>(pad_bottom),
                  static_cast<int>(pad_right)};
      }
    }

    // Tensorflow output is in data_format order.
    //     Deconv2D: NHWC or NCHW
    // oneDNN uses asymmetric padding.
    TensorShape out_shape =
        is_deconv2d
            ? ShapeFromFormat(data_format_, out_batch, out_rows, out_cols,
                              out_depth)
            : ShapeFromFormat(data_format_, out_batch,
                              {{out_planes, out_rows, out_cols}}, out_depth);
  }

  // Calculate output and pad size of forward Deconvolution operator.
  // See comment on GetDeconvOutputAndPadSize for parameters.
  // Function does not return anything, but sets error in context status.
  inline void GetOutputAndPadSize(size_t src_index, size_t filter_index,
                                  const memory::dims& strides,
                                  const memory::dims& dilations,
                                  memory::dims* pad_l, memory::dims* pad_r) {
    DCHECK(pad_l);
    DCHECK(pad_r);
    DCHECK(strides_.size() == 4 || strides_.size() == 5);

    auto input_tf_shape = GetTfShape(context_, src_index);
    auto filter_tf_shape = GetTfShape(context_, filter_index);

    if (strides_.size() == 4) {
      // Deconv2D
      OP_REQUIRES(context_, input_tf_shape.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional",
                                          input_tf_shape.DebugString()));
    } else {
      // Deconv3D
      OP_REQUIRES(context_, input_tf_shape.dims() == 5,
                  errors::InvalidArgument("input must be 5-dimensional",
                                          input_tf_shape.DebugString()));
    }

    GetOutputAndPadSize(input_tf_shape, filter_tf_shape, strides, dilations,
                        pad_l, pad_r);
  }

  // Wrapper function to calculate input, filter, and Deconv output sizes of
  // in oneDNN order: NCHW for input and output; OIHW for filter.
  // Function also calculates output shape in Tensorflow order.
  // Additionally, it also calculates strides and paddings.
  // Function does not return anything, but sets error in context status.
  inline void GetDeconvFwdSizes(const TensorShape& input_shape,
                                const TensorShape& filter_shape,
                                memory::dims* input_dims,
                                memory::dims* filter_dims,
                                memory::dims* strides, memory::dims* dilations,
                                memory::dims* pad_l, memory::dims* pad_r,
                                bool pad_enabled = false) {
    DCHECK(input_dims);
    DCHECK(filter_dims);
    DCHECK(strides);
    DCHECK(dilations);
    DCHECK(pad_l);
    DCHECK(pad_r);

    GetInputSize(input_shape, input_dims);
    if (!context_->status().ok()) return;

    GetFilterSize(input_shape, filter_shape, filter_dims);
    if (!context_->status().ok()) return;

    GetStrides(strides);
    GetDilations(dilations);

    GetOutputAndPadSize(input_shape, filter_shape, *strides, *dilations, pad_l,
                        pad_r, pad_enabled);
    if (!context_->status().ok()) return;
  }
};

}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_CONV_OPS_H_
