// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {
namespace webgpu {

using SupportedTypes =
    TypeList<
        float,
        MLFloat16,
        int32_t,
        uint32_t>;

using SupportedFloats =
    TypeList<
        float,
        MLFloat16>;

const std::vector<MLDataType>& WebGpuSupportedDataTypes() {
  static const std::vector<MLDataType> supportedDataTypes = BuildKernelDefConstraintsFromTypeList<SupportedTypes>();
  return supportedDataTypes;
}

const std::vector<MLDataType>& WebGpuSupportedFloatTypes() {
  static const std::vector<MLDataType> supportedDataTypes = BuildKernelDefConstraintsFromTypeList<SupportedFloats>();
  return supportedDataTypes;
}

}  // namespace webgpu
}  // namespace onnxruntime
