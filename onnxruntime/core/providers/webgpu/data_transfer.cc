// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "core/providers/webgpu/data_transfer.h"

namespace onnxruntime {
namespace webgpu {

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  if (bytes > 0) {
    //const void* src_data = src.DataRaw();
    //void* dst_data = dst.MutableDataRaw();

    auto& src_device = src.Location().device;
    auto& dst_device = dst.Location().device;

    if (dst_device.Type() == OrtDevice::GPU) {
      if (src_device.Type() == OrtDevice::GPU) {
        // copy from GPU to GPU
        //EM_ASM({ Module.jsepCopy($0, $1, $2, true); }, src_data, dst_data, bytes);
        ORT_ENFORCE(false, "not implemented");
      } else {
        // copy from CPU to GPU
        //EM_ASM({ Module.jsepCopy($0, $1, $2); }, src_data, dst_data, bytes);
        ORT_ENFORCE(false, "not implemented");
      }
    } else /* if (src_device.Type() == OrtDevice::GPU) */ {
      // copy from GPU to CPU
      //jsepDownload(src_data, dst_data, bytes);
      ORT_ENFORCE(false, "not implemented");
    }
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace onnxruntime
