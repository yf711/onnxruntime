// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <mutex>

#include <webgpu/webgpu_cpp.h>

namespace onnxruntime {
namespace webgpu {

// Class WebGpuInstance is a singleton class that holds the WebGPU instance.
class WebGpuInstance {
 public:
  static WebGpuInstance& Get() {
    static WebGpuInstance instance;
    return instance;
  }

  void Init();

  // wgpu::Instance GetInstance() { return instance_; }
  wgpu::Adapter Adapter() { return adapter_; }
  wgpu::Device Device() { return device_; }

 private:
  WebGpuInstance() {}

  wgpu::Instance instance_;
  wgpu::Adapter adapter_;
  wgpu::Device device_;
};

}  // namespace webgpu
}  // namespace onnxruntime
