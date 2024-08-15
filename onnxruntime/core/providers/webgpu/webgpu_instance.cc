// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/webgpu/webgpu_instance.h"

namespace onnxruntime {
namespace webgpu {

std::vector<const wgpu::FeatureName> GetAvailableRequiredFeatures(const wgpu::Adapter& adapter) {
  std::vector<const wgpu::FeatureName> required_features;
  constexpr wgpu::FeatureName features[]{
      wgpu::FeatureName::ChromiumExperimentalTimestampQueryInsidePasses,
      wgpu::FeatureName::TimestampQuery,
      wgpu::FeatureName::ShaderF16};
  for (auto feature : features) {
    if (adapter.HasFeature(feature)) {
      required_features.push_back(feature);
    }
  }
  return std::move(required_features);
}

void WebGpuInstance::Init() {
  static std::once_flag init_flag;
  std::call_once(init_flag, [this]() {
    // Initialization.Step.1 - Create wgpu::Instance

    wgpu::InstanceDescriptor instance_desc{};
    instance_desc.features.timedWaitAnyEnable = true;
    instance_ = wgpu::CreateInstance(&instance_desc);

    ORT_ENFORCE(instance_ != nullptr, "Failed to create wgpu::Instance.");

    // Initialization.Step.2 - Create wgpu::Adapter

    wgpu::RequestAdapterOptions req_adapter_options = {};
    wgpu::RequestAdapterCallbackInfo req_adapter_callback_info = {};
    req_adapter_callback_info.mode = wgpu::CallbackMode::WaitAnyOnly;
    req_adapter_callback_info.callback = [](WGPURequestAdapterStatus status,
                                            WGPUAdapter adapter, const char* message,
                                            void* userdata) {
      ORT_ENFORCE(status == WGPURequestAdapterStatus_Success, "Failed to get a WebGPU adapter: ", message);
      *static_cast<wgpu::Adapter*>(userdata) = wgpu::Adapter::Acquire(adapter);
    };
    req_adapter_callback_info.userdata = &adapter_;
    ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(instance_.RequestAdapter(&req_adapter_options, req_adapter_callback_info), UINT64_MAX));

    // Initialization.Step.3 - Create wgpu::Device

    std::vector<const wgpu::FeatureName> required_features = GetAvailableRequiredFeatures(adapter_);
    wgpu::DeviceDescriptor device_desc = {};
    device_desc.requiredFeatures = required_features;
    wgpu::RequestDeviceCallbackInfo req_device_callback_info = {};
    req_device_callback_info.mode = wgpu::CallbackMode::WaitAnyOnly;
    req_device_callback_info.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* userdata) {
      ORT_ENFORCE(status == WGPURequestAdapterStatus_Success, "Failed to get a WebGPU device: ", message);
      *static_cast<wgpu::Device*>(userdata) = wgpu::Device::Acquire(device);
    };
    req_device_callback_info.userdata = &device_;
    ORT_ENFORCE(wgpu::WaitStatus::Success == instance_.WaitAny(adapter_.RequestDevice(&device_desc, req_device_callback_info), UINT64_MAX));

    // Check limits

    wgpu::SupportedLimits limits;
    ORT_ENFORCE(device_.GetLimits(&limits));
  });
}

}  // namespace webgpu
}  // namespace onnxruntime
