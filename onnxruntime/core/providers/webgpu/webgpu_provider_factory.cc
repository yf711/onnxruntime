// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_instance.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

struct WebGpuProviderFactory : IExecutionProviderFactory {
  WebGpuProviderFactory(const ProviderOptions& provider_options, const SessionOptions* session_options)
      : info_{provider_options}, session_options_(session_options) {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    return std::make_unique<WebGpuExecutionProvider>(info_, session_options_);
  }

 private:
  WebGpuExecutionProviderInfo info_;
  const SessionOptions* session_options_;
};

std::shared_ptr<IExecutionProviderFactory> WebGpuProviderFactoryCreator::Create(
    const ProviderOptions& provider_options, const SessionOptions* session_options) {
  webgpu::WebGpuInstance::Get().Init();
  return std::make_shared<WebGpuProviderFactory>(provider_options, session_options);
}

}  // namespace onnxruntime
