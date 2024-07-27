// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Microsoft.ML.OnnxRuntime.Tests.MAUI
{
    public class PlatformTests
    {
        // All the 'AppendExecutionProvider' calls will throw if unsuccessful
#if ANDROID
        [Fact(DisplayName = "EnableNNAPI")]
        public void TestEnableNNAPI()
        {
            var opt = new SessionOptions();
            opt.AppendExecutionProvider_Nnapi();
        }
#endif

#if IOS
        [Fact(DisplayName = "EnableCoreML_NeuralNetwork")]
        public void TestEnableCoreML_NN()
        {
            var opt = new SessionOptions();
            opt.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
        }

        [Fact(DisplayName = "EnableCoreML_MLProgram")]
        public void TestEnableCoreML_MLProgram()
        {
            var opt = new SessionOptions();
            opt.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM);
        }
#endif
#if WINDOWS
        [Fact(DisplayName = "CPU_EP_NoArena")]
        public void TestEnableCoreML_MLProgram()
        {
            var opt = new SessionOptions();
            int useArena = 0;
            opt.AppendExecutionProvider_CPU(useArena);
        }
#endif

    }
}
