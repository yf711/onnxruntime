// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env} from 'onnxruntime-common';

import type {OrtWasmModule, OrtWasmThreadedModule} from './wasm-types';
import {dynamicImportDefault, preloadWorker, scriptSrc} from './wasm-utils-import';

let wasm: OrtWasmModule|undefined;
let initialized = false;
let initializing = false;
let aborted = false;

const isMultiThreadSupported = (numThreads: number): boolean => {
  // WebAssembly threads are set to 1 (single thread).
  if (numThreads === 1) {
    return false;
  }

  // If 'SharedArrayBuffer' is not available, WebAssembly threads will not work.
  if (typeof SharedArrayBuffer === 'undefined') {
    if (typeof self !== 'undefined' && !self.crossOriginIsolated) {
      // eslint-disable-next-line no-console
      console.warn(
          'env.wasm.numThreads is set to ' + numThreads +
          ', but this will not work unless you enable crossOriginIsolated mode. ' +
          'See https://web.dev/cross-origin-isolation-guide/ for more info.');
    }
    return false;
  }

  try {
    // Test for transferability of SABs (for browsers. needed for Firefox)
    // https://groups.google.com/forum/#!msg/mozilla.dev.platform/IHkBZlHETpA/dwsMNchWEQAJ
    if (typeof MessageChannel !== 'undefined') {
      new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
    }

    // Test for WebAssembly threads capability (for both browsers and Node.js)
    // This typed array is a WebAssembly program containing threaded instructions.
    return WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0,  0,  0, 1, 4, 1,  96, 0,   0,  3, 2, 1,  0, 5,
      4, 1,  3,   1,   1, 10, 11, 1, 9, 0, 65, 0,  254, 16, 2, 0, 26, 11
    ]));
  } catch (e) {
    return false;
  }
};

const isSimdSupported = (): boolean => {
  try {
    // Test for WebAssembly SIMD capability (for both browsers and Node.js)
    // This typed array is a WebAssembly program containing SIMD instructions.

    // The binary data is generated from the following code by wat2wasm:
    //
    // (module
    //   (type $t0 (func))
    //   (func $f0 (type $t0)
    //     (drop
    //       (i32x4.dot_i16x8_s
    //         (i8x16.splat
    //           (i32.const 0))
    //         (v128.const i32x4 0x00000000 0x00000000 0x00000000 0x00000000)))))

    return WebAssembly.validate(new Uint8Array([
      0,   97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 30, 1,   28,  0, 65, 0,
      253, 15, 253, 12,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0,  253, 186, 1, 26, 11
    ]));
  } catch (e) {
    return false;
  }
};

const getWasmFileNameWithoutExtension = (useSimd: boolean, useThreads: boolean) => {
  if (useSimd) {
    if (!BUILD_DEFS.DISABLE_TRAINING) {
      return 'ort-training-wasm-simd';
    }
    if (useThreads) {
      return BUILD_DEFS.DISABLE_JSEP ? 'ort-wasm-simd-threaded' : 'ort-wasm-simd-threaded.jsep';
    } else {
      return BUILD_DEFS.DISABLE_JSEP ? 'ort-wasm-simd' : 'ort-wasm-simd.jsep';
    }
  } else {
    return useThreads ? 'ort-wasm-threaded' : 'ort-wasm';
  }
};

export const initializeWebAssembly = async(flags: Env.WebAssemblyFlags): Promise<void> => {
  if (initialized) {
    return Promise.resolve();
  }
  if (initializing) {
    throw new Error('multiple calls to \'initializeWebAssembly()\' detected.');
  }
  if (aborted) {
    throw new Error('previous call to \'initializeWebAssembly()\' failed.');
  }

  initializing = true;

  // wasm flags are already initialized
  const timeout = flags.initTimeout!;
  const numThreads = flags.numThreads!;
  const simd = flags.simd!;

  const useThreads = !BUILD_DEFS.DISABLE_WASM_THREAD && isMultiThreadSupported(numThreads);
  const useSimd = simd && isSimdSupported();

  const wasmPaths = flags.wasmPaths;
  const wasmPrefixOverride = typeof wasmPaths === 'string' ? wasmPaths : undefined;
  const wasmFileName = getWasmFileNameWithoutExtension(useSimd, useThreads);
  const wasmPathOverride = typeof wasmPaths === 'object' ? wasmPaths[`${wasmFileName}.wasm`] : undefined;

  const ortWasmFactory: EmscriptenModuleFactory<OrtWasmModule> =
      (await dynamicImportDefault(`${wasmFileName}.mjs`, wasmPrefixOverride));

  const wasmWorkerFileName = !BUILD_DEFS.DISABLE_WASM_THREAD && useThreads ? `${wasmFileName}.worker.js` : '';
  const wasmWorkerUrl = !BUILD_DEFS.DISABLE_WASM_THREAD && useThreads ?
      await preloadWorker(`${wasmFileName}.worker.js`, wasmPrefixOverride ?? scriptSrc) :
      '';

  let isTimeout = false;

  const tasks: Array<Promise<void>> = [];

  // promise for timeout
  if (timeout > 0) {
    tasks.push(new Promise((resolve) => {
      setTimeout(() => {
        isTimeout = true;
        resolve();
      }, timeout);
    }));
  }

  // promise for module initialization
  tasks.push(new Promise((resolve, reject) => {
    const config: Partial<OrtWasmModule> = {
      locateFile: (fileName: string, scriptDirectory: string) => {
        if (fileName.endsWith('.wasm') && wasmPathOverride) {
          return wasmPathOverride;
        }

        if (!BUILD_DEFS.DISABLE_WASM_THREAD && useThreads && fileName === wasmWorkerFileName &&
            wasmWorkerUrl !== wasmWorkerFileName) {
          // when a valid rewritten worker URL is available, use it
          return wasmWorkerUrl;
        }

        return (wasmPrefixOverride ?? scriptDirectory) + fileName;
      }
    };

    if (!BUILD_DEFS.DISABLE_WASM_THREAD && useThreads) {
      config.numThreads = numThreads;
      if (wasmPrefixOverride) {
        config.mainScriptUrlOrBlob = wasmPrefixOverride + `${wasmFileName}.mjs`;
      }
    }

    ortWasmFactory(config).then(
        // wasm module initialized successfully
        module => {
          initializing = false;
          initialized = true;
          wasm = module;
          resolve();
        },
        // wasm module failed to initialize
        (what) => {
          initializing = false;
          aborted = true;
          reject(what);
        });
  }));

  await Promise.race(tasks);

  if (isTimeout) {
    throw new Error(`WebAssembly backend initializing failed due to timeout: ${timeout}ms`);
  }
};

export const getInstance = (): OrtWasmModule => {
  if (initialized && wasm) {
    return wasm;
  }

  throw new Error('WebAssembly is not initialized yet.');
};

export const dispose = (): void => {
  if (initialized && !initializing && !aborted) {
    initializing = true;

    (wasm as OrtWasmThreadedModule).PThread?.terminateAllThreads();
    wasm = undefined;

    initializing = false;
    initialized = false;
    aborted = true;
  }
};