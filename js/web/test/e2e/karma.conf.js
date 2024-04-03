// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const args = require('minimist')(process.argv.slice(2));
const SELF_HOST = !!args['self-host'];
const ORT_MAIN = args['ort-main'];
const TEST_MAIN = args['test-main'];
if (typeof TEST_MAIN !== 'string') {
  throw new Error('flag --test-main=<TEST_MAIN_JS_FILE> is required');
}
const USER_DATA = args['user-data'];
if (typeof USER_DATA !== 'string') {
  throw new Error('flag --user-data=<CHROME_USER_DATA_FOLDER> is required');
}

const testArgs = args['test-args'];
const normalizedTestArgs = !testArgs || Array.isArray(testArgs) ? testArgs : [testArgs];

const files = [
  {pattern: './model.onnx', included: false},
  {pattern: './model_with_orig_ext_data.onnx', included: false},
  {pattern: './model_with_orig_ext_data.bin', included: false},
];
if (ORT_MAIN) {
  files.push(
      {pattern: (SELF_HOST ? './node_modules/onnxruntime-web/dist/' : 'http://localhost:8081/dist/') + ORT_MAIN});
}
if (TEST_MAIN.endsWith('.mjs')) {
  files.push({pattern: TEST_MAIN, type: 'module'});
} else {
  files.push({pattern: './common.js'}, {pattern: TEST_MAIN});
}
files.push({pattern: './dist/**/*', included: false, nocache: true, watched: false});
if (SELF_HOST) {
  files.push({pattern: './node_modules/onnxruntime-web/dist/*.*', included: false, nocache: true});
}

const flags = ['--ignore-gpu-blocklist', '--gpu-vendor-id=0x10de', '--enable-features=SharedArrayBuffer'];

module.exports = function(config) {
  config.set({
    frameworks: ['mocha'],
    files,
    plugins: [require('@chiragrupani/karma-chromium-edge-launcher'), ...config.plugins],
    proxies: {
      '/model.onnx': '/base/model.onnx',
      '/model_with_orig_ext_data.onnx': '/base/model_with_orig_ext_data.onnx',
      '/model_with_orig_ext_data.bin': '/base/model_with_orig_ext_data.bin',
      '/test-wasm-path-override/ort-wasm.mjs': '/base/node_modules/onnxruntime-web/dist/ort-wasm.mjs',
      '/test-wasm-path-override/ort-wasm.wasm': '/base/node_modules/onnxruntime-web/dist/ort-wasm.wasm',
      '/test-wasm-path-override/renamed.wasm': '/base/node_modules/onnxruntime-web/dist/ort-wasm.wasm',
    },
    client: {captureConsole: true, args: normalizedTestArgs, mocha: {expose: ['body'], timeout: 60000}},
    reporters: ['mocha'],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    hostname: 'localhost',
    browsers: [],
    customLaunchers: {
      Chrome_default: {base: 'Chrome', flags, chromeDataDir: USER_DATA},
      Chrome_no_threads: {
        base: 'Chrome',
        chromeDataDir: USER_DATA,
        flags
        // TODO: no-thread flags
      },
      Edge_default: {base: 'Edge', edgeDataDir: USER_DATA}
    }
  });
};