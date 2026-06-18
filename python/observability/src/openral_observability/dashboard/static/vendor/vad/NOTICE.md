# Vendored voice-prompt assets

These files are vendored verbatim so the dashboard's voice prompt works **fully
offline** — no CDN, no network at runtime. The dashboard's `dashboard.js`
points `baseAssetPath` / `onnxWASMBasePath` (and the two `<script>` srcs) at
this directory (`/static/vendor/vad/`).

All third-party, redistributed under permissive licenses (OpenRAL's own code
stays Apache-2.0 per ADR-0012; these are external assets we ship unmodified).

| File | Source package | Version | License |
| --- | --- | --- | --- |
| `bundle.min.js` | [`@ricky0123/vad-web`](https://github.com/ricky0123/vad) | 0.0.29 | ISC |
| `bundle.min.js.LICENSE.txt` | (bundled attribution) | — | — |
| `vad.worklet.bundle.min.js` | `@ricky0123/vad-web` | 0.0.29 | ISC |
| `silero_vad_v5.onnx` | [Silero VAD](https://github.com/snakers4/silero-vad) (via vad-web) | v5 | MIT |
| `silero_vad_legacy.onnx` | Silero VAD (via vad-web) | legacy | MIT |
| `ort.wasm.min.js` | [`onnxruntime-web`](https://github.com/microsoft/onnxruntime) | 1.22.0 | MIT |
| `ort-wasm-simd-threaded.wasm` | `onnxruntime-web` | 1.22.0 | MIT |
| `ort-wasm-simd-threaded.mjs` | `onnxruntime-web` | 1.22.0 | MIT |

## Refreshing

```bash
npm pack @ricky0123/vad-web@0.0.29 onnxruntime-web@1.22.0
# from @ricky0123/vad-web dist/: bundle.min.js, bundle.min.js.LICENSE.txt,
#   vad.worklet.bundle.min.js, silero_vad_v5.onnx, silero_vad_legacy.onnx
# from onnxruntime-web dist/ (CPU/wasm build only — skip the 21 MB *.jsep.* WebGPU
#   variants): ort.wasm.min.js, ort-wasm-simd-threaded.wasm, ort-wasm-simd-threaded.mjs
```

Keep the versions here in sync with the `<script>` src pins and the version
constants at the top of the voice-prompt block in `../../dashboard.js`.
