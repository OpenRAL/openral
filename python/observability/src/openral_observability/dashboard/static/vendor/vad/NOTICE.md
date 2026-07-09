# Vendored voice-prompt assets

These files back the dashboard's voice prompt so it works **without any
external service at request time** — no CDN calls from the browser, no
network once the assets are in place. The dashboard's `dashboard.js` points
`baseAssetPath` / `onnxWASMBasePath` (and the two `<script>` srcs) at this
directory (`/static/vendor/vad/`).

All third-party, redistributed under permissive licenses (OpenRAL's own code
stays Apache-2.0 under OpenRAL's licensing policy; these are external assets we ship unmodified).

**The three binary files are no longer committed to git** (they totalled
~15 MB): `silero_vad_v5.onnx`, `silero_vad_legacy.onnx`, and
`ort-wasm-simd-threaded.wasm`. `openral_observability.dashboard.vad_assets`
downloads and sha256-verifies them from pinned upstream URLs into
`$OPENRAL_CACHE_DIR/dashboard_assets/vad/` (default `~/.cache/openral/…`) the
first time `openral dashboard` starts, then hard-links/copies them into this
directory so they're served from the same path. See that module's docstring
for the pinned URLs/hashes. A failed/offline first run disables the mic
button client-side (`/api/config`'s `voice_prompt_enabled`) with a loud
`structlog` warning — it never blocks the dashboard from starting.
The small JS glue files below stay committed (tiny, reviewable as text).

| File | Source package | Version | License | Committed? |
| --- | --- | --- | --- | --- |
| `bundle.min.js` | [`@ricky0123/vad-web`](https://github.com/ricky0123/vad) | 0.0.29 | ISC | yes |
| `bundle.min.js.LICENSE.txt` | (bundled attribution) | — | — | yes |
| `vad.worklet.bundle.min.js` | `@ricky0123/vad-web` | 0.0.29 | ISC | yes |
| `silero_vad_v5.onnx` | [Silero VAD](https://github.com/snakers4/silero-vad) (via vad-web) | v5 | MIT | **no — fetched** |
| `silero_vad_legacy.onnx` | Silero VAD (via vad-web) | legacy | MIT | **no — fetched** |
| `ort.wasm.min.js` | [`onnxruntime-web`](https://github.com/microsoft/onnxruntime) | 1.22.0 | MIT | yes |
| `ort-wasm-simd-threaded.wasm` | `onnxruntime-web` | 1.22.0 | MIT | **no — fetched** |
| `ort-wasm-simd-threaded.mjs` | `onnxruntime-web` | 1.22.0 | MIT | yes |

## Refreshing

Bumping either upstream version means re-pinning **both** the committed JS
files here **and** the URLs/hashes in `vad_assets.py`:

```bash
npm pack @ricky0123/vad-web@0.0.29 onnxruntime-web@1.22.0
# from @ricky0123/vad-web dist/: bundle.min.js, bundle.min.js.LICENSE.txt,
#   vad.worklet.bundle.min.js (commit these three; silero_vad_v5.onnx /
#   silero_vad_legacy.onnx are fetched at runtime — do not commit them)
# from onnxruntime-web dist/ (CPU/wasm build only — skip the 21 MB *.jsep.* WebGPU
#   variants): ort.wasm.min.js, ort-wasm-simd-threaded.mjs (commit these two;
#   ort-wasm-simd-threaded.wasm is fetched at runtime — do not commit it)
```

Keep the versions here in sync with the `<script>` src pins and the version
constants at the top of the voice-prompt block in `../../dashboard.js`, and
update the pinned URLs + sha256 hashes in `vad_assets.py` to match.
