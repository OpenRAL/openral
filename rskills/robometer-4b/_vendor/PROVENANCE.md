# Provenance — robometer package (Phase 0)

The `RBM`/`RFM` model class and its inference path cannot be loaded by vanilla
`transformers.AutoModel` (the HF `config.json` advertises `architectures: ["RFM"]`
with no `auto_map` and no Hub-side modeling code). Loading requires the upstream
`robometer` package.

- **Upstream:** https://github.com/robometer/robometer
- **Pinned commit:** `a669dffc241d7d76bec12f36efd4084d914d017c`
- **Install (isolated, one-time, user-authorized):**
  ```
  uv pip install "robometer[robometer,quantization] @ git+https://github.com/robometer/robometer@a669dffc241d7d76bec12f36efd4084d914d017c"
  ```
- **License:** weights Apache-2.0 (`robometer/Robometer-4B`); code per the repo's LICENSE.
- **Trust note (CLAUDE.md §3):** `robometer` is not an OpenRAL-trusted org. Executing its
  code was explicitly authorized by the user for this Phase 0 spike and runs in an
  isolated `/tmp/robometer-env` venv, not the main repo venv. A production sidecar must
  keep this isolation and pin the SHA.

`probe.py` here is OpenRAL-authored (not vendored) and mirrors the upstream
`scripts/example_inference_local.py` input-construction path.
