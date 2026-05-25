# Third-Party Notices

This repository is based on `llama.cpp` and preserves the upstream MIT license
and third-party license files that ship with the tree.

## Main Project

- `llama.cpp` / `ggml`
  - License: MIT
  - License file: `LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2023-2026 The ggml authors`

## Bundled Third-Party Components

- `cpp-httplib`
  - License: MIT
  - License file: `vendor/cpp-httplib/LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2017 yhirose`

- `nlohmann/json`
  - License: MIT
  - License file: `licenses/LICENSE-jsonhpp`
  - Copyright notice in this checkout: `Copyright (c) 2013-2025 Niels Lohmann`

- `gguf-py`
  - License: MIT
  - License file: `gguf-py/LICENSE`
  - Copyright notice in this checkout: `Copyright (c) 2023 Georgi Gerganov`

## Generated and Ignored Artifacts

Build directories, generated benchmark reports, logs, and GGUF model files are
ignored by `.gitignore` and are not intended to be published in this source
repository.

Model weights are not included. Any model downloaded or quantized for ROCmFP4
testing remains subject to the original model publisher's license and terms.

## ROCmFP4 Additions

The ROCmFP4 source files, scripts, and documentation added in this branch are
provided under the same MIT license as the rest of this source tree unless a
file states otherwise.
