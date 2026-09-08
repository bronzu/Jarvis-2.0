JARVIS 2.0 — Third-Party Notices
=================================

This document lists the third-party software components used by JARVIS 2.0,
their licenses, and attribution requirements.

---

## Runtime Dependencies (from requirements.txt)

### 1. numpy
- **Version used**: 2.4.2 (observed in current environment)
- **License**: BSD-3-Clause
- **Copyright**: Copyright (c) 2005-2024, NumPy Developers. All rights reserved.
- **Project URL**: https://numpy.org/
- **License obligations**: Retain copyright notice, conditions, and disclaimer in redistributions.
- **Role in JARVIS**: Core numerical computing for MicroTransformer (tensor operations, linear algebra, random number generation).
- **License text**: https://github.com/numpy/numpy/blob/main/LICENSE.txt

```
Copyright (c) 2005-2024, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

### 2. requests
- **Version used**: 2.32.5 (observed in current environment)
- **License**: Apache-2.0
- **Copyright**: Copyright (c) 2011-2024, Kenneth Reitz and contributors.
- **Project URL**: https://requests.readthedocs.io/
- **License obligations**: Include license copy, retain NOTICE file if present, state changes.
- **Role in JARVIS**: HTTP client for InternetTool and WebBrowserTool (network-enabled tools).
- **License text**: https://github.com/psf/requests/blob/main/LICENSE

```
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

[Full Apache-2.0 license text available at the project URL above]
```

---

### 3. beautifulsoup4
- **Version declared**: Not pinned in requirements.txt (latest compatible)
- **Version observed**: Not installed in current environment (declared dependency)
- **License**: MIT
- **Copyright**: Copyright (c) 2004-2024, Leonard Richardson and contributors.
- **Project URL**: https://www.crummy.com/software/BeautifulSoup/
- **License obligations**: Retain copyright notice and license in all copies.
- **Role in JARVIS**: HTML parsing for WebBrowserTool (web scraping/extraction).
- **License text**: https://github.com/beautifulsoup/beautifulsoup4/blob/main/LICENSE

```
The MIT License (MIT)

Copyright (c) 2004-2024 Leonard Richardson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

### 4. pyyaml
- **Version declared**: Not pinned in requirements.txt (latest compatible)
- **Version observed**: Not installed in current environment (declared dependency)
- **License**: MIT
- **Copyright**: Copyright (c) 2017-2024, PyYAML contributors.
- **Project URL**: https://pyyaml.org/
- **License obligations**: Retain copyright notice and license in all copies.
- **Role in JARVIS**: YAML parsing for config.yaml configuration loading.
- **License text**: https://github.com/yaml/pyyaml/blob/master/LICENSE

```
The MIT License (MIT)

Copyright (c) 2017-2024 PyYAML contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

### 5. pytest
- **Version used**: 9.1.1 (observed in current environment)
- **License**: MIT
- **Copyright**: Copyright (c) 2004-2024, Holger Krekel and pytest-dev team.
- **Project URL**: https://pytest.org/
- **License obligations**: Retain copyright notice and license in all copies.
- **Role in JARVIS**: Test framework (development/test dependency only, not runtime).
- **License text**: https://github.com/pytest-dev/pytest/blob/main/LICENSE

```
The MIT License (MIT)

Copyright (c) 2004-2024 Holger Krekel and pytest-dev team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Optional / Development Dependencies

### 6. PyTorch (torch)
- **Status**: Optional — used only for GPU parity testing
- **Imported in**: `neural_core/torch_model.py`, `neural_core/gpu_benchmark_extended.py`
- **Not in requirements.txt**: Not required for production inference
- **License**: BSD-3-Clause
- **Copyright**: Copyright (c) 2016-2024, PyTorch contributors.
- **Project URL**: https://pytorch.org/
- **Role in JARVIS**: GPU/MPS benchmarking and numerical parity verification only; the production model runs on pure NumPy (MicroTransformer).
- **License text**: https://github.com/pytorch/pytorch/blob/main/LICENSE

---

## System-Level Dependencies

### 7. macOS sandbox-exec
- **Type**: System binary (not distributed with JARVIS)
- **Platform**: macOS only
- **License**: Apple Public Source License 2.0 (APSL-2.0)
- **Role in JARVIS**: Kernel-enforced Mandatory Access Control for OS-level sandbox isolation (`neural_core/os_sandbox.py`).
- **Note**: JARVIS invokes `sandbox-exec` as a subprocess; it does not bundle or modify it.

### 8. Linux chroot / rlimits / setsid
- **Type**: System calls (Linux kernel)
- **Platform**: Linux only (requires root for full chroot isolation)
- **License**: GPL-2.0 (Linux kernel) — system calls are not distribution
- **Role in JARVIS**: Process isolation, resource limits, filesystem boundaries for OS sandbox (`neural_core/os_sandbox.py`).

---

## Python Standard Library

JARVIS 2.0 uses the Python standard library extensively. No additional licensing obligations beyond the Python Software Foundation License (PSF-2.0).

Key modules used: `ast`, `asyncio`, `collections`, `contextlib`, `dataclasses`, `datetime`, `enum`, `hashlib`, `html`, `importlib`, `inspect`, `io`, `json`, `logging`, `math`, `os`, `pathlib`, `pickle`, `random`, `re`, `resource`, `shlex`, `signal`, `socket`, `subprocess`, `sys`, `tempfile`, `textwrap`, `threading`, `time`, `traceback`, `types`, `typing`, `uuid`, `warnings`, `weakref`, `xml`.

---

## Summary of License Compatibility

| Dependency | License | Type | Copyleft? | Compatible with Proprietary? |
|---|---|---|---|---|
| numpy | BSD-3-Clause | Runtime | No | Yes |
| requests | Apache-2.0 | Runtime | No | Yes |
| beautifulsoup4 | MIT | Runtime | No | Yes |
| pyyaml | MIT | Runtime | No | Yes |
| pytest | MIT | Dev/Test | No | Yes |
| PyTorch | BSD-3-Clause | Optional/Dev | No | Yes |
| macOS sandbox-exec | APSL-2.0 | System | No | Yes (system call) |
| Linux syscalls | GPL-2.0 (kernel) | System | N/A | Yes (syscall boundary) |

**All declared and observed dependencies use permissive licenses compatible with proprietary distribution.**

---

## Verification Notes

- **Versions verified via pip/environment**: numpy (2.4.2), requests (2.32.5), pytest (9.1.1)
- **Versions NOT verified (not installed in current env)**: beautifulsoup4, pyyaml — declared in requirements.txt but not currently installed
- **PyTorch**: Optional, not in requirements.txt, not required for production
- **No GPL/copyleft dependencies** in the runtime dependency chain
- **No bundled third-party code** — all dependencies are external packages

---

## Attribution Requirements for Redistribution

If JARVIS 2.0 is redistributed (in binary or source form), the following must be included:

1. This THIRD_PARTY_NOTICES.md file (or equivalent)
2. License texts for all runtime dependencies (numpy, requests, beautifulsoup4, pyyaml)
3. Copyright notices preserved as required by each license

For Apache-2.0 (requests): Include the NOTICE file if present in the distribution.

---

*Generated as part of the Technical Acquisition Package*
*Date: 2026-09-08*
*Classification: Due Diligence Only*