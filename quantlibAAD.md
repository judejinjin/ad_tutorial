# QuantLibAAD — Complete Tutorial

> Build · Test · Run · Architecture

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Concepts](#2-core-concepts)
   - 2.1 [Automatic Differentiation (AAD)](#21-automatic-differentiation-aad)
   - 2.2 [XAD — The AD Engine](#22-xad--the-ad-engine)
   - 2.3 [Integration Strategy: `QL_INCLUDE_FIRST`](#23-integration-strategy-ql_include_first)
   - 2.4 [JIT Compilation with Forge](#24-jit-compilation-with-forge)
3. [Repository Layout](#3-repository-layout)
4. [Dependencies](#4-dependencies)
5. [Building the Project](#5-building-the-project)
   - 5.1 [Prerequisites](#51-prerequisites)
   - 5.2 [Directory Layout Convention](#52-directory-layout-convention)
   - 5.3 [Build with CMake Presets (Recommended)](#53-build-with-cmake-presets-recommended)
   - 5.4 [Manual CMake Build](#54-manual-cmake-build)
   - 5.5 [Windows / Visual Studio](#55-windows--visual-studio)
   - 5.6 [CMake Options Reference](#56-cmake-options-reference)
   - 5.7 [Building with xad-forge (Optional JIT Backend)](#57-building-with-xad-forge-optional-jit-backend)
   - 5.8 [Building QuantLib-Risks-Py (Python Bindings)](#58-building-quantlib-risks-py-python-bindings)
6. [Running the Examples](#6-running-the-examples)
   - 6.1 [AdjointEuropeanEquityOption](#61-adjointeuropeanequityoption)
   - 6.2 [AdjointSwap](#62-adjointswap)
   - 6.3 [AdjointBermudanSwaption](#63-adjointbermudanswaption)
   - 6.4 [AdjointHestonModel](#64-adjointnestonmodel)
   - 6.5 [AdjointCDS](#65-adjointcds)
   - 6.6 [AdjointAmericanEquityOption](#66-adjointamericanequityoption)
   - 6.7 [AdjointMulticurveBootstrapping](#67-adjointmulticurvebootstrapping)
   - 6.8 [AdjointReplication](#68-adjointreplication)
7. [Running the Test Suite](#7-running-the-test-suite)
   - 7.1 [Enabling Tests](#71-enabling-tests)
   - 7.2 [Running Tests](#72-running-tests)
   - 7.3 [Benchmarks (FD vs AAD)](#73-benchmarks-fd-vs-aad)
8. [Technical Architecture](#8-technical-architecture)
   - 8.1 [Big Picture](#81-big-picture)
   - 8.2 [The `qlaad.hpp` Adaptor Header](#82-the-qlaadhpp-adaptor-header)
   - 8.3 [CMake Targets](#83-cmake-targets)
   - 8.4 [The XAD Tape Model](#84-the-xad-tape-model)
   - 8.5 [AAD Workflow Pattern](#85-aad-workflow-pattern)
   - 8.6 [JIT Pipeline Architecture (Forge)](#86-jit-pipeline-architecture-forge)
   - 8.7 [Dual-Mode Operation](#87-dual-mode-operation)
9. [Deep Dive: How `qlaad.hpp` Works](#9-deep-dive-how-qlaadhpp-works)
10. [Writing Your Own AAD-Enabled Code](#10-writing-your-own-aad-enabled-code)
11. [Common Issues and Troubleshooting](#11-common-issues-and-troubleshooting)
12. [License Summary](#12-license-summary)

---

## 1. Project Overview

**QuantLibAAD** is an integration layer that enables [QuantLib](https://www.quantlib.org) — one of the world's most widely used open-source quantitative finance libraries — to compute financial risk sensitivities (Greeks, DV01, CS01, etc.) using **Adjoint Algorithmic Differentiation (AAD)** via the [XAD](https://auto-differentiation.github.io) library.

Instead of rewriting QuantLib, QuantLibAAD injects XAD's active number type (`xad::AReal<double>`) as QuantLib's internal `Real` type **before** compilation. The result is that the entire QuantLib computation graph is automatically differentiated with near-zero code changes to the pricing library itself.

### What problems does it solve?

| Traditional approach | QuantLibAAD approach |
|---|---|
| Bump-and-reprice (finite differences): O(N) pricings per sensitivity set | One forward + one adjoint sweep: O(1) sweeps regardless of input count |
| Error-prone bump sizes | Exact machine-precision derivatives |
| Slow for large portfolios | Demonstrated many-fold speedups vs other AD tools |
| Requires code duplication | Single source of truth — same pricing functions used for price and sensitivities |

### What this repository contains

- **`ql/`** — The `qlaad.hpp` adaptor header and its CMake library target (`QuantLibAAD`)
- **`Examples/`** — Eight self-contained C++ applications demonstrating AAD on real instruments
- **`test-suite/`** — Unit tests validating AAD-computed derivatives against bump-and-reprice results, plus benchmarks
- **`presets/`** — Ready-made `CMakeUserPresets.json` templates for QuantLib integration

This repository is **not stand-alone** — it must be compiled as a subdirectory inside QuantLib's build tree.

---

## 2. Core Concepts

### 2.1 Automatic Differentiation (AAD)

AAD (also called *reverse-mode* AD or *backpropagation* in ML) computes exact derivatives of a scalar function with respect to all inputs in **a single backward sweep**.

For a pricing function $f: \mathbb{R}^N \to \mathbb{R}$:

$$\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_N}\right)$$

AAD computes this entire gradient with a cost roughly equivalent to **2–5× a single function evaluation**, independent of $N$ (number of inputs). Compare this to finite differences which require $N+1$ evaluations.

### 2.2 XAD — The AD Engine

[XAD](https://auto-differentiation.github.io) is the C++ AD library used here. Its key types are:

| Type | Purpose |
|---|---|
| `xad::AReal<double>` | Active scalar — records operations on a global tape |
| `xad::Tape<double>` | The computation tape (call graph) |
| `xad::AReal<double>::tape_type` | Convenience alias for the tape |
| `derivative(x)` | Returns the adjoint (sensitivity) of variable `x` |
| `tape.registerInput(x)` | Marks `x` as a differentiation input |
| `tape.registerOutput(y)` | Marks `y` as the output to differentiate |
| `tape.newRecording()` | Starts recording operations |
| `tape.computeAdjoints()` | Executes the reverse sweep |

### 2.3 Integration Strategy: `QL_INCLUDE_FIRST`

QuantLib supports a preprocessor hook: if `QL_INCLUDE_FIRST` is defined as a header path, that header is included at the top of every QuantLib translation unit before anything else.

QuantLibAAD sets:

```cmake
target_compile_definitions(QuantLibAAD INTERFACE QL_INCLUDE_FIRST=ql/qlaad.hpp)
```

And `qlaad.hpp` defines:

```cpp
#define QL_REAL xad::AReal<double>
```

This replaces QuantLib's `Real` type with XAD's active type everywhere — no changes to QuantLib's source are required.

### 2.4 JIT Compilation with Forge

For workflows requiring repeated evaluation over many scenarios (Monte Carlo, XVA, stress testing), QuantLibAAD also supports a **JIT pipeline** via the optional [xad-forge](https://github.com/da-roth/xad-forge) library:

1. **Stage 1 (Tape mode)**: Curve bootstrapping using XAD's tape. Produces a Jacobian from market quotes → curve discount factors.
2. **Stage 2 (JIT mode)**: Monte Carlo pricing compiled to native machine code (optionally AVX-vectorised). Produces sensitivities from discount factors → swaption price.
3. **Stage 3 (Chain rule relay)**: The two Jacobians are combined via chain rule to get $\frac{\partial \text{price}}{\partial \text{market quotes}}$.

---

## 3. Repository Layout

```
QuantLibAAD/
├── CMakeLists.txt                   # Top-level CMake (add_subdirectory inside QuantLib)
├── cmake/
│   └── QuantLibAADConfig.cmake.in   # Config for find_package(QuantLibAAD)
├── ql/
│   ├── CMakeLists.txt               # Defines the INTERFACE target QuantLibAAD
│   └── qlaad.hpp                    # The single adaptor header — heart of the integration
├── Examples/
│   ├── CMakeLists.txt
│   ├── AdjointEuropeanEquityOption/ # Black-Scholes Greeks via AAD
│   ├── AdjointAmericanEquityOption/ # American option (finite-difference engine) + AAD
│   ├── AdjointBermudanSwaption/     # Short-rate model calibration + swaption pricing
│   ├── AdjointCDS/                  # Credit Default Swap sensitivities
│   ├── AdjointHestonModel/          # Heston model calibration + pricing
│   ├── AdjointMulticurveBootstrapping/ # Dual-curve bootstrapping sensitivities
│   ├── AdjointReplication/          # Replication strategy sensitivities
│   └── AdjointSwap/                 # Vanilla swap DV01 via curve bootstrapping
├── test-suite/
│   ├── CMakeLists.txt
│   ├── europeanoption_xad.cpp       # Tests: European option Greeks
│   ├── americanoption_xad.cpp       # Tests: American option
│   ├── barrieroption_xad.cpp        # Tests: Barrier option
│   ├── batesmodel_xad.cpp           # Tests: Bates stochastic vol model
│   ├── bermudanswaption_xad.cpp     # Tests: Bermudan swaption
│   ├── bonds_xad.cpp                # Tests: Bond sensitivities
│   ├── creditdefaultswap_xad.cpp    # Tests: CDS
│   ├── forwardrateagreement_xad.cpp # Tests: FRA sensitivities
│   ├── hestonmodel_xad.cpp          # Tests: Heston model
│   ├── swap_xad.cpp                 # Tests: Vanilla swap
│   ├── swaption_jit_pipeline_xad.cpp# Tests: JIT hybrid pipeline (requires Forge)
│   ├── benchmark_aad.cpp            # AAD performance benchmark (tape + JIT)
│   ├── benchmark_fd.cpp             # FD baseline benchmark
│   ├── benchmark_common.hpp         # Shared benchmark utilities and config
│   ├── benchmark_pricing.hpp        # Swaption pricing primitives
│   ├── utilities_xad.hpp/cpp        # Test helpers, BOOST_CHECK wrappers for AReal
│   └── toplevelfixture.hpp          # Boost.Test top-level fixture
└── presets/
    ├── CMakeUserPresets.json         # Presets for QuantLib >= 1.32
    ├── CMakeUserPresets-1.31.json    # Presets for QuantLib 1.31 and earlier
    └── README.md                    # How to use presets
```

---

## 4. Dependencies

| Dependency | Role | Where to get |
|---|---|---|
| **QuantLib** (master / ≥ 1.28) | Financial pricing library | https://github.com/lballabio/QuantLib |
| **XAD** (C++ library) | Automatic differentiation engine | https://github.com/auto-differentiation/xad |
| **Boost** (≥ 1.58, headers + `unit_test_framework`) | Used by both QuantLib and the test suite | https://www.boost.org |
| **CMake** (≥ 3.15) | Build system | https://cmake.org |
| **C++14** compiler | GCC ≥ 9, Clang ≥ 10, MSVC 2019+ | — |
| **xad-forge** *(optional)* | JIT backend for Forge pipeline | https://github.com/da-roth/xad-forge |

> **Important:** QuantLibAAD is **not** a stand-alone project. It must be compiled as a subdirectory added to QuantLib's build via `QL_EXTERNAL_SUBDIRECTORIES`.

---

## 5. Building the Project

### 5.1 Prerequisites

Clone all three repositories side-by-side:

```bash
mkdir ~/quant && cd ~/quant
git clone https://github.com/lballabio/QuantLib.git
git clone https://github.com/auto-differentiation/xad.git
git clone https://github.com/auto-differentiation/QuantLibAAD.git
```

Expected directory structure:

```
~/quant/
├── QuantLib/       ← the main QuantLib source
├── xad/            ← XAD automatic differentiation library
└── QuantLibAAD/    ← this repository
```

**Optional (for JIT support):** If you plan to use the Forge JIT backend, also clone the Forge and xad-forge repositories (see [Section 5.7](#57-building-with-xad-forge-optional-jit-backend)).

Install Boost (headers + compiled libraries):

```bash
# Ubuntu / Debian
sudo apt-get install libboost-all-dev

# macOS (Homebrew)
brew install boost

# Windows: download from https://www.boost.org/users/download/
```

### 5.2 Directory Layout Convention

The CMake presets and the `_xad-base` configuration preset assume this relative layout:

```json
"QL_EXTERNAL_SUBDIRECTORIES": "${sourceDir}/../xad;${sourceDir}/../QuantLibAAD"
```

where `${sourceDir}` is the QuantLib root. This is why the side-by-side clone structure matters.

### 5.3 Build with CMake Presets (Recommended)

Copy the preset file into the QuantLib directory:

```bash
# For QuantLib >= 1.32
cp ~/quant/QuantLibAAD/presets/CMakeUserPresets.json ~/quant/QuantLib/

# For QuantLib 1.31
cp ~/quant/QuantLibAAD/presets/CMakeUserPresets-1.31.json ~/quant/QuantLib/CMakeUserPresets.json
```

Then configure and build from the QuantLib root using one of the available presets:

```bash
cd ~/quant/QuantLib

# Linux with GCC, Release build
cmake --preset linux-xad-gcc-release
cd build/linux-xad-gcc-release
cmake --build . --parallel $(nproc)
```

**Available presets (Linux):**

| Preset | Compiler | Config |
|---|---|---|
| `linux-xad-gcc-release` | GCC | Release |
| `linux-xad-gcc-debug` | GCC | Debug |
| `linux-xad-gcc-relwithdebinfo` | GCC | RelWithDebInfo |
| `linux-xad-clang-release` | Clang | Release |
| `linux-xad-clang-debug` | Clang | Debug |
| `linux-xad-gcc-ninja-release` | GCC + Ninja | Release |
| `linux-xad-clang-ninja-release` | Clang + Ninja | Release |

Each preset also has a `-noxad-` variant (e.g. `linux-noxad-gcc-release`) that sets `QLAAD_DISABLE_AAD=ON`, compiling with plain `double` for benchmarking purposes.

**Available presets (Windows):**

| Preset | Compiler | Config |
|---|---|---|
| `windows-xad-msvc-release` | MSVC | Release |
| `windows-xad-msvc-debug` | MSVC | Debug |
| `windows-xad-clang-release` | Clang-cl | Release |
| `windows-xad-clang-debug` | Clang-cl | Debug |

### 5.4 Manual CMake Build

If you prefer full control over the CMake invocation, configure from the QuantLib source directory:

```bash
cd ~/quant/QuantLib

cmake -B build/xad-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_EXTERNAL_SUBDIRECTORIES="$(pwd)/../xad;$(pwd)/../QuantLibAAD" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLibAAD \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_STATIC_MSVC_RUNTIME=ON \
  -DXAD_NO_THREADLOCAL=ON \
  -DQLAAD_DISABLE_AAD=OFF

cmake --build build/xad-release --parallel $(nproc)
```

**Key variables explained:**

| Variable | Purpose |
|---|---|
| `QL_EXTERNAL_SUBDIRECTORIES` | Tells QuantLib to add XAD and QuantLibAAD as subdirectories. Order matters: XAD before QuantLibAAD. |
| `QL_EXTRA_LINK_LIBRARIES` | Links `QuantLibAAD` into the QuantLib library target |
| `QL_NULL_AS_FUNCTIONS` | Required to avoid macro collisions with XAD's expression templates |
| `XAD_STATIC_MSVC_RUNTIME` | Ensures compatible MSVC runtime on Windows |
| `XAD_NO_THREADLOCAL` | Disables thread-local tape storage for compatibility |
| `QLAAD_DISABLE_AAD` | Set `OFF` for XAD mode, `ON` for plain double mode |

### 5.5 Windows / Visual Studio

Use Visual Studio's "Open Folder" mode pointing to the QuantLib folder. After placing `CMakeUserPresets.json` there, the XAD presets appear in the configuration dropdown in the top toolbar. Select the desired preset and build normally.

For command-line builds on Windows:

```cmd
cd C:\quant\QuantLib
cmake --preset windows-xad-msvc-release
cmake --build build\windows-xad-msvc-release --config Release
```

### 5.6 CMake Options Reference

| CMake Option | Default | Description |
|---|---|---|
| `QLAAD_DISABLE_AAD` | `OFF` | When `ON`, uses plain `double` for QuantLib's Real (no XAD), allows comparing against FD |
| `QLAAD_ENABLE_FORGE` | `OFF` | Enable the optional xad-forge JIT backend |
| `QLAAD_USE_FORGE_CAPI` | `OFF` | Use Forge's C API for binary compatibility (instead of C++ API) |
| `QLAAD_BUILD_TEST_SUITE` | `OFF` | Build the QuantLibAAD test suite executable |
| `QLAAD_BUILD_BENCHMARK_FD` | `OFF` | Build the finite differences benchmark executable |
| `QLAAD_BUILD_BENCHMARK_AAD` | `OFF` | Build the AAD tape/JIT benchmark executable |
| `QLAAD_ENABLE_FORGE_TESTS` | `OFF` | Enable Forge JIT tests in the test suite |

### 5.7 Building with xad-forge (Optional JIT Backend)

The [xad-forge](https://github.com/da-roth/xad-forge) library provides compiled JIT backends for XAD, replacing the default interpreter with native x86-64 machine code. This is beneficial for repeated-evaluation workflows such as Monte Carlo simulation, risk scenarios, and XVA calculations (see [Section 2.4](#24-jit-compilation-with-forge) and [Section 8.6](#86-jit-pipeline-architecture-forge)).

xad-forge is **not** part of the XAD source tree — it is a separate project with its own dependency on the [Forge](https://github.com/da-roth/forge) JIT compiler.

#### Additional Dependencies

| Dependency | Role | Where to get |
|---|---|---|
| **Forge** | JIT compiler engine (provides the `forge_capi` target) | https://github.com/da-roth/forge |
| **xad-forge** | Bridges Forge into XAD's JIT interface | https://github.com/da-roth/xad-forge |
| **CMake** ≥ 3.20 | Required by xad-forge | https://cmake.org |

#### Clone the Additional Repositories

```bash
cd ~/quant
git clone https://github.com/da-roth/forge.git
git clone https://github.com/da-roth/xad-forge.git
```

Updated directory structure:

```
~/quant/
├── QuantLib/       ← QuantLib source
├── xad/            ← XAD automatic differentiation library
├── QuantLibAAD/    ← this repository
├── forge/          ← Forge JIT compiler
└── xad-forge/      ← xad-forge bridge library
```

#### Build Approach: Subdirectory Mode

The recommended approach is to add Forge, xad-forge, and XAD together so that all targets are available when QuantLibAAD configures. Include them in `QL_EXTERNAL_SUBDIRECTORIES` in the correct order — Forge first, then XAD (with JIT enabled), then xad-forge, then QuantLibAAD:

```bash
cd ~/quant/QuantLib

cmake -B build/xad-forge-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_EXTERNAL_SUBDIRECTORIES="$(pwd)/../forge/api/c;$(pwd)/../xad;$(pwd)/../xad-forge;$(pwd)/../QuantLibAAD" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLibAAD \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_ENABLE_JIT=ON \
  -DQLAAD_DISABLE_AAD=OFF \
  -DQLAAD_ENABLE_FORGE=ON \
  -DQLAAD_USE_FORGE_CAPI=ON

cmake --build build/xad-forge-release --parallel $(nproc)
```

**Key additions compared to a standard build:**

| Variable | Purpose |
|---|---|
| `forge/api/c` in `QL_EXTERNAL_SUBDIRECTORIES` | Adds the Forge C API (`forge_capi` target) |
| `xad-forge` in `QL_EXTERNAL_SUBDIRECTORIES` | Adds the xad-forge bridge library (`XADForge::xad-forge` target) |
| `XAD_ENABLE_JIT=ON` | Enables XAD's JIT compiler infrastructure (`xad::JITCompiler`) |
| `QLAAD_ENABLE_FORGE=ON` | Tells QuantLibAAD to look for xad-forge and create the `QLAAD::forge` target |
| `QLAAD_USE_FORGE_CAPI=ON` | Uses Forge's C API for binary compatibility across compilers |

When `QLAAD_ENABLE_FORGE=ON`, QuantLibAAD's CMake will:
1. Look for an existing `xad-forge` target (subdirectory mode) or try `find_package(xad-forge)` (pre-built mode)
2. Create the `QLAAD::forge` / `qlaad-forge` INTERFACE target that wraps `XADForge::xad-forge`
3. Add the `QLAAD_HAS_FORGE=1` compile definition for code that conditionally uses Forge

#### Alternative: Pre-built xad-forge

If you prefer to build and install xad-forge separately:

```bash
# Build and install Forge
cd ~/quant/forge
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/quant/install
cmake --build build --parallel $(nproc)
cmake --install build

# Build and install xad-forge
cd ~/quant/xad-forge
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=~/quant/install \
  -DCMAKE_INSTALL_PREFIX=~/quant/install
cmake --build build --parallel $(nproc)
cmake --install build

# Then build QuantLib+XAD+QuantLibAAD with CMAKE_PREFIX_PATH
cd ~/quant/QuantLib
cmake -B build/xad-forge-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=~/quant/install \
  -DQL_EXTERNAL_SUBDIRECTORIES="$(pwd)/../xad;$(pwd)/../QuantLibAAD" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLibAAD \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_ENABLE_JIT=ON \
  -DQLAAD_DISABLE_AAD=OFF \
  -DQLAAD_ENABLE_FORGE=ON \
  -DQLAAD_USE_FORGE_CAPI=ON

cmake --build build/xad-forge-release --parallel $(nproc)
```

#### Running Forge-Enabled Tests and Benchmarks

To build the test suite with Forge JIT tests enabled:

```bash
cmake -B build/xad-forge-release \
  ...                              \
  -DQLAAD_ENABLE_FORGE=ON          \
  -DQLAAD_BUILD_TEST_SUITE=ON      \
  -DQLAAD_ENABLE_FORGE_TESTS=ON    \
  -DQLAAD_BUILD_BENCHMARK_AAD=ON
```

This adds the `swaption_jit_pipeline_xad.cpp` Forge tests to the test suite and enables the Forge JIT and Forge JIT-AVX benchmark modes in `benchmark_aad`.

The xad-forge library provides two backends:

| Backend | Description | Use case |
|---|---|---|
| `ScalarBackend` | Compiles to scalar x86-64 native code | General purpose, replaces XAD's default interpreter |
| `AVXBackend` | Compiles to AVX2 SIMD instructions (4 doubles in parallel) | Batch evaluation (Monte Carlo, scenarios) |

#### Verifying the Build

After building, confirm Forge is available:

```bash
# Run the test suite — Forge tests should appear
./quantlib-aad-test-suite --run_test=SwaptionJitPipelineXadTest --log_level=message

# Run the AAD benchmark with Forge modes
./benchmark-aad --production
# Look for "Forge JIT" and "Forge JIT-AVX" rows in the output
```

### 5.8 Building QuantLib-Risks-Py (Python Bindings)

[QuantLib-Risks-Py](https://github.com/auto-differentiation/QuantLib-Risks-Py) provides Python bindings for QuantLib with automatic differentiation support via XAD. It wraps the C++ [QuantLib-Risks](https://github.com/auto-differentiation/QuantLib-Risks-Cpp) library using SWIG and pybind11, and publishes the result as the `QuantLib-Risks` PyPI package.

#### Quick Install (Pre-built Wheels)

The simplest way to use QuantLib-Risks in Python is to install the pre-built package:

```bash
pip install QuantLib-Risks
```

This provides the same interface as the standard `QuantLib` Python package, plus the ability to compute risks (sensitivities) using automatic differentiation via XAD.

#### Building from Source

Building from source is a two-step process: first build and install the C++ QuantLib-Risks library, then build the Python bindings on top of it.

**Prerequisites:**

- CMake >= 3.15
- A C++ compiler (GCC, Clang, or MSVC)
- Ninja (recommended) or Make
- SWIG
- Python >= 3.8 with development headers
- Boost (headers at minimum)
- [poetry-core](https://pypi.org/project/poetry-core/) and [setuptools](https://pypi.org/project/setuptools/) (for the Python package build)

**Step 1: Clone the repository with submodules**

QuantLib-Risks-Py uses Git submodules for its C++ dependencies (QuantLib, XAD, and QuantLib-Risks-Cpp):

```bash
git clone --recurse-submodules https://github.com/auto-differentiation/QuantLib-Risks-Py.git
cd QuantLib-Risks-Py
```

This creates the following layout inside `lib/`:

```
QuantLib-Risks-Py/
├── lib/
│   ├── QuantLib/              ← QuantLib source (submodule)
│   ├── QuantLib-Risks-Cpp/    ← C++ QuantLib-Risks (submodule)
│   └── XAD/                   ← XAD library (submodule)
├── Python/                    ← Python package source & SWIG bindings
├── SWIG/                      ← SWIG interface files
├── tools/
│   ├── prebuild_ql-risks.sh   ← Linux/macOS build script
│   └── prebuild_ql-risks.bat  ← Windows build script
└── CMakeLists.txt
```

**Step 2: Build the C++ library (QuantLib + XAD + QuantLib-Risks)**

The prebuild script automates configuring and building the C++ library. On Linux/macOS:

```bash
# Set optional environment variables (defaults shown)
export CMAKE_INSTALL_PREFIX=$(pwd)/build/prefix
export QL_PRESET=linux-xad-gcc-ninja-release   # or another CMake preset

bash tools/prebuild_ql-risks.sh
```

On Windows (from a Developer Command Prompt):

```cmd
tools\prebuild_ql-risks.bat
```

The prebuild script performs these steps internally:
1. Configures QuantLib with XAD and QuantLib-Risks-Cpp as external subdirectories using the CMake preset
2. Builds and installs QuantLib with AAD support to `CMAKE_INSTALL_PREFIX`
3. Configures and builds the Python SWIG/pybind11 bindings against the installed library

**Step 3: Build the Python wheel**

After the prebuild completes, build the Python package from the generated build directory:

```bash
cd build/${QL_PRESET}/Python
pip install poetry-core setuptools
pip wheel . --no-build-isolation
```

Or install directly:

```bash
pip install . --no-build-isolation
```

#### Targeting a Specific Python Version

To build for a specific Python version, set `QLR_PYTHON_VERSION` before running the prebuild script:

```bash
export QLR_PYTHON_VERSION=cp312   # Options: cp38, cp39, cp310, cp311, cp312
bash tools/prebuild_ql-risks.sh
```

This passes `-DQLR_PYTHON_VERSION=3.12` (or the corresponding version) to CMake.

#### Building with cibuildwheel (CI/Production)

For producing distributable wheels across multiple platforms and Python versions, the project uses [cibuildwheel](https://cibuildwheel.readthedocs.io/). The CI workflow in `.github/workflows/quantlib-risks.yml` builds wheels for:

| Platform | Build Preset |
|---|---|
| `manylinux_x86_64` | `linux-xad-gcc-ninja-release` |
| `musllinux_x86_64` | `linux-xad-gcc-ninja-release` |
| `macosx_x86_64` | `linux-xad-gcc-ninja-release` |
| `win_amd64` | `windows-xad-msvc-release` |

To replicate locally:

```bash
pip install cibuildwheel
cibuildwheel --platform linux
```

#### Building with xad-forge (JIT Support)

The standard prebuild script does not include Forge JIT support. To build the Python bindings with xad-forge enabled, you need to perform a manual build that adds Forge and xad-forge to `QL_EXTERNAL_SUBDIRECTORIES`.

**Additional prerequisites:**

- [Forge](https://github.com/da-roth/forge) — JIT compiler engine
- [xad-forge](https://github.com/da-roth/xad-forge) — Bridge library connecting Forge to XAD
- CMake ≥ 3.20 (required by xad-forge)

**Step 1: Clone the additional repositories alongside QuantLib-Risks-Py**

```bash
cd QuantLib-Risks-Py
git clone https://github.com/da-roth/forge.git lib/forge
git clone https://github.com/da-roth/xad-forge.git lib/xad-forge
```

Updated layout:

```
QuantLib-Risks-Py/
├── lib/
│   ├── QuantLib/              ← QuantLib source (submodule)
│   ├── QuantLib-Risks-Cpp/    ← C++ QuantLib-Risks (submodule)
│   ├── XAD/                   ← XAD library (submodule)
│   ├── forge/                 ← Forge JIT compiler (manually cloned)
│   └── xad-forge/             ← xad-forge bridge (manually cloned)
├── Python/
├── SWIG/
├── tools/
└── CMakeLists.txt
```

**Step 2: Build the C++ library with Forge**

Instead of using the prebuild script, configure QuantLib manually with all five subdirectories. The order in `QL_EXTERNAL_SUBDIRECTORIES` matters — Forge C API first, then XAD, then xad-forge, then QuantLib-Risks-Cpp:

```bash
cd QuantLib-Risks-Py
export QL_DIR=$(pwd)/lib/QuantLib
export QLRISKS_DIR=$(pwd)/lib/QuantLib-Risks-Cpp
export XAD_DIR=$(pwd)/lib/XAD
export FORGE_CAPI_DIR=$(pwd)/lib/forge/api/c
export XAD_FORGE_DIR=$(pwd)/lib/xad-forge
export QL_PRESET=linux-xad-gcc-ninja-release
export CMAKE_INSTALL_PREFIX=$(pwd)/build/prefix

cd "$QL_DIR"
cp -f "$QLRISKS_DIR/presets/CMakeUserPresets.json" .

cmake --preset "$QL_PRESET" \
  -DCMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX" \
  -DXAD_ENABLE_TESTS=OFF \
  -DQL_BUILD_TEST_SUITE=OFF \
  -DQL_BUILD_EXAMPLES=OFF \
  -DQL_BUILD_BENCHMARK=OFF \
  -DXAD_ENABLE_JIT=ON \
  -DQLAAD_ENABLE_FORGE=ON \
  -DQLAAD_USE_FORGE_CAPI=ON
```

> **Note:** The CMake preset already sets `QL_EXTERNAL_SUBDIRECTORIES` to include XAD and QuantLib-Risks-Cpp. You will need to extend it to also include `$FORGE_CAPI_DIR` and `$XAD_FORGE_DIR`. You can either edit the preset JSON or override via a manual `-B` build (see below).

For a fully manual configure without presets:

```bash
cd "$QL_DIR"
mkdir -p build/$QL_PRESET
cmake -B build/$QL_PRESET -S . -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX" \
  -DQL_EXTERNAL_SUBDIRECTORIES="$FORGE_CAPI_DIR;$XAD_DIR;$XAD_FORGE_DIR;$QLRISKS_DIR" \
  -DQL_EXTRA_LINK_LIBRARIES="QuantLibAAD" \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DXAD_NO_THREADLOCAL=ON \
  -DXAD_ENABLE_JIT=ON \
  -DQLAAD_DISABLE_AAD=OFF \
  -DQLAAD_ENABLE_FORGE=ON \
  -DQLAAD_USE_FORGE_CAPI=ON

cd build/$QL_PRESET
cmake --build . --parallel $(nproc)
cmake --install .
```

**Step 3: Build the Python bindings**

Return to the QuantLib-Risks-Py root and build the Python extension as normal:

```bash
cd "$QLSWIG_DIR"    # (i.e. the QuantLib-Risks-Py root)
mkdir -p build/$QL_PRESET
cd build/$QL_PRESET
cmake -G Ninja ../.. \
  -DCMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**Step 4: Build and install the wheel**

```bash
cd Python
pip install poetry-core setuptools
pip install . --no-build-isolation
```

The resulting `QuantLib-Risks` Python package will include Forge JIT support. Code using `xad.adj_1st.Tape` and `JITCompiler` with Forge backends will work through the Python bindings, provided the Forge shared libraries are available at runtime.

#### Python Package Dependencies

The built `QuantLib-Risks` Python package depends on:
- `xad >= 1.5.2` (the XAD Python bindings)
- `python >= 3.8.1`

#### Usage Example

```python
import QuantLib_Risks as ql
from xad.adj_1st import Tape

with Tape() as t:
    rate = ql.Real(0.2)
    t.registerInput(rate)

    # ... QuantLib pricing code ...
    npv = option.NPV()

    t.registerOutput(npv)
    npv.derivative = 1.0
    t.computeAdjoints()

    print(f"price = {npv}")
    print(f"delta = {rate.derivative}")
```

For full documentation, see <https://auto-differentiation.github.io/quantlib-risks/python/>.

---

## 6. Running the Examples

After a successful build, the example executables are located in your build directory. On Linux with the `linux-xad-gcc-release` preset they are under `build/linux-xad-gcc-release/Examples/<Name>/`.

> All examples output timing information so you can compare AAD performance against bump-and-reprice. To disable AAD and run in bump mode, rebuild with `-DQLAAD_DISABLE_AAD=ON`.

### 6.1 AdjointEuropeanEquityOption

Prices a portfolio of European equity options using the analytic Black-Scholes formula. Computes all Greeks simultaneously via AAD (delta, vega, rho, dividend rho, strike sensitivity).

```bash
./Examples/AdjointEuropeanEquityOption/AdjointEuropeanEquityOption
```

**What it demonstrates:**
- Registering a vector of inputs (rates, vols, strikes, underlyings) on the tape
- Computing a portfolio NPV as a single sum
- Extracting `derivative(x)` for each input after `tape.computeAdjoints()`

**Expected output:** A table of option prices + sensitivities, followed by timing comparison of AAD vs analytic bump.

### 6.2 AdjointSwap

Prices a portfolio of vanilla interest rate swaps bootstrapped from Euribor deposit, FRA, and swap quotes. Computes DV01 and curve sensitivities.

```bash
./Examples/AdjointSwap/AdjointSwap
```

**What it demonstrates:**
- AAD through curve bootstrapping (piecewise yield curve)
- Sensitivity to all market quotes simultaneously
- Comparison of AAD vs finite differences timing

### 6.3 AdjointBermudanSwaption

Calibrates a short-rate model (Hull-White / Black-Karasinski / G2) to market swaption volatilities, then prices a Bermudan swaption, computing sensitivities to the flat rate.

```bash
./Examples/AdjointBermudanSwaption/AdjointBermudanSwaption
```

**What it demonstrates:**
- AAD through model calibration (Levenberg-Marquardt optimization)
- Use of tree/finite-difference swaption engines with XAD

### 6.4 AdjointHestonModel

Calibrates the Heston stochastic volatility model to DAX options market data, then prices options and computes sensitivities to interest rates and dividend yields.

```bash
./Examples/AdjointHestonModel/AdjointHestonModel
```

**What it demonstrates:**
- AAD through complex model calibration
- Sensitivity computation for a 5-parameter stochastic vol model

### 6.5 AdjointCDS

Prices a Credit Default Swap (CDS) bootstrapped from hazard rate quotes. Computes sensitivities to hazard rates and risk-free rates.

```bash
./Examples/AdjointCDS/AdjointCDS
```

**What it demonstrates:**
- AAD through credit term structure bootstrapping
- CS01 (credit spread sensitivity) computation

### 6.6 AdjointAmericanEquityOption

Prices American equity options using the Barone-Adesi-Whaley approximation engine and computes Greeks via AAD.

```bash
./Examples/AdjointAmericanEquityOption/AdjointAmericanEquityOption
```

### 6.7 AdjointMulticurveBootstrapping

Bootstraps a dual-curve setup (Euribor forecasting curve + OIS discounting curve) from market data and computes full Jacobian of the resulting discount factors with respect to all input quotes.

```bash
./Examples/AdjointMulticurveBootstrapping/AdjointMulticurveBootstrapping
```

**What it demonstrates:**
- The most complex bootstrapping scenario — multi-curve with OIS discounting
- A large Jacobian computed in a single AAD sweep

### 6.8 AdjointReplication

Demonstrates AAD on a replication-based pricing strategy.

```bash
./Examples/AdjointReplication/AdjointReplication
```

---

## 7. Running the Test Suite

### 7.1 Enabling Tests

The test suite is not built by default. Enable it by adding `-DQLAAD_BUILD_TEST_SUITE=ON` or `-DQL_BUILD_TEST_SUITE=ON` to your CMake configuration:

```bash
cmake -B build/xad-test \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_EXTERNAL_SUBDIRECTORIES="$(pwd)/../xad;$(pwd)/../QuantLibAAD" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLibAAD \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DQLAAD_BUILD_TEST_SUITE=ON

cmake --build build/xad-test --parallel $(nproc)
```

The test binary is named `quantlib-aad-test-suite`.

### 7.2 Running Tests

```bash
# Run all tests with message-level logging
./quantlib-aad-test-suite --log_level=message

# Run a specific test suite
./quantlib-aad-test-suite --run_test=EuropeanOptionXadTests

# List all available tests
./quantlib-aad-test-suite --list_content

# Run with verbose output
./quantlib-aad-test-suite --log_level=all --report_level=detailed
```

**Test suites available:**

| Test file | Tests |
|---|---|
| `europeanoption_xad.cpp` | AAD vs analytic bump for Black-Scholes Greeks |
| `americanoption_xad.cpp` | American option sensitivities |
| `barrieroption_xad.cpp` | Barrier option sensitivities |
| `batesmodel_xad.cpp` | Bates jump-diffusion model |
| `bermudanswaption_xad.cpp` | Bermudan swaption sensitivities |
| `bonds_xad.cpp` | Fixed income bond sensitivities (DV01, duration) |
| `creditdefaultswap_xad.cpp` | CDS sensitivities (CS01) |
| `forwardrateagreement_xad.cpp` | FRA sensitivities |
| `hestonmodel_xad.cpp` | Heston model Greeks |
| `swap_xad.cpp` | Vanilla swap DV01 |

**Via CTest:**

```bash
cd build/xad-test
ctest --output-on-failure
```

### 7.3 Benchmarks (FD vs AAD)

The repository contains dedicated benchmark executables that compare:
- **Finite Differences (FD)**: plain `double` QuantLib, O(N) bumps per sensitivity set
- **AAD tape**: XAD adjoint, O(1) sweeps
- **Forge JIT**: native-compiled evaluation (optional)
- **Forge JIT-AVX**: AVX2-vectorised native code (optional)

Build both benchmark executables:

```bash
cmake -B build/bench \
  -DCMAKE_BUILD_TYPE=Release \
  -DQL_EXTERNAL_SUBDIRECTORIES="$(pwd)/../xad;$(pwd)/../QuantLibAAD" \
  -DQL_EXTRA_LINK_LIBRARIES=QuantLibAAD \
  -DQL_NULL_AS_FUNCTIONS=ON \
  -DQLAAD_DISABLE_AAD=OFF \
  -DQLAAD_BUILD_BENCHMARK_AAD=ON \
  -DQLAAD_BUILD_BENCHMARK_FD=ON

cmake --build build/bench --parallel $(nproc)
```

Run both benchmarks and compare:

```bash
# Production scenario (45 inputs, 100k Monte Carlo paths)
./benchmark-fd  --production
./benchmark-aad --production

# CVA scenario
./benchmark-fd  --cva
./benchmark-aad --cva

# Run all scenarios
./benchmark-fd  --all
./benchmark-aad --all

# Quick run for testing
./benchmark-aad --quick
```

The benchmark measures and prints mean ± stddev of wall-clock time across multiple runs, allowing quantitative comparison of each approach.

---

## 8. Technical Architecture

### 8.1 Big Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QuantLib Source                             │
│                                                                     │
│  Every .cpp file begins: #include <ql/qlaad.hpp>  (via             │
│                            QL_INCLUDE_FIRST mechanism)             │
│                                                                     │
│  typedef xad::AReal<double>  Real;   (from qlaad.hpp)              │
│                                                                     │
│  Pricing functions operate on Real — the XAD active type           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ links to
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     QuantLibAAD (INTERFACE target)                  │
│                                                                     │
│  ql/qlaad.hpp                                                       │
│   ├── #define QL_REAL xad::AReal<double>                           │
│   ├── Boost.Math specialisations for xad::AReal<double>            │
│   ├── Boost.Accumulators type traits                                │
│   ├── Boost.uBLAS type promotion traits                             │
│   ├── std::is_floating_point / is_arithmetic overrides             │
│   └── Platform fixes (MSVC, Apple Clang)                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ links to
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         XAD::xad                                    │
│                                                                     │
│  xad::AReal<double>     — active scalar type                       │
│  xad::Tape<double>      — computation tape (DAG)                   │
│  derivative(x)          — get/set adjoint of x                     │
│  xad::JITGraph          — (optional) JIT-compiled graph            │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 The `qlaad.hpp` Adaptor Header

The file [ql/qlaad.hpp](ql/qlaad.hpp) is the cornerstone of the integration. It is included **before** any QuantLib header in every translation unit. Its responsibilities are:

1. **Type injection**: `#define QL_REAL xad::AReal<double>` causes QuantLib's `typedef Real` to become the XAD active type.

2. **Boost.Math specialisations**: Extends `boost::math::promote_args`, `boost::math::policies::evaluation`, and individual special functions (`erfc`, `ibeta`, `lgamma`, `tgamma`, etc.) to accept `xad::AReal<double>` and its expression template intermediates (`xad::UnaryExpr`, `xad::BinaryExpr`). This is necessary because Boost's generic templates cannot automatically promote through XAD's lazy expression type.

3. **Boost.Accumulators traits**: Extends `result_of_divides` and `result_of_multiplies` for `AReal` so that accumulator-based statistics work correctly.

4. **Boost.uBLAS traits**: Extends `promote_traits` for `AReal` so that matrix/vector operations in QuantLib's linear algebra layer differentiate correctly.

5. **Type trait overrides**: Registers `AReal<double>` as floating-point and arithmetic (so QuantLib's numeric dispatching works), but *not* as a POD type (which is important for correct destruction semantics).

6. **Platform fixes**:
   - **MSVC**: Imports `xad::sqrt`, `xad::pow`, etc. into the global namespace and patches `std::mt19937` constraints that otherwise break with non-native float types.
   - **Apple Clang**: Imports math functions into the `std::_LIBCPP_ABI_NAMESPACE` namespace used internally by libc++.

### 8.3 CMake Targets

| Target | Type | Purpose |
|---|---|---|
| `QuantLibAAD` | INTERFACE library | Carries include dirs, compile definitions (`QL_INCLUDE_FIRST`), and links to `XAD::xad` |
| `QuantLib-Risks` | ALIAS → `QuantLibAAD` | Backward-compatibility alias |
| `qlaad-forge` | INTERFACE library | Optional; wraps `XADForge::xad-forge`, adds `QLAAD_HAS_FORGE=1` define |
| `QLAAD::forge` | ALIAS → `qlaad-forge` | Clean namespaced alias for Forge |
| `QuantLibAAD_test_suite` | Executable | The test binary (`quantlib-aad-test-suite`) |
| `benchmark_fd` | Executable | FD baseline benchmark |
| `benchmark_aad` | Executable | AAD tape + optional Forge benchmark |

The `QuantLibAAD` INTERFACE target is designed to be linked *alongside* the QuantLib library:

```cmake
target_link_libraries(myapp PRIVATE ql_library QuantLibAAD)
```

### 8.4 The XAD Tape Model

XAD uses an **operator-overloading** approach with a global tape. Each arithmetic operation on an `AReal` variable appends an entry to the tape recording:
- The operation (add, mul, exp, log, ...)
- The indices of the operands
- The partial derivatives of the operation w.r.t. each operand

The reverse sweep traverses this tape backwards, propagating adjoint values from outputs to inputs via the chain rule. This is the classic *reverse-mode* AD approach.

**Key tape operations:**

```cpp
using tape_type = xad::Tape<double>;
tape_type tape;

// 1. Register inputs — these are the variables we want ∂f/∂x for
tape.registerInput(x1);
tape.registerInput(x2);
// or: tape.registerInputs(my_vector);

// 2. Start recording
tape.newRecording();

// 3. Compute forward — all operations on AReal values are recorded
xad::AReal<double> result = f(x1, x2);   // any QuantLib pricing function

// 4. Register output and seed the adjoint
tape.registerOutput(result);
derivative(result) = 1.0;

// 5. Reverse sweep
tape.computeAdjoints();

// 6. Read sensitivities
double sens_x1 = derivative(x1);
double sens_x2 = derivative(x2);

// 7. Clear for next run
tape.clearAll();
```

### 8.5 AAD Workflow Pattern

All examples follow the same three-function pattern:

```
priceOnly(inputs)           → double         (base price, no AD)
priceWithSensi(inputs, ...) → double + sensi  (AAD path)
priceWithBumping(inputs,...) → double + sensi  (FD path, for validation or QLAAD_DISABLE_AAD)
```

The `main()` function calls both, compares timings, and optionally validates that the AAD and bumped sensitivities agree.

### 8.6 JIT Pipeline Architecture (Forge)

The JIT pipeline (in `swaption_jit_pipeline_xad.cpp` and the AAD benchmark) implements a hybrid three-stage workflow:

```
Market Quotes (double)
        │
        │ Stage 1: XAD tape recording
        ▼
┌───────────────────────┐
│  Curve Bootstrapping  │  xad::AReal<double>
│  (PiecewiseYieldCurve)│
└───────────────────────┘
        │
        │  Jacobian: ∂(discount factors) / ∂(market quotes)
        │  [numIntermediates × numInputs] — from tape
        ▼
┌───────────────────────┐
│   Intermediate        │  ZeroCurve node values
│   Variables           │
└───────────────────────┘
        │
        │ Stage 2: Forge JIT compilation + evaluation
        ▼
┌───────────────────────┐
│  Monte Carlo Swaption │  Forge-compiled native code
│  Pricing (LMM)        │  (optionally AVX2 vectorised)
└───────────────────────┘
        │
        │  Jacobian: ∂(price) / ∂(discount factors)
        │  [1 × numIntermediates] — from Forge
        ▼
┌───────────────────────┐
│  Chain Rule Relay     │  result[j] = Σᵢ dPrice/dInter[i] · dInter[i]/dInput[j]
└───────────────────────┘
        │
        ▼
Full sensitivities: ∂(price) / ∂(market quotes)
```

This hybrid approach is beneficial when the Monte Carlo simulation has thousands of paths but the number of market inputs is moderate. The JIT graph is compiled once and re-executed per scenario, amortising the compilation cost.

### 8.7 Dual-Mode Operation

Every example and test supports two build modes:

| Mode | `QLAAD_DISABLE_AAD` | `QL_REAL` | Sensitivities |
|---|---|---|---|
| **AAD** (default) | `OFF` | `xad::AReal<double>` | Exact, computed via adjoint sweep |
| **No-AAD** | `ON` | `double` (QuantLib default) | Approximate, computed via bump-and-reprice |

The `#ifndef QLAAD_DISABLE_AAD` / `#else` pattern in source files selects the appropriate implementation at compile time — the no-AAD path gives identical pricing output and can be used to benchmark or validate independently of XAD.

---

## 9. Deep Dive: How `qlaad.hpp` Works

The file addresses a fundamental challenge: QuantLib and the Boost libraries it depends on were written for `double`. Making them work with a custom type like `xad::AReal<double>` requires type system surgery in several places.

### Expression templates vs eager evaluation

XAD uses *expression templates* — intermediate results of `a + b * c` are not immediately computed as `AReal` values; instead, a lazy `BinaryExpr<double, MulOp, AReal<double>, AReal<double>>` object is created. This avoids unnecessary temporary `AReal` allocations and keeps the tape tight.

However, many Boost template functions (e.g. `boost::math::erfc`) have specialisations only for specific types. `qlaad.hpp` adds overloads that pattern-match on `xad::UnaryExpr<...>` and `xad::BinaryExpr<...>` and materialise them into `AReal<double>` before forwarding:

```cpp
template <class Op, class Expr, class Policy>
inline xad::AReal<double> erfc(xad::UnaryExpr<double, Op, Expr> z, const Policy& pol) {
    return boost::math::erfc(xad::AReal<double>(z), pol);   // materialise then call
}
```

### Type trait injection

Boost's `is_floating_point<T>` and `is_arithmetic<T>` traits are `false` for `AReal<double>` by default. QuantLib uses these to select numeric implementations. The overrides:

```cpp
template <> struct is_floating_point<xad::AReal<double>> : public true_type {};
template <> struct is_arithmetic  <xad::AReal<double>> : public true_type {};
```

make QuantLib treat `AReal<double>` identically to `double` for dispatch purposes.

### Convertibility restriction

QuantLib's `numeric_cast<Target>()` must not silently strip the active part from an `AReal` to a plain type. The override:

```cpp
template <class To>
struct is_convertible<xad::AReal<double>, To> : public false_type {};
```

prevents accidental implicit conversion. An explicit `value(x)` is required to extract the double value, making such conversions visible in the source code.

---

## 10. Writing Your Own AAD-Enabled Code

Once QuantLibAAD is built, writing new pricing code that computes sensitivities follows the same pattern:

```cpp
#include <ql/qldefines.hpp>
#include <ql/quantlib.hpp>      // Real is now xad::AReal<double>
#include <XAD/XAD.hpp>          // for tape operations
#include <vector>

using namespace QuantLib;

// Standard pricing function — write it exactly as you would with doubles
Real myPricer(const std::vector<Real>& inputs) {
    // ... QuantLib pricing code using Real arithmetic ...
    return npv;
}

int main() {
    std::vector<double> inputValues = { 0.05, 0.2, 100.0 };

    // Set up tape and inputs
    using tape_type = Real::tape_type;
    tape_type tape;

    std::vector<Real> inputs(inputValues.begin(), inputValues.end());
    tape.registerInputs(inputs);
    tape.newRecording();

    // Price
    Real price = myPricer(inputs);

    // Reverse sweep
    tape.registerOutput(price);
    derivative(price) = 1.0;
    tape.computeAdjoints();

    // Read sensitivities
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << "dPrice/dInput[" << i << "] = " << derivative(inputs[i]) << "\n";
    }
    return 0;
}
```

**CMake integration for your own project:**

```cmake
find_package(QuantLibAAD REQUIRED)     # finds XAD and QuantLib transitively

add_executable(myapp main.cpp)
target_link_libraries(myapp
    PRIVATE QuantLib::QuantLib
            QuantLibAAD::QuantLibAAD)
```

---

## 11. Common Issues and Troubleshooting

### `XAD::xad` not found during CMake configure

**Cause:** XAD was not cloned or the path in `QL_EXTERNAL_SUBDIRECTORIES` is wrong.

**Fix:** Ensure `xad/` is a sibling of `QuantLib/` and `QuantLibAAD/`, or adjust the path in `CMakeUserPresets.json` under `QL_EXTERNAL_SUBDIRECTORIES`.

### Compilation errors about `squared` or template ambiguity

**Cause:** `qlaad.hpp` was not included first — QuantLib headers were pulled in before the type substitution took effect.

**Fix:** Ensure `QL_INCLUDE_FIRST=ql/qlaad.hpp` is set. This is done automatically by the `QuantLibAAD` CMake target. If building manually, add `-DQL_INCLUDE_FIRST=ql/qlaad.hpp` to the compiler flags.

### `QL_NULL_AS_FUNCTIONS` must be ON

**Cause:** QuantLib by default defines `Null<T>()` as a macro. XAD's expression templates conflict with this.

**Fix:** Always pass `-DQL_NULL_AS_FUNCTIONS=ON` when configuring QuantLib with XAD.

### Test suite not built

**Cause:** Tests are opt-in.

**Fix:** Add `-DQLAAD_BUILD_TEST_SUITE=ON` or `-DQL_BUILD_TEST_SUITE=ON` to the CMake configuration.

### Sensitivities are zero after `computeAdjoints()`

**Cause 1:** `tape.newRecording()` was not called before the forward computation.

**Cause 2:** Inputs were not registered with `tape.registerInput()` before `newRecording()`.

**Cause 3:** The output was not registered with `tape.registerOutput()` and its adjoint was not seeded (`derivative(price) = 1.0`).

**Fix:** Follow the 7-step tape workflow exactly as shown in [Section 8.4](#84-the-xad-tape-model).

### `MSVC` `/bigobj` compilation failures

**Cause:** The `qlaad.hpp` specialisations generate a large number of symbols.

**Fix:** The `QuantLibAAD` CMake target already adds `/bigobj` for MSVC. Ensure you are linking against the target rather than setting flags manually.

### Forge tests fail to compile

**Cause:** `QLAAD_ENABLE_FORGE_TESTS=ON` but `xad-forge` is not available.

**Fix:** Either set up xad-forge as described in its README, or leave `QLAAD_ENABLE_FORGE_TESTS=OFF`.

---

## 12. License Summary

QuantLibAAD uses two different licenses depending on the subdirectory:

| Directory | License | Reason |
|---|---|---|
| `ql/` | GNU AGPL v3 | New XAD adaptor code by Xcelerit |
| `Examples/` | QuantLib license (BSD-like) | Derived from QuantLib example code |
| `test-suite/` | QuantLib license (BSD-like) | Derived from QuantLib test code |

The dual-license structure reflects that the adaptor code is original work (AGPL) while the examples and tests are modifications of QuantLib's own Apache/BSD-licensed examples.

Each subdirectory contains its own `LICENSE` or `LICENSE.TXT` / `LICENSE.md` file to make the applicable license explicit.

---

*Last updated: February 2026. For the latest information, see the [official documentation](https://auto-differentiation.github.io/quantlib-risks/cxx/) and the [GitHub repository](https://github.com/auto-differentiation/QuantLibAAD).*
