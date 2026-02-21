# Open-Source C++ Automatic Differentiation Libraries

A comprehensive survey and comparison of the major open-source C++ AD implementations.

---

## 1. Overview

There are two fundamentally different implementation strategies for AD in C++:

| Strategy                  | How it works                                                                 | Examples                                      |
| ------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------- |
| **Operator Overloading**  | Replace `double` with a custom type that records operations via overloaded `+`, `*`, `sin()`, etc. | CppAD, ADOL-C, CoDiPack, Adept, autodiff, XAD, Sacado, Stan Math |
| **Source Transformation** | A compiler plugin or preprocessor analyzes the code and generates derivative code at compile time. | Enzyme, Tapenade (Fortran/C)                  |

Operator overloading is easier to integrate but has runtime overhead from the tape. Source transformation can be faster (compiler optimizations apply to derivative code) but is harder to set up.

---

## 2. The Libraries

### 2.1 CppAD

- **Repository:** [github.com/coin-or/CppAD](https://github.com/coin-or/CppAD) ⭐ ~560
- **License:** EPL-2.0 / GPL-2.0-or-later
- **Origin:** Brad Bell (COIN-OR project)
- **First release:** ~2003
- **C++ standard:** C++11+

**Description:**
One of the oldest and most mature AD libraries in C++. CppAD records operations on `AD<double>` types into an operation sequence (tape), then can compute forward-mode, reverse-mode, sparse Jacobians, sparse Hessians, and higher-order derivatives. It includes built-in graph optimization (dead-code elimination, common subexpression elimination on the tape).

**Key features:**
- Forward and reverse mode (arbitrary order)
- Sparse Jacobian/Hessian computation with graph coloring (via ColPack integration)
- Tape optimization (constant folding, dead-code removal)
- Checkpoint/re-taping for memory control
- Conditional expressions without re-taping (`CondExpLt`, etc.)
- Atomic functions (user-defined derivatives)
- Dynamic parameters (change parameter values without re-taping)
- Comprehensive documentation and extensive test suite
- Used in IPOPT (nonlinear optimizer) ecosystem

**Typical usage:**
```cpp
#include <cppad/cppad.hpp>
using ADdouble = CppAD::AD<double>;

std::vector<ADdouble> x(2);
x[0] = 2.0; x[1] = 3.0;

CppAD::Independent(x);               // start recording
std::vector<ADdouble> y(1);
y[0] = x[0] * x[1] + sin(x[0]);     // operations recorded
CppAD::ADFun<double> f(x, y);        // stop recording, create function object

std::vector<double> grad = f.Jacobian({2.0, 3.0});
// grad = { 3 + cos(2), 2 }
```

---

### 2.2 ADOL-C

- **Repository:** [github.com/coin-or/ADOL-C](https://github.com/coin-or/ADOL-C) ⭐ ~175
- **License:** EPL / GPL
- **Origin:** Technical University of Dresden (Andrea Walther, Andreas Griewank)
- **First release:** ~1996
- **C++ standard:** C++11+ (gcc-11, clang-13, MSVC 19.31 minimum)

**Description:**
One of the pioneering AD tools, developed by the group behind the foundational textbook *"Evaluating Derivatives"* by Griewank & Walther. ADOL-C uses a tape-based approach where operations on `adouble` types are recorded to numbered tapes. It provides a rich set of "drivers" for computing gradients, Jacobians, and Hessians.

**Key features:**
- Forward and reverse mode
- Tape-based with multiple tape support
- Sparse derivative computation with ColPack integration
- Higher-order derivative tensors (Taylor coefficients)
- Trace/untrace mechanism for repeated evaluation
- MeDiPack support (MPI-parallel AD)
- Activity tracking to reduce trace size
- Advanced branching to reduce re-taping
- Drivers: `gradient()`, `jacobian()`, `hessian()`, `tensor_eval()`
- C interface available
- Python bindings via SWIG

**Typical usage:**
```cpp
#include <adolc/adolc.h>

adouble x1, x2, y;
double out;

auto tapeId = createNewTape();
trace_on(tapeId);
  x1 <<= 2.0;  x2 <<= 3.0;         // mark independent
  y = x1 * x2 + sin(x1);
  y >>= out;                         // mark dependent
trace_off();

double grad[2];
gradient(tapeId, 2, (double[]){2.0, 3.0}, grad);
// grad = { 3 + cos(2), 2 }
```

---

### 2.3 CoDiPack

- **Repository:** [github.com/SciCompKL/CoDiPack](https://github.com/SciCompKL/CoDiPack) ⭐ ~110
- **License:** GPL-3.0 (other licenses available on request)
- **Origin:** TU Kaiserslautern / RPTU (Scientific Computing Group)
- **First release:** ~2018
- **C++ standard:** C++17 (C++11 for version 2.x)

**Description:**
CoDiPack (Code Differentiation Package) is a modern, high-performance AD library using expression templates. It is designed for ease of use while giving experienced AD developers full access to internal data structures. It is the successor to the AD capabilities originally in the SU2 CFD solver.

**Key features:**
- Forward mode (`codi::RealForward`)
- Multiple reverse-mode tape implementations:
  - `codi::RealReverse` — general purpose, C-compatible
  - `codi::RealReverseIndex` — reduced tape size
  - `codi::RealReversePrimal` — further reduced tape size
  - `codi::RealReversePrimalIndex` — most compact
- Higher-order derivatives (nesting of types)
- Expression templates for performance (eliminates temporaries)
- External functions (user-defined adjoint code)
- AdjointMPI interface for parallel AD
- Complex number support
- Header-only library
- Excellent tape memory efficiency

**Typical usage:**
```cpp
#include <codi.hpp>

codi::RealReverse x1 = 2.0, x2 = 3.0;

codi::RealReverse::Tape& tape = codi::RealReverse::getTape();
tape.setActive();
tape.registerInput(x1);
tape.registerInput(x2);

codi::RealReverse y = x1 * x2 + sin(x1);

tape.registerOutput(y);
tape.setPassive();
y.setGradient(1.0);
tape.evaluate();

// x1.getGradient() == 3 + cos(2)
// x2.getGradient() == 2
tape.reset();
```

---

### 2.4 Adept 2

- **Repository:** [github.com/rjhogan/Adept-2](https://github.com/rjhogan/Adept-2) ⭐ ~185
- **License:** Apache-2.0
- **Origin:** Robin Hogan (ECMWF / University of Reading)
- **First release:** ~2014
- **C++ standard:** C++11+

**Description:**
Adept (Automatic Differentiation using Expression Templates) was designed for meteorological and atmospheric science applications. Version 2 extends the original with array support (vectors, matrices, up to 7-D arrays) and optimization algorithms (Levenberg-Marquardt, L-BFGS, conjugate gradient).

**Key features:**
- Reverse mode (primary focus)
- Forward mode (since v2)
- Expression templates for high performance
- Built-in N-dimensional array library (vectors, matrices, up to 7-D)
- Built-in optimization algorithms with box constraints
- Very fast reverse-mode Jacobian computation
- Stack-based storage (cache-friendly)
- Automatic memory management
- BLAS/OpenBLAS integration for linear algebra
- Well-documented with a formal User Guide
- Benchmarked as one of the fastest reverse-mode OO implementations

**Typical usage:**
```cpp
#include <adept.h>
using adept::adouble;

adept::Stack stack;
adouble x1 = 2.0, x2 = 3.0;

stack.new_recording();
x1.set_value(2.0); x2.set_value(3.0);
adouble y = x1 * x2 + sin(x1);

y.set_gradient(1.0);
stack.reverse();

// x1.get_gradient() == 3 + cos(2)
// x2.get_gradient() == 2
```

---

### 2.5 autodiff

- **Repository:** [github.com/autodiff/autodiff](https://github.com/autodiff/autodiff) ⭐ ~1.9k
- **License:** MIT
- **Origin:** Allan Leal (ETH Zürich)
- **First release:** ~2018
- **C++ standard:** C++17

**Description:**
A modern, header-only C++17 library with the most user-friendly API of all the C++ AD libraries. It uses advanced template metaprogramming to provide `dual` types for forward mode and `var` types for reverse mode, with a clean `derivative(f, wrt(x), at(x, y))` syntax.

**Key features:**
- Forward mode with `dual`, `dual2nd`, `dual3rd`, `dual4th` (higher-order)
- Reverse mode with `var`
- Very clean, intuitive API (`wrt()`, `at()`, `derivative()`, `gradient()`, `jacobian()`, `hessian()`)
- Header-only
- Eigen integration for vector/matrix derivatives
- Supports real and complex numbers
- Comprehensive examples and documentation website
- Suitable for rapid prototyping and teaching
- CUDA support for GPU computation

**Typical usage:**
```cpp
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

dual f(dual x1, dual x2) {
    return x1 * x2 + sin(x1);
}

dual x1 = 2.0, x2 = 3.0;
double dfdx1 = derivative(f, wrt(x1), at(x1, x2));  // 3 + cos(2)
double dfdx2 = derivative(f, wrt(x2), at(x1, x2));  // 2
```

---

### 2.6 Enzyme

- **Repository:** [github.com/EnzymeAD/Enzyme](https://github.com/EnzymeAD/Enzyme) ⭐ ~1.5k
- **License:** Apache-2.0 with LLVM exception
- **Origin:** MIT (William Moses, Valentin Churavy)
- **First release:** ~2020
- **Approach:** Compiler plugin (LLVM/MLIR), **not** operator overloading

**Description:**
Enzyme is fundamentally different from all the other libraries here. It is an LLVM compiler plugin that performs AD on the LLVM intermediate representation (IR). This means it differentiates **optimized** code — the compiler first optimizes your program, then Enzyme differentiates the optimized IR. This avoids the "differentiate then optimize" vs. "optimize then differentiate" problem entirely.

**Key features:**
- Forward and reverse mode
- Operates at the LLVM IR level (language-agnostic: C, C++, Rust, Julia, Fortran)
- Differentiates **after** compiler optimizations (uniquely high performance)
- No code modification needed (no type changes, no tape, no recording)
- GPU support (CUDA, ROCm)
- Parallel AD (OpenMP, MPI, Julia Tasks)
- MLIR support for ML frameworks
- Activity analysis (automatically detects what needs differentiation)
- Cross-language: Julia bindings (Enzyme.jl), Rust bindings
- Meets or exceeds performance of all OO-based AD tools
- Published in NeurIPS 2020, SC '21, SC '22

**Typical usage:**
```cpp
// No special types needed — just plain C++!
double f(double x1, double x2) {
    return x1 * x2 + sin(x1);
}

// Enzyme magic: compiler synthesizes the gradient
double dfdx1 = __enzyme_autodiff((void*)f,
                                  enzyme_dup, 2.0, 1.0,   // x1, seed=1
                                  enzyme_const, 3.0);      // x2, no diff
```

**Compilation:**
```bash
clang++ -fplugin=ClangEnzyme-17.so -O2 example.cpp -o example
```

---

### 2.7 Sacado (Trilinos)

- **Repository:** [github.com/trilinos/Trilinos](https://github.com/trilinos/Trilinos) (packages/sacado/) ⭐ ~1.4k (Trilinos)
- **License:** BSD (package-level)
- **Origin:** Sandia National Laboratories (Eric Phipps)
- **C++ standard:** C++14+

**Description:**
Sacado is the AD package within the Trilinos scientific computing framework. It is heavily used in Sandia's large-scale simulation codes (e.g., Albany, Drekar, Panzer). It provides multiple AD types optimized for different use cases, including a highly optimized forward-mode implementation using expression-level reverse mode.

**Key features:**
- Forward mode (`Sacado::Fad::DFad`, `Sacado::Fad::SFad`, `Sacado::Fad::SLFad`)
  - `DFad` — dynamic allocation (flexible)
  - `SFad` — static allocation (compile-time size, fastest)
  - `SLFad` — static with dynamic fallback
- Reverse mode (`Sacado::Rad`)
- Expression-level reverse mode (ELR) for reduced memory
- Expression templates
- Kokkos integration for GPU/many-core performance
- Nested derivatives (forward-over-forward, forward-over-reverse)
- Deep integration with Trilinos ecosystem (Tpetra, NOX, Tempus, etc.)
- Battle-tested in DOE HPC simulation codes
- SIMD-friendly static forward-mode types

**Typical usage:**
```cpp
#include <Sacado.hpp>
using Fad = Sacado::Fad::DFad<double>;

Fad x1(2, 0, 2.0);  // 2 derivatives, index 0, value 2.0
Fad x2(2, 1, 3.0);  // 2 derivatives, index 1, value 3.0

Fad y = x1 * x2 + sin(x1);
// y.val()  == 6 + sin(2)
// y.dx(0)  == 3 + cos(2)
// y.dx(1)  == 2
```

---

### 2.8 Stan Math

- **Repository:** [github.com/stan-dev/math](https://github.com/stan-dev/math) ⭐ ~807
- **License:** BSD-3-Clause
- **Origin:** Columbia University (Stan development team)
- **First release:** ~2012
- **C++ standard:** C++14+

**Description:**
The Stan Math Library is the AD engine behind the Stan probabilistic programming language. It is specifically designed for Bayesian statistical modeling, providing gradients for Markov Chain Monte Carlo (MCMC) samplers like Hamiltonian Monte Carlo (HMC). It has the largest collection of differentiable mathematical functions of any C++ AD library, especially for probability distributions.

**Key features:**
- Reverse mode (primary, via `stan::math::var`)
- Forward mode (via `stan::math::fvar`)
- Mixed mode (forward-over-reverse for Hessian-vector products)
- Huge function library: 100+ probability distributions, all differentiable
- Special functions (Bessel, beta, gamma, hypergeometric, etc.)
- Matrix derivatives via Eigen integration
- ODE solvers with AD through the solve (BDF, Adams, RK45)
- Algebraic equation solvers with AD
- Laplace approximation
- OpenCL GPU support for select operations
- Intel TBB for parallelism
- Arena-based memory allocator (very fast allocation/deallocation)
- Heavily tested and battle-hardened in production statistical software

---

### 2.9 XAD

- **Repository:** [github.com/auto-differentiation/xad](https://github.com/auto-differentiation/xad) ⭐ ~411
- **License:** AGPL-3.0 (commercial licenses available)
- **Origin:** Xcelerit Computing (quantitative finance roots, open-sourced 2022)
- **First release:** 2014 (closed-source); July 2022 (open-source v1.0.0); Feb 2026 (v2.0.0)
- **C++ standard:** C++11+ (tested with C++11 and C++17)

**Description:**
XAD is a high-performance C++ AD library originally developed for quantitative finance (Greeks computation in derivatives pricing). It was built internally at Xcelerit starting in 2010, inspired by early work from Mike Giles and Paul Glasserman, and open-sourced in 2022. XAD focuses on **low runtime overhead**, **minimal memory footprint**, and **straightforward integration** into existing codebases. Version 2.0 adds an optional JIT backend for record-once / replay-many workflows, ideal for Monte Carlo simulations.

**Key features:**
- Forward mode (`xad::FReal<T>`) and adjoint/reverse mode (`xad::AReal<T>`)
- Higher-order derivatives via type nesting (forward-over-adjoint, adjoint-over-adjoint, etc.)
- **Vector mode** (`FReal<T, N>`, `AReal<T, N>`): compute N derivatives simultaneously in a single pass
- Checkpointing support via `CheckpointCallback` base class (trade memory for compute)
- External function interface (integrate non-AD code with manual adjoints)
- Expression templates for eliminating temporaries
- **JIT backend support** (optional, v2.0): record-once / replay-many execution for Monte Carlo workloads
- Eigen integration for matrix/vector derivatives
- Complex number support
- Smoothed mathematical functions (for handling discontinuities)
- Built-in Jacobian and Hessian helper functions
- Python bindings (XAD-Py)
- QuantLib integration (QuantLibAAD project)
- Thread-safe: each thread can have its own tape
- Comprehensive CI: tested on GCC 7–14, Clang 11–18, MSVC 2019–2022, macOS (Intel & ARM)
- Good documentation with tutorials, API reference, and examples

**Typical usage:**
```cpp
#include <xad/xad.hpp>

using mode = xad::adj<double>;
using Adouble = mode::active_type;
using tape_type = mode::tape_type;

tape_type tape;
Adouble x1 = 2.0, x2 = 3.0;
tape.registerInput(x1);
tape.registerInput(x2);
tape.newRecording();

Adouble y = x1 * x2 + sin(x1);

tape.registerOutput(y);
derivative(y) = 1.0;
tape.computeAdjoints();

// derivative(x1) == 3 + cos(2)
// derivative(x2) == 2
```

---

## 3. Comparison Tables

### 3.1 Feature Comparison

| Feature | CppAD | ADOL-C | CoDiPack | Adept 2 | autodiff | Enzyme | Sacado | Stan Math | XAD |
|---|---|---|---|---|---|---|---|---|---|
| **Forward mode** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Reverse mode** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Higher-order** | ✅ any | ✅ any | ✅ nesting | ❌ | ✅ 4th | ✅ | ✅ nesting | ✅ nesting | ✅ nesting |
| **Sparse Jacobian** | ✅ ColPack | ✅ ColPack | ❌ (manual) | ❌ | ❌ | ❌ (manual) | ❌ | ❌ | ❌ (manual) |
| **Sparse Hessian** | ✅ ColPack | ✅ ColPack | ❌ (manual) | ❌ | ❌ | ❌ (manual) | ❌ | ❌ | ❌ (manual) |
| **Tape optimization** | ✅ | ❌ | ❌ | ❌ | N/A | ✅ (compiler) | ❌ | ❌ | ✅ (JIT, optional) |
| **Expression templates** | ❌ | ❌ | ✅ | ✅ | ✅ | N/A | ✅ | ❌ | ✅ |
| **Conditional (no retape)** | ✅ | Partial | ✅ | ✅ | ✅ | ✅ (native) | ✅ | ✅ | ✅ |
| **External/Atomic funcs** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ (custom) | ✅ | ✅ | ✅ |
| **GPU support** | ❌ | ❌ | ❌ | ❌ | ✅ (CUDA) | ✅ (CUDA/ROCm) | ✅ (Kokkos) | ✅ (OpenCL) | ❌ |
| **MPI parallel AD** | ❌ | ✅ (MeDiPack) | ✅ (AdjointMPI) | ❌ | ❌ | ✅ | ✅ (Trilinos) | ❌ | ❌ |
| **Array/Matrix lib** | ❌ | ❌ | ❌ | ✅ (built-in) | ✅ (Eigen) | N/A | ✅ (Trilinos) | ✅ (Eigen) | ✅ (Eigen) |
| **Optimization algos** | ❌ | ❌ | ❌ | ✅ (LM, L-BFGS, CG) | ❌ | ❌ | ❌ (via Trilinos) | ❌ (via Stan) | ❌ |
| **Header-only** | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ (plugin) | ❌ | ❌ | ❌ (CMake lib) |
| **Prob. distributions** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (100+) | ❌ |
| **ODE solver AD** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| **Complex numbers** | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Vector mode** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (FReal/AReal<T,N>) |
| **JIT replay** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (v2.0) |
| **Checkpointing** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Python bindings** | ❌ | ✅ (SWIG) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ (via Stan) | ✅ (XAD-Py) |

### 3.2 Implementation & Usability Comparison

| Aspect | CppAD | ADOL-C | CoDiPack | Adept 2 | autodiff | Enzyme | Sacado | Stan Math | XAD |
|---|---|---|---|---|---|---|---|---|---|
| **Approach** | OO (tape) | OO (tape) | OO (tape+ET) | OO (stack+ET) | OO (ET) | Source xform | OO (ET) | OO (tape) | OO (tape+ET) |
| **C++ standard** | C++11 | C++11 | C++17 | C++11 | C++17 | Any (LLVM) | C++14 | C++14 | C++11 |
| **Code changes needed** | Type change | Type change + trace_on/off | Type change | Type change | Type change | **None** | Type change | Type change | Type change |
| **API complexity** | Medium | High | Medium | Low | **Very low** | **Very low** | Medium | Medium-High | Low-Medium |
| **Learning curve** | Medium | Steep | Medium | Gentle | **Easiest** | Low (setup hard) | Medium | Steep | Gentle |
| **Documentation** | Excellent | Good | Good | Good (User Guide) | Good (website) | Good (website) | Fair (Trilinos) | Good | Good (website + tutorials) |
| **Build system** | CMake | CMake/Autotools | Header (CMake) | Autotools | Header (CMake) | CMake+LLVM | CMake (Trilinos) | Make/CMake | CMake |
| **Install complexity** | Easy | Medium | **Trivial** | Medium | **Trivial** | Hard (LLVM dep) | Hard (Trilinos) | Medium | Easy |
| **Dependencies** | None | ColPack (optional) | None | BLAS (optional) | Eigen (optional) | LLVM | Trilinos/Kokkos | Eigen, TBB, Sundials, Boost | None (Eigen optional) |
| **Maturity** | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★★ | ★★★★ | ★★★★★ | ★★★★★ | ★★★★ |
| **Active development** | ✅ | ✅ | ✅ | Moderate | Moderate | ✅ Very active | ✅ | ✅ | ✅ Very active |

### 3.3 Performance Comparison

Performance depends heavily on the problem (size, sparsity, forward vs. reverse, etc.). The following is a qualitative summary based on published benchmarks and community experience:

| Aspect | CppAD | ADOL-C | CoDiPack | Adept 2 | autodiff | Enzyme | Sacado | Stan Math | XAD |
|---|---|---|---|---|---|---|---|---|---|
| **Forward-mode overhead** | 3–8× | 5–15× | 2–5× | 3–6× | 2–4× | **1.2–3×** | **1.5–4×** | 3–8× | **2–5×** |
| **Reverse-mode overhead** | 3–10× | 5–20× | **2–6×** | **2–5×** | 3–8× | **1.5–4×** | 3–8× | 3–10× | **2–6×** |
| **Tape memory** | Medium | High | **Low** (multiple options) | **Low** (stack) | Low (tree) | **None** | Low (ET) | Medium | **Low** (optimized) |
| **Tape recording** | Medium | Slow | Fast | Fast | Fast | **None** | Fast | Medium | Fast |
| **Large-scale suitability** | Good | Good | **Excellent** | Good | Fair | **Excellent** | **Excellent** | Good | **Excellent** |
| **Small function perf.** | Fair | Poor (overhead) | Good | Good | **Excellent** | **Excellent** | **Excellent** | Fair | Good |
| **Sparse derivative perf.** | **Excellent** | **Excellent** | Manual | Manual | N/A | Manual | Manual | N/A | Manual |
| **JIT replay perf.** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **Excellent** |

> **Legend for overhead:** The multiple $x \times$ indicates the ratio of time for `(function + derivative)` compared to `function only`. Lower is better. An overhead of $2\times$ means computing the gradient takes about the same time as computing the function value alone.

**Key performance insights:**

1. **Enzyme** consistently achieves the lowest overhead because it differentiates optimized LLVM IR — the compiler can optimize across the derivative code, eliminate dead stores, vectorize, etc. Published benchmarks show it meeting or exceeding hand-written derivatives.

2. **CoDiPack** and **Adept** are the fastest among operator-overloading libraries for reverse mode, thanks to expression templates and efficient tape/stack designs.

3. **Sacado's `SFad`** (static forward) is extremely fast for forward mode when the number of derivatives is known at compile time, because derivative arrays live on the stack and enable SIMD vectorization.

4. **CppAD** and **ADOL-C** have higher tape overhead but compensate with built-in sparse derivative support via graph coloring, making them superior for large sparse Jacobians/Hessians.

5. **autodiff** has minimal overhead for small functions and simple use cases but may not scale as well for very large tapes due to the expression tree approach.

6. **Stan Math** is not the fastest general-purpose AD but is highly optimized for its target domain (statistical distributions, matrix operations) with arena-based allocation.

7. **XAD** achieves competitive reverse-mode overhead through expression templates and a memory-efficient tape. Its unique JIT backend (v2.0) shines for Monte Carlo workloads where the same tape is replayed millions of times — the JIT compilation amortizes setup cost across replays. Vector mode (`FReal<T,N>`, `AReal<T,N>`) allows computing multiple derivatives in a single pass, further reducing overhead.

---

## 4. Decision Guide: Choosing a Library

### Choose **autodiff** if:
- You want the easiest API and fastest setup
- Your project is small to medium scale
- You want both forward and reverse mode with minimal code changes
- You value modern C++17 design
- You want to teach AD or prototype quickly

### Choose **Enzyme** if:
- You need maximum performance (close to hand-written derivatives)
- You don't want to change your source code types
- You need GPU differentiation (CUDA/ROCm)
- You can handle the LLVM build dependency
- You want to differentiate existing legacy code
- You need parallel AD (MPI, OpenMP)

### Choose **CoDiPack** if:
- You need high-performance reverse mode for large-scale CFD/simulation
- You want multiple tape type options to trade off memory vs. speed
- You need MPI-parallel AD (AdjointMPI)
- You are in the computational fluid dynamics (SU2) community

### Choose **CppAD** if:
- You need sparse Jacobians and Hessians with automatic graph coloring
- You need tape optimization (constant folding, dead-code elimination)
- You interface with IPOPT or other COIN-OR optimization solvers
- You need conditional expressions without re-taping
- You value long-term stability and excellent documentation

### Choose **ADOL-C** if:
- You need sparse higher-order derivative tensors
- You want the most theoretically complete AD tool
- You're in an academic environment familiar with Griewank/Walther methods
- You need multiple simultaneous tapes

### Choose **Sacado** if:
- You're already using the Trilinos ecosystem
- You need Kokkos-based GPU/many-core performance
- You need compile-time-sized forward-mode derivatives (SFad) for innermost loops
- You're at a DOE lab or working on large-scale multiphysics simulations

### Choose **Adept 2** if:
- You need combined AD + array library + optimization
- You work in atmospheric/meteorological science
- You want a self-contained package with built-in linear algebra
- You want fast reverse mode with a gentle learning curve

### Choose **Stan Math** if:
- You're doing Bayesian inference or statistical modeling
- You need differentiable probability distributions
- You need AD through ODE/algebraic solvers
- You're building on the Stan ecosystem

### Choose **XAD** if:
- You work in **quantitative finance** (Greeks, risk sensitivities, XVA)
- You need a **JIT replay** workflow for Monte Carlo simulations (record once, replay millions of times)
- You want **vector mode** to compute multiple derivatives in a single pass
- You need checkpointing for memory-constrained large-scale computations
- You want QuantLib integration out of the box (QuantLibAAD)
- You need higher-order derivatives with flexible mode nesting (fwd/adj in any combination)
- You want both C++ and Python bindings from the same library
- You value production-grade quality with extensive CI (GCC, Clang, MSVC, macOS)

---

## 5. Quick Reference: Overhead Factors from Literature

The following are approximate derivative-to-function cost ratios from published benchmarks. The "cheap gradient principle" states that reverse-mode gradient cost should be ≤ 5× the function cost.

| Benchmark problem | CppAD | ADOL-C | CoDiPack | Adept | Enzyme | Sacado |
|---|---|---|---|---|---|---|
| Polynomial (n=100) | 4.2× | 8.1× | 2.8× | 2.5× | 1.8× | 2.1× |
| Ackley function (n=100) | 5.1× | 10.3× | 3.2× | 3.0× | 2.0× | 2.5× |
| Neural network (small) | 4.5× | 12.0× | 3.5× | 3.1× | 1.5× | 3.0× |
| ODE (Lorenz, n=3) | 3.8× | 7.5× | 2.5× | 2.3× | 1.7× | 2.2× |

> **Note:** These numbers are approximate and vary significantly with compiler, optimization flags, problem structure, and library version. Always benchmark on your specific problem. Numbers sourced from Sagebaum et al. (2019), Hogan (2014), Moses et al. (2020, 2021), and community benchmarks.

---

## 6. Ecosystem and Community

| Library | GitHub Stars | Contributors | Last Release | Primary Domain |
|---|---|---|---|---|
| **autodiff** | ~1,900 | 28 | v1.1.2 (2024) | General purpose |
| **Enzyme** | ~1,500 | 71 | Nightly builds | HPC, ML, general |
| **Stan Math** | ~807 | 110 | v5.2.0 (Jan 2026) | Statistics, MCMC |
| **CppAD** | ~560 | 18 | 20260000.0 (Jan 2026) | Optimization |
| **Adept 2** | ~185 | 5 | v2.1.3 (May 2025) | Meteorology, arrays |
| **ADOL-C** | ~175 | 17 | 34 tags | Academic, optimization |
| **CoDiPack** | ~109 | 9 | v3.1.0 (Feb 2026) | CFD, simulation |
| **XAD** | ~411 | 34 | v2.0.0 (Feb 2026) | Quant. finance, general |
| **Sacado** | (Trilinos ~1,400) | 302 (Trilinos) | Trilinos 17.0 (Feb 2026) | DOE/HPC |

---

## 7. Summary

There is no single "best" C++ AD library — the right choice depends on your domain, scale, and performance requirements:

- For **ease of use**: `autodiff` (best API) or `Enzyme` (no code changes)
- For **raw performance**: `Enzyme` (compiler-level) or `Sacado SFad` (forward) / `CoDiPack` (reverse)
- For **sparse large-scale derivatives**: `CppAD` or `ADOL-C` (built-in coloring)
- For **HPC/parallel**: `Enzyme` (MPI/OpenMP/GPU) or `CoDiPack` (AdjointMPI) or `Sacado` (Kokkos)
- For **quantitative finance / Monte Carlo**: `XAD` (JIT replay, QuantLib integration, vector mode)
- For **statistical modeling**: `Stan Math` (unmatched probability function library)
- For **all-in-one (AD + arrays + optimization)**: `Adept 2`
- For **maximum maturity and documentation**: `CppAD` (20+ years)
