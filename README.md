# Automatic Differentiation in C++

A from-scratch C++17 implementation of **forward-mode** and **reverse-mode** automatic differentiation (AD), with full Jacobian computation. Companion code for [ad.md](ad.md).

---

## Project Structure

```
ad/
├── Makefile                     # Build system
├── ad.md                        # Theory & derivations
├── cpp_ad_libraries.md          # Open-source C++ AD library comparison
├── README.md                    # ← you are here
├── build/                       # Compiled binaries (generated)
└── src/
    ├── dual.hpp                 # Header-only forward-mode AD (dual numbers)
    ├── var.hpp                  # Header-only reverse-mode AD (Wengert tape)
    ├── forward_mode.cpp         # Forward-mode examples
    ├── reverse_mode.cpp         # Reverse-mode examples
    ├── jacobian_forward.cpp     # Jacobian via forward-mode (column by column)
    └── jacobian_reverse.cpp     # Jacobian via reverse-mode (row by row)
```

---

## Requirements

- **C++17** compiler (GCC ≥ 7, Clang ≥ 5, or MSVC ≥ 19.14)
- **GNU Make**

No external libraries are needed — everything is self-contained.

---

## Build & Run

```bash
make            # Build all four example programs into build/
make run        # Build and run all examples
make clean      # Remove build/ directory
make help       # List all available targets
```

Build individual targets:

```bash
make forward_mode       # Only the forward-mode example
make reverse_mode       # Only the reverse-mode example
make jacobian_forward   # Only the Jacobian (forward) example
make jacobian_reverse   # Only the Jacobian (reverse) example
```

Compiler flags used: `-std=c++17 -O2 -Wall -Wextra -Wpedantic`.

---

## File Overview

### Headers

#### `src/dual.hpp` — Forward-Mode AD via Dual Numbers

A header-only library that implements the **dual number** type for forward-mode AD. A dual number is a pair $(v, \dot{v})$ where $v$ is the primal value and $\dot{v}$ is the tangent (derivative). The key identity is $\varepsilon^2 = 0$, so $f(a + b\varepsilon) = f(a) + b \cdot f'(a) \cdot \varepsilon$.

**Provides:**

| Component | Description |
|---|---|
| `struct Dual` | Holds `real` (value) and `dual` (derivative) as `double` |
| Arithmetic operators | `+`, `-`, `*`, `/`, unary `-` (all follow dual-number algebra) |
| Scalar-Dual mixed ops | `double ⊕ Dual` and `Dual ⊕ double` for `+`, `-`, `*`, `/` |
| Math functions | `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `pow`, `abs` |
| `operator<<` | Pretty-prints as `3.000000 + 1.000000ε` |

**Usage pattern — compute f(x) and f'(x) in one pass:**

```cpp
#include "dual.hpp"

Dual x(3.0, 1.0);        // x = 3, seed dx/dx = 1
Dual y = sin(x * x);     // f(x) = sin(x²)
// y.real = sin(9)        — the value
// y.dual = 6·cos(9)      — the derivative (chain rule applied automatically)
```

---

#### `src/var.hpp` — Reverse-Mode AD via Wengert Tape

A header-only library that implements **reverse-mode AD** using a global tape of operations. Each `Var` records its value and a closure that propagates gradients backward. This is the same mechanism behind backpropagation in neural networks.

**Provides:**

| Component | Description |
|---|---|
| `struct TapeEntry` | Holds `value`, `adjoint`, and a `std::function<void(double)>` backward closure |
| `get_tape()` | Returns a reference to the global `std::vector<TapeEntry>` (static local) |
| `struct Var` | A handle (index into the tape); leaf constructor and computed-node constructor |
| Arithmetic operators | `+`, `-`, `*`, `/` — each records its local backward rule |
| Math functions | `sin`, `cos`, `exp`, `log`, `sqrt` |
| `backward(Var)` | Seeds the output adjoint to 1.0 and sweeps the tape in reverse |
| `backward_from(int)` | Backward sweep from a specific tape index (for Jacobian rows) |
| `reset_adjoints()` | Zeros all adjoints (for multiple backward passes on the same tape) |
| `clear_tape()` | Clears the tape entirely (call before a new computation) |

**Usage pattern — compute all gradients in one backward pass:**

```cpp
#include "var.hpp"

clear_tape();
Var x1(2.0), x2(3.0);
Var y = x1 * x2 + sin(x1);    // forward pass records the tape

backward(y);                    // backward sweep

double df_dx1 = x1.adjoint();  // ∂f/∂x₁ = x₂ + cos(x₁) = 3.583…
double df_dx2 = x2.adjoint();  // ∂f/∂x₂ = x₁ = 2.0
```

---

### Example Programs

#### `src/forward_mode.cpp` — Forward-Mode Examples

Demonstrates forward-mode AD using `dual.hpp` across 10 examples from [ad.md](ad.md):

| # | Example | ad.md Section | What it tests |
|---|---|---|---|
| 1 | $f(x) = x^2$ at $x = 3$ | §5 | Basic squaring |
| 2 | $f(x) = x^3 + 2x$ at $x = 2$ | §5 | Polynomial |
| 3 | $f(x) = \sin(x)$ at $x = 0$ | §5 | Trig function |
| 4 | $f(x) = \sin(x^2)$ at $x = 3$ | §5 | Chain rule |
| 5 | $f(x) = \sin(x^2)$ at $x = 3$ | §8 | C++ sketch from the text |
| 6 | $f(x) = e^{\sin(x^2)}$ at $x = 1$ | §15 | Multi-layer composition |
| 7 | All math functions at $x = 2$ | §6 | `exp`, `log`, `sqrt`, `tan`, `cos`, `pow` |
| 8 | $f(x) = \sin(x)/x$ at $x = \pi$ | §3 | Quotient rule |
| 9 | $f(x_1,x_2) = x_1 x_2 + \sin(x_1)$ | §7 | Partial derivatives (two passes) |
| 10 | $f(x) = x \cdot x$ at $x = 5$ | §18 | Fan-out |

Each example verifies the computed derivative against the known analytical result (tolerance: $10^{-10}$).

---

#### `src/reverse_mode.cpp` — Reverse-Mode Examples

Demonstrates reverse-mode AD using `var.hpp` across 8 examples:

| # | Example | ad.md Section | What it tests |
|---|---|---|---|
| 1 | $f(x_1,x_2) = x_1 x_2 + \sin(x_1)$ | §13, §22 | Basic gradient (both partials in one pass) |
| 2 | $f(x) = e^{\sin(x^2)}$ | §15 | Multi-layer chain |
| 3 | $f(x) = x \cdot x$ | §18 | Fan-out (adjoint accumulation) |
| 4 | $f(x) = (x^2-1)/(x+1)$ | — | Division & subtraction |
| 5 | $f(x) = \log(1 + e^x)$ | — | Softplus (derivative = sigmoid) |
| 6 | $f(x,y) = x\sin(y) + y\cos(x)$ | — | Complex mixed expression |
| 7 | $f(x) = \sqrt{x^2+1}$ | — | `sqrt` with chain rule |
| 8 | $f(a,b,c) = a \cdot b \cdot c$ | — | Three variables, all gradients in one pass |

---

#### `src/jacobian_forward.cpp` — Jacobian via Forward-Mode

Computes the full Jacobian of a vector-valued function **column by column** using forward-mode AD ($n$ passes for $n$ inputs).

**Test function:**

$$
f(x_1, x_2) = \begin{pmatrix} x_1 \cdot x_2 \\ \sin(x_1) + x_2^2 \end{pmatrix}
$$

At $(x_1, x_2) = (2, 3)$, the Jacobian is:

$$
J = \begin{pmatrix} 3 & 2 \\ \cos(2) & 6 \end{pmatrix}
$$

**Also demonstrates JVP (Jacobian-Vector Product):**

- $J \cdot e_1 = $ column 1
- $J \cdot e_2 = $ column 2
- $J \cdot (1,1)^T = $ sum of columns

---

#### `src/jacobian_reverse.cpp` — Jacobian via Reverse-Mode

Computes the full Jacobian **row by row** using reverse-mode AD ($m$ passes for $m$ outputs). Same test function as above.

**Also demonstrates:**

- **VJP (Vector-Jacobian Product):** $u^T J$ for various covectors $u$
- **Scalar gradient (ML use case):** $L(w_1,w_2,w_3) = (w_1 w_2 + w_3)^2$ — all 3 gradients from a single backward pass

---

## Key Concepts

### Forward Mode vs Reverse Mode

| | Forward Mode | Reverse Mode |
|---|---|---|
| **Implementation** | Dual numbers (operator overloading) | Wengert tape + backward sweep |
| **One pass gives** | One column of $J$ (or one JVP) | One row of $J$ (or one VJP) |
| **Cost for full $J$** | $n$ passes ($n$ = number of inputs) | $m$ passes ($m$ = number of outputs) |
| **Best when** | $m \gg n$ (few inputs, many outputs) | $n \gg m$ (many inputs, few outputs) |
| **ML / backprop** | Rarely used alone | This **is** backpropagation |

### When to Use Which

- **Gradient of a scalar loss** (the typical ML case): reverse mode — one pass gives all $n$ partial derivatives.
- **Jacobian of $f : \mathbb{R}^n \to \mathbb{R}^m$:**
  - $n \leq m$: forward mode ($n$ passes)
  - $m < n$: reverse mode ($m$ passes)
- **Directional derivative** $J \cdot v$: a single forward-mode pass.
- **Gradient-transpose product** $u^T J$: a single reverse-mode pass.

---

## Theory Reference

See [ad.md](ad.md) for detailed derivations covering:

- **§1–§9:** Dual numbers, arithmetic rules, Taylor expansion proof, worked examples, function derivative table, multi-variable partials, C++ implementation sketch, geometric interpretation
- **§10–§22:** Reverse-mode motivation, computation graphs, forward/backward passes, local derivative rules, Wengert tape, the backward algorithm, fan-out handling, forward vs reverse comparison, neural network backpropagation, checkpointing, C++ implementation sketch
- **§23–§33:** Jacobian matrix definition, JVP/VJP, building the full Jacobian (forward vs reverse strategy), worked example, duality between JVP and VJP, sparse Jacobians with graph coloring, higher-order derivatives (Hessian-vector products), C++ implementations, decision flowchart

See [cpp_ad_libraries.md](cpp_ad_libraries.md) for a comparison of 8 open-source C++ AD libraries (CppAD, ADOL-C, CoDiPack, Adept 2, autodiff, Enzyme, Sacado, Stan Math).
