# XAD Technical Reference: Active Types, Modes, and the Tape

> A deep-dive into `AReal`, `FReal`, dual numbers, tape-based adjoint differentiation,
> and why QuantLibAAD exclusively uses `AReal<double>`.

---

## Table of Contents

1. [Automatic Differentiation — Two Perspectives](#1-automatic-differentiation--two-perspectives)
2. [The Dual Number and Forward Mode — `FReal<T>`](#2-the-dual-number-and-forward-mode--frealt)
   - 2.1 [What Is a Dual Number?](#21-what-is-a-dual-number)
   - 2.2 [XAD's `FReal<T>` Implementation](#22-xads-frealt-implementation)
   - 2.3 [Forward Mode in Practice](#23-forward-mode-in-practice)
   - 2.4 [Vector Forward Mode `FReal<T, N>`](#24-vector-forward-mode-frealt-n)
   - 2.5 [Cost Model of Forward Mode](#25-cost-model-of-forward-mode)
3. [The Tape and Adjoint Mode — `AReal<T>`](#3-the-tape-and-adjoint-mode--arealt)
   - 3.1 [Adjoint / Reverse Mode Concept](#31-adjoint--reverse-mode-concept)
   - 3.2 [What the Tape Stores](#32-what-the-tape-stores)
   - 3.3 [`AReal<T>` Internal Layout](#33-arealt-internal-layout)
   - 3.4 [Adjoint Mode in Practice](#34-adjoint-mode-in-practice)
   - 3.5 [Vector Adjoint Mode `AReal<T, N>`](#35-vector-adjoint-mode-arealt-n)
   - 3.6 [Cost Model of Adjoint Mode](#36-cost-model-of-adjoint-mode)
4. [Expression Templates: the Performance Bridge](#4-expression-templates-the-performance-bridge)
5. [Side-by-Side Comparison: `FReal` vs `AReal`](#5-side-by-side-comparison-freal-vs-areal)
6. [Higher-Order Derivatives by Nesting](#6-higher-order-derivatives-by-nesting)
7. [Why QuantLibAAD Uses Only `AReal<double>`](#7-why-quantlibaad-uses-only-arealdouble)
   - 7.1 [The Financial Sensitivity Problem](#71-the-financial-sensitivity-problem)
   - 7.2 [Cost Comparison in Quant Finance](#72-cost-comparison-in-quant-finance)
   - 7.3 [How the Substitution Works](#73-how-the-substitution-works)
   - 7.4 [The `qlaad.hpp` Glue Header](#74-the-qlaadhpp-glue-header)
8. [Full AAD Workflow in QuantLibAAD](#8-full-aad-workflow-in-quantlibaad)
9. [Tape Lifecycle and Memory](#9-tape-lifecycle-and-memory)
10. [The AD Mode Interface — A Unified Vocabulary](#10-the-ad-mode-interface--a-unified-vocabulary)

---

## 1. Automatic Differentiation — Two Perspectives

Automatic differentiation (AD) is the mechanical application of the chain rule to a
computer program, producing exact derivatives without symbolic manipulation and without
finite-difference approximation errors.

Every numeric computation, no matter how complex, ultimately decomposes into a composition
of elementary operations whose derivatives are known analytically:

```
y = f(g(h(x)))
dy/dx = f'(g(h(x))) · g'(h(x)) · h'(x)
```

The chain rule can be applied in **two directions**:

| Direction | Name | Propagates | Cost per input | Cost per output |
|---|---|---|---|---|
| Left-to-right (input → output) | **Forward / Tangent mode** | Tangents $\dot{x}$ | O(1) per derivative | O(M) runs for M outputs |
| Right-to-left (output → input) | **Adjoint / Reverse mode** | Adjoints $\bar{x}$ | O(N) derivatives in 1 sweep | O(1) per derivative |

For a function $f: \mathbb{R}^N \rightarrow \mathbb{R}^M$:
- Forward mode computes one **column** of the Jacobian per run (one input direction at a time).
- Adjoint mode computes one **row** of the Jacobian per run (one output direction at a time).

XAD provides a concrete type for each mode:
- **`FReal<T>`** — forward/tangent mode (dual number)
- **`AReal<T>`** — adjoint/reverse mode (tape-based)

---

## 2. The Dual Number and Forward Mode — `FReal<T>`

### 2.1 What Is a Dual Number?

A **dual number** extends the real line with a nilpotent perturbation element $\epsilon$
where $\epsilon^2 = 0$:

$$
x + \dot{x}\epsilon \qquad x, \dot{x} \in \mathbb{R}
$$

Arithmetic on dual numbers automatically transports the derivative alongside the value.
For any smooth function $f$:

$$
f(x + \dot{x}\epsilon) = f(x) + f'(x)\dot{x}\,\epsilon
$$

So after evaluating $f$ on a dual number, the $\epsilon$-component of the result is
exactly $f'(x)\,\dot{x}$ — no approximation, no symbolic manipulation.

Example chain for $y = x^2$, seed $\dot{x} = 1$:

```
input : (x, ε·1)     => value=x,   tangent=1
y = x*x : (x², ε·2x) => value=x²,  tangent=2x = dy/dx  ✓
```

### 2.2 XAD's `FReal<T>` Implementation

```cpp
// Declaration (from XAD source, simplified)
template <typename T, std::size_t N = 1>
class FReal : public Expression<T, FReal<T, N>>
{
    T       value_;       // primal value
    T       derivative_;  // tangent (ε-component)   [scalar when N=1]
                          // Vec<T,N> when N>1
};
```

Key properties:
- **Self-contained** — no tape, no global state. Each `FReal` object carries its own tangent.
- **Operator overloading** propagates the chain rule inline through every arithmetic expression.
- **No destructor work** — variables can be created and destroyed freely.

### 2.3 Forward Mode in Practice

```cpp
#include <XAD/XAD.hpp>

// Scalar forward mode
using AD = xad::FReal<double>;

AD x(1.3, 1.0);   // value=1.3, seed dx/dx = 1.0 (tangent)

AD y = sin(x) + x * x;

double val  = xad::value(y);      // primal:    sin(1.3) + 1.3² ≈ 2.654
double dydx = xad::derivative(y); // tangent: cos(1.3)*1 + 2*1.3 ≈ 3.264
```

To compute the derivative with respect to a different input, you re-run the computation
with a different tangent seed (independent variable), or use the vector mode.

### 2.4 Vector Forward Mode `FReal<T, N>`

When $N > 1$, `FReal<T, N>` stores $N$ tangent components simultaneously, computing
$N$ derivatives in a single forward pass:

```cpp
using AD = xad::FReal<double, 2>;  // compute dy/dx0 and dy/dx1 together

AD x0(1.3, {1.0, 0.0});   // tangent for x0: dx0/dx0=1, dx0/dx1=0
AD x1(5.2, {0.0, 1.0});   // tangent for x1: dx1/dx0=0, dx1/dx1=1

AD y = x0 * x1 + sin(x0);

// derivative()[0] = dy/dx0,  derivative()[1] = dy/dx1
auto d0 = xad::derivative(y)[0]; // dy/dx0 = x1 + cos(x0)
auto d1 = xad::derivative(y)[1]; // dy/dx1 = x0
```

### 2.5 Cost Model of Forward Mode

For a function $f: \mathbb{R}^N \rightarrow \mathbb{R}^M$:

- **Per-run cost**: ~2-3× the cost of the primal evaluation (scalar mode).
- **For all $N$ input derivatives**: $N$ separate passes, or a single pass with `FReal<T, N>`.
- **Memory**: $O(1)$ extra per variable (one extra `double` per variable in scalar mode).
- **Thread safe**: no shared global state.

Forward mode is most efficient when **$N$ is small** (few inputs, potentially many outputs).

---

## 3. The Tape and Adjoint Mode — `AReal<T>`

### 3.1 Adjoint / Reverse Mode Concept

Adjoint (reverse) mode computes all input derivatives in a **single backward sweep**.
It requires two phases:

1. **Forward (recording) pass**: Execute the program normally while recording every
   elementary operation and the values required for their local Jacobians onto a data
   structure called the **tape** (also called Wengert list or computational graph).

2. **Backward (adjoint) pass**: Traverse the tape in *reverse order*, propagating
   adjoint values $\bar{v}_i = \partial y / \partial v_i$ backwards through the chain rule.
   For each intermediate variable $v_i = g(v_j, v_k)$:

$$
\bar{v}_j \mathrel{+}= \bar{v}_i \cdot \frac{\partial g}{\partial v_j}, \qquad
\bar{v}_k \mathrel{+}= \bar{v}_i \cdot \frac{\partial g}{\partial v_k}
$$

The result is $\nabla_x y$ — the gradient w.r.t. **all inputs** — obtained in a single
forward+backward pass, regardless of how many inputs there are.

### 3.2 What the Tape Stores

`xad::Tape<double>` is a dynamic list of operation records. Each record contains:

| Field | Purpose |
|---|---|
| Operation type | Which elementary op was performed (mul, add, sin, …) |
| Local partial derivatives | Pre-computed $\partial g / \partial v_j$ for the inputs |
| Slot indices | Where to find (and accumulate into) the adjoint of each operand |

The tape also maintains an **adjoint array**: a dense `double[]` indexed by slot numbers,
one entry per registered variable. This is where backward-sweep accumulations happen.

### 3.3 `AReal<T>` Internal Layout

```cpp
// Declaration (from XAD source, simplified)
template <typename T, std::size_t N = 1>
class AReal : public ADTypeBase<T, AReal<T, N>>
{
    T         value_;   // primal value (same role as FReal::value_)
    slot_type slot_;    // index into the active Tape's adjoint array
                        // INVALID_SLOT if not registered with any tape
};
```

**Critical insight**: `AReal` does **not** store the derivative inside itself.
The derivative lives on the `Tape` at position `slot_`. Calling `xad::derivative(x)`
simply reads `tape.adjoints[x.slot_]`. This indirection is what allows a single backward
sweep to update all variables' adjoints simultaneously.

Lifecycle of a slot:
1. `tape.registerInput(x)` — allocates a slot, sets `x.slot_`, initialises adjoint to 0.
2. Arithmetic operations on `x` — new slots are allocated for intermediates; operation
   records are pushed onto the tape.
3. `tape.registerOutput(y)` + `derivative(y) = 1.0` — seeds the output adjoint.
4. `tape.computeAdjoints()` — walks the tape in reverse, accumulating adjoints.
5. `derivative(x)` — reads the final accumulated adjoint $\partial y / \partial x$.

### 3.4 Adjoint Mode in Practice

```cpp
#include <XAD/XAD.hpp>

xad::Tape<double> tape;

xad::AReal<double> x0 = 1.3;
xad::AReal<double> x1 = 5.2;

// Step 1: register independent variables
tape.registerInput(x0);
tape.registerInput(x1);

// Step 2: start recording
tape.newRecording();

// Step 3: forward pass — the computation graph is recorded onto tape
xad::AReal<double> y = x0 * x1 + sin(x0);

// Step 4: register output and seed adjoint dy/dy = 1
tape.registerOutput(y);
xad::derivative(y) = 1.0;

// Step 5: backward sweep — computes all adjoints in one pass
tape.computeAdjoints();

// Step 6: read results
double dy_dx0 = xad::derivative(x0); // x1 + cos(x0) = 5.2 + cos(1.3)
double dy_dx1 = xad::derivative(x1); // x0 = 1.3
double val    = xad::value(y);
```

For a function with $N$ inputs, **steps 1-6 produce all $N$ derivatives** in roughly
the same time as 3-5 forward evaluations — completely independent of $N$.

### 3.5 Vector Adjoint Mode `AReal<T, N>`

When `N > 1`, the adjoint array stores `Vec<T, N>` adjoints per slot, allowing $N$
output seeds to be propagated simultaneously in a single backward pass.
This is useful when you have multiple outputs and want all their gradients at once:

```cpp
using AD = xad::AReal<double, 2>;
xad::Tape<double, 2> tape;

AD x = 1.3;
tape.registerInput(x);
tape.newRecording();

AD y0 = sin(x);
AD y1 = x * x;
tape.registerOutput(y0);
tape.registerOutput(y1);

// seed both outputs simultaneously
xad::derivative(y0) = {1.0, 0.0};  // dy0/d(·)
xad::derivative(y1) = {0.0, 1.0};  // dy1/d(·)

tape.computeAdjoints();

auto dx = xad::derivative(x);  // dx[0] = dy0/dx = cos(1.3),  dx[1] = dy1/dx = 2*1.3
```

### 3.6 Cost Model of Adjoint Mode

For a function $f: \mathbb{R}^N \rightarrow \mathbb{R}^M$:

- **Per-run cost**: ~3-5× the cost of the primal evaluation (forward + backward pass).
- **For all $N$ input derivatives**: only **1 run**. Cost is constant w.r.t. $N$.
- **Memory**: the tape grows proportionally to the number of recorded operations — i.e.,
  the computational path length (not the number of inputs).
- **For $M > 1$ outputs**: adjoint mode still provides the whole gradient in $M$ backward
  sweeps (one per output); or use `AReal<T, M>` to do all $M$ sweeps simultaneously.

Adjoint mode is most efficient when **$N \gg M$** — many inputs, few outputs.

---

## 4. Expression Templates: the Performance Bridge

Naive operator-overloading implementations of AD suffer from creating temporary objects
for every sub-expression. For example, `a + b * sin(c)` would allocate and register
several intermediate `AReal` objects onto the tape, even though only the final result
is needed as a named variable.

XAD avoids this via **expression templates**: arithmetic on `AReal` objects returns
lazy expression nodes instead of fully evaluated `AReal` values:

```cpp
// Types involved in:  y = x0 * x1 + sin(x0)
//
// x0 * x1           => BinaryExpr<double, MulOp,  ADVar<double>, ADVar<double>>
// sin(x0)            => UnaryExpr<double,  SinOp,  ADVar<double>>
// x0*x1 + sin(x0)    => BinaryExpr<double, AddOp,  BinaryExpr<...>, UnaryExpr<...>>
//
// Only when assigned to AReal y is the whole tree evaluated and pushed to tape.
xad::AReal<double> y = x0 * x1 + sin(x0);
```

The entire sub-expression tree is walked once during the assignment, and a **single
tape entry** is pushed that captures all partial derivatives of the combined expression.
This eliminates spurious intermediate tape entries and reduces memory and backward-pass
work significantly.

Because `BinaryExpr` and `UnaryExpr` types are not `AReal`, the Boost and QuantLib
function overloads in [ql/qlaad.hpp](ql/qlaad.hpp) must explicitly handle both `AReal`
**and** the expression template types — which is the primary reason for the hundreds of
template specialisations in that header.

---

## 5. Side-by-Side Comparison: `FReal` vs `AReal`

| Property | `FReal<T>` (Forward) | `AReal<T>` (Adjoint) |
|---|---|---|
| **AD mode** | Tangent / Forward | Adjoint / Reverse |
| **Mathematical object** | Dual number $(x, \dot x)$ | Primal + tape slot $(x, \text{slot})$ |
| **Derivative stored** | Inline, inside the object | On the `Tape` at `slot_` |
| **Tape required** | No | Yes — `Tape<T>` singleton per thread |
| **Derivative direction** | Input → Output (column of Jacobian) | Output → Input (row of Jacobian) |
| **Derivatives per run** | 1 (or $N$ with vector mode) | All $N$ inputs (or $M$ outputs in vector mode) |
| **Memory per variable** | $O(1)$ — one extra `T` | $O(1)$ per variable + tape growth $O(\text{ops})$ |
| **Thread safety** | Fully thread-safe (no shared state) | Thread-local tape (safe with care) |
| **Best for** | Few inputs, many outputs; higher-order building block | Many inputs, few outputs (typical finance Greeks) |
| **Nesting for 2nd order** | `FReal<FReal<T>>` or `FReal<AReal<T>>` | `AReal<AReal<T>>` or `AReal<FReal<T>>` |

---

## 6. Higher-Order Derivatives by Nesting

XAD supports arbitrary-order derivatives by **nesting** the active types.
Both `FReal` and `AReal` can take another active type as their scalar template parameter.

Common patterns from the XAD source (`Interface.hpp`):

| Mode alias | Active type | Computes |
|---|---|---|
| `adj` | `AReal<double>` | 1st-order adjoints |
| `fwd` | `FReal<double>` | 1st-order tangents |
| `fwd_adj` | `AReal<FReal<double>>` | Mixed 2nd order: forward outer, adjoint inner |
| `adj_adj` | `AReal<AReal<double>>` | 2nd-order adjoints |
| `adj_fwd` | `FReal<AReal<double>>` | Mixed 2nd order: adjoint outer, forward inner |

Example: computing value, first derivative and second derivative simultaneously:

```cpp
using AD = xad::AReal<xad::FReal<double>>;  // adj_fwd mode
xad::Tape<xad::FReal<double>> tape;

AD x(1.3);
xad::derivative(xad::value(x)) = 1.0;       // seed forward tangent
tape.registerInput(x);
tape.newRecording();

AD y = sin(x);                               // records on outer tape

tape.registerOutput(y);
xad::value(xad::derivative(y)) = 1.0;       // seed adjoint
tape.computeAdjoints();

double val   = xad::value(xad::value(y));            // sin(1.3)
double dy_dx = xad::derivative(xad::value(y));       // cos(1.3)  (forward)
double d2y   = xad::derivative(xad::derivative(x));  // -sin(1.3) (second order)
```

---

## 7. Why QuantLibAAD Uses Only `AReal<double>`

### 7.1 The Financial Sensitivity Problem

A typical interest rate risk run in QuantLib involves:

- **Instrument**: one IRS, bond, swaption, CDS, etc.
- **Market data inputs**: dozens to hundreds of yield curve rates, vol surface points,
  credit spreads — all of which are "independent variables" whose sensitivities (DV01s,
  Vegas, CS01s) are needed.
- **Outputs**: usually just the NPV / fair value — a single scalar output.

This is a textbook $f: \mathbb{R}^N \rightarrow \mathbb{R}$ problem with $N \gg 1$.

For example, the `AdjointSwap` example has roughly:
- $N$ = 10 deposit rates + 5 FRA quotes + several swap rates = ~30 market inputs
- $M$ = 1 output (portfolio NPV)

### 7.2 Cost Comparison in Quant Finance

| Method | Passes needed | Time scaling |
|---|---|---|
| Bump-and-reprice (finite differences) | $N + 1$ pricing calls | $O(N \times T_{\text{price}})$ |
| Forward mode — scalar | $N$ passes | $O(N \times 2T_{\text{price}})$ |
| Forward mode — vector `FReal<T, N>` | 1 pass | $O(N \times T_{\text{price}})$ |
| **Adjoint mode — `AReal<T>`** | **1 forward + 1 backward** | **$O(3$–$5 \times T_{\text{price}})$** independent of $N$ |

For $N = 100$ inputs, adjoint mode is ~20-30× faster than bump-and-reprice and ~60× faster
than scalar forward mode. This advantage only grows with $N$.

### 7.3 How the Substitution Works

QuantLib is parameterised on `Real`, which defaults to `double`. QuantLibAAD redirects
this type definition **at compile time** so the entire QuantLib library is compiled against
`xad::AReal<double>` rather than `double`:

```cpp
// ql/qlaad.hpp  (included BEFORE any QuantLib headers via QL_INCLUDE_FIRST)
#define QL_REAL  xad::AReal<double>
#define QL_AAD   1
#define QL_RISKS 1
```

When QuantLib then defines:
```cpp
// ql/types.hpp (inside QuantLib)
#ifdef QL_REAL
    typedef QL_REAL Real;
#else
    typedef double  Real;
#endif
```

the result is that `QuantLib::Real` becomes `xad::AReal<double>` throughout the entire
library — in all pricing engines, term structures, bootstrappers, etc. — **without
modifying a single QuantLib source file**.

### 7.4 The `qlaad.hpp` Glue Header

Because Boost and QuantLib contain many generic algorithms that use `Real` / `double`
internally, a significant amount of template specialisation is required to make them work
correctly with `AReal<double>` and its expression-template intermediates.

[ql/qlaad.hpp](ql/qlaad.hpp) provides:

| Category | What it does |
|---|---|
| **Boost `promote_args`** | Tells Boost math functions that `AReal` + anything promotes to `AReal` |
| **Boost `policies::evaluation`** | Propagates `AReal` through Boost policy machinery |
| **Boost math overloads** | `erfc`, `log1p`, `tgamma`, `lgamma`, `gamma_p`, `ibeta`, etc. — each overloaded for `UnaryExpr` and `BinaryExpr` to avoid ambiguous conversions |
| **Boost `numeric_cast`** | Handles explicit casts from expression types |
| **Boost accumulator traits** | `result_of_divides`, `result_of_multiplies` for `AReal` |
| **Boost ublas traits** | `promote_traits` for matrix/vector operations with `AReal` |
| **Boost type traits** | `is_floating_point`, `is_arithmetic`, `is_pod`, `is_convertible` |
| **QuantLib overloads** | `squared()` for expression types |

The majority of this code is boilerplate that exists solely because QuantLib and Boost
were written for built-in floating-point types, and expression templates break implicit
conversion chains that those libraries rely on.

---

## 8. Full AAD Workflow in QuantLibAAD

The pattern below is consistent across all test cases and examples in this repository:

```cpp
#include <ql/qldefines.hpp>   // must come first; triggers QL_INCLUDE_FIRST inclusion of qlaad.hpp
// ... other QuantLib headers ...
using namespace QuantLib;

// ── Step 0: Type aliases ─────────────────────────────────────────────────
// Real == xad::AReal<double>  (set by qlaad.hpp)
using tape_type = Real::tape_type;  // xad::Tape<double>

// ── Step 1: Create tape ─────────────────────────────────────────────────
tape_type tape;

// ── Step 2: Copy and register inputs ────────────────────────────────────
std::vector<Real> marketRates = { /* ... */ };
tape.registerInputs(marketRates);   // assigns slots; adjoints initialised to 0

// ── Step 3: Start recording ─────────────────────────────────────────────
tape.newRecording();

// ── Step 4: Forward pass — call unchanged QuantLib pricing code ─────────
Real npv = pricer(marketRates, /* ... */);   // entire QuantLib graph recorded

// ── Step 5: Seed output and roll back ───────────────────────────────────
tape.registerOutput(npv);
derivative(npv) = 1.0;            // dNPV/dNPV = 1 (seed)
tape.computeAdjoints();           // backward sweep

// ── Step 6: Extract sensitivities ───────────────────────────────────────
for (Size i = 0; i < marketRates.size(); ++i) {
    double dv01_i = derivative(marketRates[i]);  // dNPV/d(rate_i)
}
```

For a portfolio with $N = 30$ market inputs, steps 4-6 take roughly the same wall-clock
time as 4-5 single pricings, regardless of $N$.

---

## 9. Tape Lifecycle and Memory

The `Tape<double>` object maintains thread-local global state — it becomes the "active
tape" for any `AReal<double>` created in the same thread after `tape.activate()` is
called (which happens automatically on construction).

Key tape operations:

| Operation | Effect |
|---|---|
| `tape.registerInput(x)` | Allocates slot, tags `x` for differentiation |
| `tape.registerInputs(vec)` | Batch version |
| `tape.newRecording()` | Clears previous recording, starts fresh |
| `tape.registerOutput(y)` | Marks `y` as a dependent variable |
| `derivative(y) = 1.0` | Seeds the output adjoint |
| `tape.computeAdjoints()` | Backward sweep accumulates all input adjoints |
| `tape.clearAll()` | Resets tape and all adjoints (for repeated use) |
| `tape.resetTo(pos)` | Partial reset — reuses tape storage from a checkpoint |
| `tape.getPosition()` | Returns a checkpoint position for `resetTo` / `computeAdjointsTo` |

**Memory considerations**:
- Tape memory grows proportionally to the **number of elementary operations** in the
  forward pass — not the number of inputs/outputs.
- For deep Monte Carlo simulations or very long bootstrapping chains, tape memory can
  become substantial. XAD supports **checkpointing** to bound peak memory:
  register a `CheckpointCallback` that re-computes a segment of the forward pass
  during the backward sweep instead of storing its intermediates.

---

## 10. The AD Mode Interface — A Unified Vocabulary

XAD provides mode-selection structs in `<XAD/Interface.hpp>` as a consistent API
for parameterising code that needs to work with different AD modes:

```cpp
// 1st order, adjoint mode (the QuantLibAAD default)
using mode = xad::adj<double>;
using AD   = mode::active_type;    // xad::AReal<double>
using tape = mode::tape_type;      // xad::Tape<double>

// 1st order, forward mode
using mode = xad::fwd<double>;
using AD   = mode::active_type;    // xad::FReal<double>
// mode::tape_type == void  (no tape)

// 2nd order: adjoint of adjoint
using mode = xad::adj_adj<double>;
using AD   = mode::active_type;    // xad::AReal<xad::AReal<double>>
// two tapes needed: inner (Tape<double>) and outer (Tape<AReal<double>>)

// 2nd order: forward-over-adjoint (Hessian-vector products)
using mode = xad::fwd_adj<double>;
using AD   = mode::active_type;    // xad::AReal<xad::FReal<double>>
```

These mode structs mean you can write a single templated function that works
for any combination of AD order and direction without changing the algorithm logic.

```cpp
template <class Mode>
typename Mode::passive_type computeNPV(/* inputs */) {
    using AD   = typename Mode::active_type;
    using Tape = typename Mode::tape_type;
    // ... unchanged pricing code with AD types ...
}

// Instantiate for adjoint mode
auto npv = computeNPV<xad::adj<double>>(/* ... */);
```

---

## Summary

| Concept | Type | Key Characteristic |
|---|---|---|
| Dual number / Forward mode | `FReal<T, N>` | Stores `(value, tangent)` inline; no tape; thread-safe; $O(N)$ runs for full gradient |
| Tape-based / Adjoint mode | `AReal<T, N>` | Stores `(value, slot)`; adjoints on `Tape`; $O(1)$ runs for full gradient |
| Expression template node | `BinaryExpr`, `UnaryExpr` | Lazy arithmetic; single tape entry for compound expressions |
| QuantLib integration | `#define QL_REAL xad::AReal<double>` | Replaces `double` throughout QuantLib at compile time |
| Why adjoint only in QuantLibAAD | Finance: many inputs ($N$ market quotes), one output (NPV) | Adjoint mode is $O(5\times)$ vs $O(N \times \text{price})$ for bump-and-reprice |
