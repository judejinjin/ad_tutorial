# Dual Numbers

## 1. Motivation: Why Dual Numbers?

In calculus, computing derivatives is fundamental. We typically have three options:

- **Symbolic differentiation** — exact but can lead to expression swell (exponentially large expressions).
- **Numerical differentiation** — easy to implement (finite differences) but suffers from truncation and round-off errors.
- **Automatic differentiation (AD)** — exact (to machine precision) and efficient. Dual numbers provide the algebraic foundation for **forward-mode AD**.

Dual numbers let us compute **exact derivatives** by simply evaluating a function — no limit process, no symbolic manipulation, and no numerical approximation.

---

## 2. Analogy: Complex Numbers → Dual Numbers

### Complex numbers

A complex number has the form:

$$
z = a + b\,i \quad \text{where } i^2 = -1
$$

The imaginary unit $i$ encodes **rotation** in the complex plane.

### Dual numbers

A dual number has the form:

$$
z = a + b\,\varepsilon \quad \text{where } \varepsilon^2 = 0, \; \varepsilon \neq 0
$$

- $a$ is called the **real part** (or primal).
- $b$ is called the **dual part** (or tangent).
- $\varepsilon$ is the **infinitesimal unit** — it is *nilpotent* (its square vanishes), but it is **not zero** itself.

> Think of $\varepsilon$ as an "infinitely small" perturbation that is algebraically non-zero but whose square is negligible.

---

## 3. Arithmetic of Dual Numbers

Let $x = a + b\,\varepsilon$ and $y = c + d\,\varepsilon$. Since $\varepsilon^2 = 0$:

### Addition

$$
x + y = (a + c) + (b + d)\,\varepsilon
$$

### Subtraction

$$
x - y = (a - c) + (b - d)\,\varepsilon
$$

### Multiplication

$$
x \cdot y = (a + b\,\varepsilon)(c + d\,\varepsilon) = ac + (ad + bc)\,\varepsilon + bd\,\varepsilon^2
$$

Since $\varepsilon^2 = 0$:

$$
x \cdot y = ac + (ad + bc)\,\varepsilon
$$

> Notice this is exactly the **product rule**: if $x$ represents a value $a$ with derivative $b$, and $y$ represents value $c$ with derivative $d$, then $x \cdot y$ has value $ac$ and derivative $ad + bc$.

### Division

$$
\frac{x}{y} = \frac{a + b\,\varepsilon}{c + d\,\varepsilon} = \frac{a}{c} + \frac{bc - ad}{c^2}\,\varepsilon
$$

(Derived by multiplying numerator and denominator by the conjugate $c - d\,\varepsilon$.)

> This matches the **quotient rule**: $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$.

---

## 4. The Key Insight: Taylor Expansion with Dual Numbers

Consider a smooth function $f$ and evaluate it at $x + \varepsilon$:

$$
f(x + \varepsilon) = f(x) + f'(x)\,\varepsilon + \frac{f''(x)}{2!}\,\varepsilon^2 + \frac{f'''(x)}{3!}\,\varepsilon^3 + \cdots
$$

Since $\varepsilon^2 = 0$ (and therefore $\varepsilon^3 = 0$, $\varepsilon^4 = 0$, etc.), **all higher-order terms vanish**:

$$
\boxed{f(x + \varepsilon) = f(x) + f'(x)\,\varepsilon}
$$

This is the central result:
- The **real part** of the output gives $f(x)$ — the function value.
- The **dual part** of the output gives $f'(x)$ — the exact derivative.

**No approximation. No limits. Exact.**

---

## 5. Worked Examples

### Example 1: $f(x) = x^2$ at $x = 3$

Evaluate $f(3 + \varepsilon)$:

$$
(3 + \varepsilon)^2 = 9 + 6\varepsilon + \varepsilon^2 = 9 + 6\varepsilon
$$

- Real part: $f(3) = 9$ ✓
- Dual part: $f'(3) = 6$ ✓  (since $f'(x) = 2x$, $f'(3) = 6$)

### Example 2: $f(x) = x^3 + 2x$ at $x = 2$

Evaluate $f(2 + \varepsilon)$:

$$
(2 + \varepsilon)^3 + 2(2 + \varepsilon)
$$

Expand $(2 + \varepsilon)^3$:

$$
= 8 + 12\varepsilon + 6\varepsilon^2 + \varepsilon^3 = 8 + 12\varepsilon
$$

So:

$$
f(2 + \varepsilon) = (8 + 12\varepsilon) + (4 + 2\varepsilon) = 12 + 14\varepsilon
$$

- Real part: $f(2) = 12$ ✓
- Dual part: $f'(2) = 14$ ✓  (since $f'(x) = 3x^2 + 2$, $f'(2) = 14$)

### Example 3: $f(x) = \sin(x)$ at $x = 0$

Using the Taylor expansion rule directly:

$$
\sin(0 + \varepsilon) = \sin(0) + \cos(0)\,\varepsilon = 0 + 1 \cdot \varepsilon = \varepsilon
$$

- Real part: $\sin(0) = 0$ ✓
- Dual part: $\cos(0) = 1$ ✓

### Example 4: Composition — $f(x) = \sin(x^2)$ at $x = 3$

Step 1: Compute $g(x) = x^2$ at $3 + \varepsilon$:

$$
(3 + \varepsilon)^2 = 9 + 6\varepsilon
$$

Step 2: Compute $\sin(9 + 6\varepsilon)$:

$$
\sin(9 + 6\varepsilon) = \sin(9) + 6\cos(9)\,\varepsilon
$$

- Real part: $\sin(9)$ ✓
- Dual part: $6\cos(9)$ ✓  (chain rule: $f'(x) = 2x\cos(x^2)$, $f'(3) = 6\cos(9)$)

> The chain rule is handled **automatically** — no special logic needed!

---

## 6. Dual Number Rules for Common Functions

To implement dual-number AD, you overload standard functions using the formula $f(a + b\varepsilon) = f(a) + b \cdot f'(a)\,\varepsilon$:

| Function         | Dual Number Rule                                                         |
| ---------------- | ------------------------------------------------------------------------ |
| $x^n$            | $a^n + n \cdot a^{n-1} \cdot b\,\varepsilon$                            |
| $\sin(x)$        | $\sin(a) + b \cdot \cos(a)\,\varepsilon$                                |
| $\cos(x)$        | $\cos(a) - b \cdot \sin(a)\,\varepsilon$                                |
| $e^x$            | $e^a + b \cdot e^a\,\varepsilon$                                        |
| $\ln(x)$         | $\ln(a) + \frac{b}{a}\,\varepsilon$                                     |
| $\sqrt{x}$       | $\sqrt{a} + \frac{b}{2\sqrt{a}}\,\varepsilon$                           |
| $\frac{1}{x}$    | $\frac{1}{a} - \frac{b}{a^2}\,\varepsilon$                              |
| $\tan(x)$        | $\tan(a) + \frac{b}{\cos^2(a)}\,\varepsilon$                            |

Each rule simply applies the standard calculus derivative — the dual part carries the derivative forward automatically.

---

## 7. Forward-Mode Automatic Differentiation

Forward-mode AD is a direct implementation of dual number arithmetic:

1. **Seed** the input: replace $x$ with $x + 1 \cdot \varepsilon$ (the dual part is $1$ because $\frac{dx}{dx} = 1$).
2. **Evaluate** the function using overloaded dual-number arithmetic.
3. **Read off** the result: the dual part of the output is $\frac{df}{dx}$.

### For multiple variables

If $f(x, y)$ and you want $\frac{\partial f}{\partial x}$:
- Set $x \to x + 1 \cdot \varepsilon$ and $y \to y + 0 \cdot \varepsilon$.

If you want $\frac{\partial f}{\partial y}$:
- Set $x \to x + 0 \cdot \varepsilon$ and $y \to y + 1 \cdot \varepsilon$.

> For $n$ inputs and $1$ output, forward-mode requires **$n$ passes** (one per input variable). This is why **reverse-mode AD** (backpropagation) is preferred for functions with many inputs and few outputs (like neural network loss functions).

---

## 8. C++ Implementation Sketch

```cpp
struct Dual {
    double real;  // f(x)   — the value
    double dual;  // f'(x)  — the derivative

    Dual(double r, double d = 0.0) : real(r), dual(d) {}
};

// Arithmetic operators
Dual operator+(Dual a, Dual b) {
    return { a.real + b.real, a.dual + b.dual };
}

Dual operator*(Dual a, Dual b) {
    return { a.real * b.real, a.real * b.dual + a.dual * b.real };
}

Dual operator/(Dual a, Dual b) {
    return { a.real / b.real,
             (a.dual * b.real - a.real * b.dual) / (b.real * b.real) };
}

// Math functions
Dual sin(Dual x) {
    return { std::sin(x.real), x.dual * std::cos(x.real) };
}

Dual cos(Dual x) {
    return { std::cos(x.real), -x.dual * std::sin(x.real) };
}

Dual exp(Dual x) {
    double ex = std::exp(x.real);
    return { ex, x.dual * ex };
}

Dual log(Dual x) {
    return { std::log(x.real), x.dual / x.real };
}

// Usage: compute f(x) = sin(x^2) and its derivative at x = 3
Dual x(3.0, 1.0);      // seed: value = 3, derivative = 1
Dual result = sin(x * x);
// result.real == sin(9)       — the function value
// result.dual == 6 * cos(9)  — the exact derivative
```

---

## 9. Geometric Interpretation

| Number System | Algebra           | Geometry                                    |
| ------------- | ----------------- | ------------------------------------------- |
| Real          | $a$               | Points on a line                            |
| Complex       | $a + bi$          | Points in a plane (rotation + scaling)      |
| Dual          | $a + b\varepsilon$| A point with a **tangent direction** attached|

Dual numbers represent **jets** — a point together with a velocity/direction. Evaluating $f$ on a dual number propagates both the point and its tangent through the function, which is exactly what a derivative does.

---

---

# Reverse-Mode Automatic Differentiation

---

## 10. Why Reverse Mode?

Forward-mode AD (dual numbers) computes one directional derivative per pass. If $f : \mathbb{R}^n \to \mathbb{R}^m$:

| Mode    | Cost per pass           | Passes needed for full Jacobian |
| ------- | ----------------------- | ------------------------------- |
| Forward | $O(\text{cost of } f)$  | $n$ (one per input)             |
| Reverse | $O(\text{cost of } f)$  | $m$ (one per output)            |

For a neural network loss function with millions of parameters ($n \gg 1$) and a single scalar loss ($m = 1$):
- Forward mode: **millions of passes** — impractical.
- Reverse mode: **1 pass** — this is why backpropagation works.

> **Reverse-mode AD computes the gradient of a scalar function with respect to ALL inputs in a single backward pass.**

---

## 11. The Computation Graph (Wengert Tape)

Every computation can be decomposed into a sequence of elementary operations. We record these as a **directed acyclic graph (DAG)** called the **computation graph** or **Wengert list/tape**.

### Example: $f(x_1, x_2) = x_1 \cdot x_2 + \sin(x_1)$

We introduce intermediate variables for each elementary operation:

$$
\begin{aligned}
v_1 &= x_1 &\quad &\text{(input)} \\
v_2 &= x_2 &\quad &\text{(input)} \\
v_3 &= v_1 \cdot v_2 &\quad &\text{(multiply)} \\
v_4 &= \sin(v_1) &\quad &\text{(sin)} \\
v_5 &= v_3 + v_4 &\quad &\text{(add)} \\
y &= v_5 &\quad &\text{(output)}
\end{aligned}
$$

The graph looks like:

```
x1 (v1) ──┬──→ [×] ──→ v3 ──┐
           │                  ├──→ [+] ──→ v5 = y
           └──→ [sin] ──→ v4 ┘
x2 (v2) ──────→ [×]
```

Each node knows:
- Its **value** (computed during the forward pass).
- Which **inputs** it depends on (for routing gradients backward).
- The **local derivative** $\frac{\partial v_i}{\partial v_j}$ for each input $v_j$.

---

## 12. Forward Pass: Computing Values

We evaluate the graph from inputs to output, storing every intermediate value.

Using $x_1 = 2, x_2 = 3$:

| Variable | Expression         | Value                  |
| -------- | ------------------ | ---------------------- |
| $v_1$    | $x_1$              | $2$                    |
| $v_2$    | $x_2$              | $3$                    |
| $v_3$    | $v_1 \cdot v_2$    | $2 \times 3 = 6$      |
| $v_4$    | $\sin(v_1)$        | $\sin(2) \approx 0.909$ |
| $v_5$    | $v_3 + v_4$        | $6 + 0.909 = 6.909$   |

Result: $f(2, 3) \approx 6.909$.

> The forward pass is identical to just running the function — we simply **record** each operation on a tape as we go.

---

## 13. Backward Pass: Computing Gradients (Adjoints)

### The adjoint

For each intermediate variable $v_i$, we define its **adjoint** (also called the **sensitivity** or **grad**):

$$
\bar{v}_i = \frac{\partial y}{\partial v_i}
$$

This tells us: "How much does the final output $y$ change if $v_i$ changes by a tiny amount?"

### Why "adjoint"?

The word comes from linear algebra / functional analysis. Given a linear map $A : V \to W$, its **adjoint** (or transpose, for real spaces) is the unique map $A^{\top} : W \to V$ satisfying:

$$
\langle A\mathbf{u},\, \mathbf{v} \rangle = \langle \mathbf{u},\, A^{\top}\mathbf{v} \rangle \quad \text{for all } \mathbf{u} \in V,\; \mathbf{v} \in W
$$

The Jacobian $J$ of $f$ at a point is exactly such a linear map $J : \mathbb{R}^n \to \mathbb{R}^m$. The forward pass applies $J$ (a JVP):

$$
\dot{\mathbf{y}} = J\,\dot{\mathbf{x}}
$$

The reverse pass applies the **adjoint** $J^{\top}$ (a VJP):

$$
\bar{\mathbf{x}} = J^{\top}\bar{\mathbf{y}}
$$

The inner-product relationship is the key: for any forward tangent $\dot{\mathbf{x}}$ and reverse adjoint $\bar{\mathbf{y}}$,

$$
\langle \dot{\mathbf{y}},\, \bar{\mathbf{y}} \rangle = \langle J\dot{\mathbf{x}},\, \bar{\mathbf{y}} \rangle = \langle \dot{\mathbf{x}},\, J^{\top}\bar{\mathbf{y}} \rangle = \langle \dot{\mathbf{x}},\, \bar{\mathbf{x}} \rangle
$$

In words: **the inner product of the output tangent with the output adjoint equals the inner product of the input tangent with the input adjoint**. This symmetry is the defining property of adjoint operators, and it is exactly what the chain rule preserves layer by layer.

The same terminology appears independently in optimal control (the *adjoint state method* / *co-state equation*), PDE-constrained optimisation, and quantum mechanics — all contexts where you need to propagate sensitivities backward through a system. Reverse-mode AD is the computational realisation of this same idea applied to arbitrary programs.

> **Summary of names for the same concept:**
> - *Adjoint* — functional analysis / AD literature
> - *Co-state* — optimal control (Pontryagin)
> - *Lagrange multiplier* — constrained optimisation
> - *Sensitivity* — engineering / finance
> - *Gradient* — machine learning (when the output is a scalar loss)

### The backward propagation rule

By the chain rule, if $v_i$ feeds into nodes $v_j, v_k, \ldots$, then:

$$
\bar{v}_i = \sum_{j \in \text{children}(i)} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_i}
$$

In words: **accumulate** the gradients flowing back from all nodes that $v_i$ contributes to, each scaled by the local derivative.

### Step-by-step backward pass for our example

**Start at the output:**

$$
\bar{v}_5 = \frac{\partial y}{\partial v_5} = 1 \quad \text{(seed: the output's gradient w.r.t. itself is always 1)}
$$

**Backpropagate through $v_5 = v_3 + v_4$:**

$$
\bar{v}_3 = \bar{v}_5 \cdot \frac{\partial v_5}{\partial v_3} = 1 \cdot 1 = 1
$$
$$
\bar{v}_4 = \bar{v}_5 \cdot \frac{\partial v_5}{\partial v_4} = 1 \cdot 1 = 1
$$

(Addition distributes gradient equally.)

**Backpropagate through $v_4 = \sin(v_1)$:**

$$
\bar{v}_1 \mathrel{+}= \bar{v}_4 \cdot \frac{\partial v_4}{\partial v_1} = 1 \cdot \cos(v_1) = \cos(2) \approx -0.416
$$

**Backpropagate through $v_3 = v_1 \cdot v_2$:**

$$
\bar{v}_1 \mathrel{+}= \bar{v}_3 \cdot \frac{\partial v_3}{\partial v_1} = 1 \cdot v_2 = 3
$$
$$
\bar{v}_2 \mathrel{+}= \bar{v}_3 \cdot \frac{\partial v_3}{\partial v_2} = 1 \cdot v_1 = 2
$$

**Final gradients:**

$$
\bar{v}_1 = \frac{\partial y}{\partial x_1} = 3 + \cos(2) \approx 3 - 0.416 = 2.584
$$
$$
\bar{v}_2 = \frac{\partial y}{\partial x_2} = 2
$$

**Verification:** $f(x_1, x_2) = x_1 x_2 + \sin(x_1)$, so:
- $\frac{\partial f}{\partial x_1} = x_2 + \cos(x_1) = 3 + \cos(2) \approx 2.584$ ✓
- $\frac{\partial f}{\partial x_2} = x_1 = 2$ ✓

> **Both partial derivatives computed in a single backward pass!**

---

## 14. Local Derivative Rules for Backward Pass

Each elementary operation has a simple local rule for how it routes gradients backward:

| Operation              | Forward              | Backward: $\bar{v}_i \mathrel{+}= \ldots$                                |
| ---------------------- | -------------------- | ------------------------------------------------------------------------- |
| $v_k = v_i + v_j$     | sum                  | $\bar{v}_i \mathrel{+}= \bar{v}_k$, $\bar{v}_j \mathrel{+}= \bar{v}_k$                   |
| $v_k = v_i - v_j$     | subtract             | $\bar{v}_i \mathrel{+}= \bar{v}_k$, $\bar{v}_j \mathrel{+}= -\bar{v}_k$                  |
| $v_k = v_i \cdot v_j$ | multiply             | $\bar{v}_i \mathrel{+}= \bar{v}_k \cdot v_j$, $\bar{v}_j \mathrel{+}= \bar{v}_k \cdot v_i$ |
| $v_k = v_i / v_j$     | divide               | $\bar{v}_i \mathrel{+}= \bar{v}_k / v_j$, $\bar{v}_j \mathrel{+}= -\bar{v}_k \cdot v_i / v_j^2$ |
| $v_k = \sin(v_i)$     | sin                  | $\bar{v}_i \mathrel{+}= \bar{v}_k \cdot \cos(v_i)$                               |
| $v_k = \cos(v_i)$     | cos                  | $\bar{v}_i \mathrel{+}= -\bar{v}_k \cdot \sin(v_i)$                              |
| $v_k = e^{v_i}$       | exp                  | $\bar{v}_i \mathrel{+}= \bar{v}_k \cdot e^{v_i}$                                 |
| $v_k = \ln(v_i)$      | log                  | $\bar{v}_i \mathrel{+}= \bar{v}_k / v_i$                                         |
| $v_k = v_i^n$         | power                | $\bar{v}_i \mathrel{+}= \bar{v}_k \cdot n \cdot v_i^{n-1}$                       |
| $v_k = \sqrt{v_i}$    | sqrt                 | $\bar{v}_i \mathrel{+}= \bar{v}_k / (2\sqrt{v_i})$                               |

> Key pattern: the local derivative is the same calculus derivative you already know — reverse mode just applies them **in reverse order** and **accumulates** with $\mathrel{+}=$.

---

## 15. A Larger Worked Example: Multi-Layer Composition

### $f(x) = e^{\sin(x^2)}$ at $x = 1$

**Forward pass** (build tape):

| Step | Variable | Expression      | Value                              |
| ---- | -------- | --------------- | ---------------------------------- |
| 1    | $v_1$    | $x$             | $1$                                |
| 2    | $v_2$    | $v_1^2$         | $1$                                |
| 3    | $v_3$    | $\sin(v_2)$     | $\sin(1) \approx 0.8415$          |
| 4    | $v_4$    | $e^{v_3}$       | $e^{0.8415} \approx 2.3198$       |

**Backward pass** (reverse through tape):

$$
\bar{v}_4 = 1 \quad \text{(seed)}
$$

$$
\bar{v}_3 = \bar{v}_4 \cdot e^{v_3} = 1 \cdot 2.3198 = 2.3198 \quad \text{(exp rule)}
$$

$$
\bar{v}_2 = \bar{v}_3 \cdot \cos(v_2) = 2.3198 \cdot \cos(1) \approx 2.3198 \cdot 0.5403 = 1.2534 \quad \text{(sin rule)}
$$

$$
\bar{v}_1 = \bar{v}_2 \cdot 2v_1 = 1.2534 \cdot 2 = 2.5068 \quad \text{(power rule)}
$$

**Result:** $\frac{df}{dx}\bigg|_{x=1} \approx 2.5068$

**Verification by chain rule:**

$$
f'(x) = e^{\sin(x^2)} \cdot \cos(x^2) \cdot 2x
$$
$$
f'(1) = e^{\sin(1)} \cdot \cos(1) \cdot 2 \approx 2.3198 \times 0.5403 \times 2 \approx 2.5068 \; ✓
$$

---

## 16. The Wengert Tape Data Structure

In implementation, each tape entry stores:

```
TapeEntry {
    op:       operation type (ADD, MUL, SIN, EXP, ...)
    inputs:   indices of input nodes on the tape
    value:    computed value from forward pass
    adjoint:  ∂y/∂(this node), accumulated during backward pass
}
```

The tape is a **linear array** of these entries. The forward pass appends to it; the backward pass iterates it **in reverse**.

```
Forward:  tape[0] → tape[1] → tape[2] → ... → tape[N]
Backward: tape[N] → tape[N-1] → ... → tape[1] → tape[0]
```

---

## 17. Reverse-Mode AD: The Algorithm

```
ALGORITHM: Reverse-Mode AD for f : R^n → R

INPUT:  Function f, input values x₁, ..., xₙ
OUTPUT: Gradient (∂f/∂x₁, ..., ∂f/∂xₙ)

1. FORWARD PASS:
   - Evaluate f(x₁, ..., xₙ) from inputs to output.
   - Record each elementary operation on the tape.
   - Store all intermediate values.

2. INITIALIZE:
   - Set ȳ = 1  (adjoint of the output)
   - Set all other adjoints to 0

3. BACKWARD PASS (iterate tape in reverse):
   For each operation v_k = op(v_i, v_j, ...) in reverse order:
     For each input v_i of this operation:
       v̄_i += v̄_k × (∂v_k / ∂v_i)    // local derivative × upstream adjoint

4. RETURN:
   - The adjoints of the input nodes: (v̄₁, ..., v̄ₙ) = (∂f/∂x₁, ..., ∂f/∂xₙ)
```

---

## 18. Fan-Out and the Multivariate Chain Rule

A critical subtlety: when a variable is **used more than once** (fan-out), gradients from all uses must be **summed**. This is why we use $\mathrel{+}=$ rather than $=$.

### Example: $f(x) = x \cdot x$ (i.e., $x^2$ but computed as a product)

**Forward pass:**

$$
v_1 = x, \quad v_2 = v_1 \cdot v_1
$$

**Backward pass:**

$$
\bar{v}_2 = 1
$$

From the multiplication rule ($v_2 = v_1 \cdot v_1$), $v_1$ appears as **both** inputs:

$$
\bar{v}_1 \mathrel{+}= \bar{v}_2 \cdot v_1 = 1 \cdot x = x \quad \text{(from the "right" input)}
$$
$$
\bar{v}_1 \mathrel{+}= \bar{v}_2 \cdot v_1 = 1 \cdot x = x \quad \text{(from the "left" input)}
$$
$$
\bar{v}_1 = 2x \; ✓
$$

> If you forget to accumulate ($\mathrel{+}=$) and instead overwrite ($=$), you get $\bar{v}_1 = x$ — **wrong!** This is a common implementation bug.

---

## 19. Forward vs. Reverse Mode Comparison

| Aspect                      | Forward Mode (Dual Numbers)           | Reverse Mode (Backpropagation)        |
| --------------------------- | ------------------------------------- | ------------------------------------- |
| **Direction**               | Input → Output                        | Output → Input                        |
| **Propagates**              | Tangents (perturbations)              | Adjoints (sensitivities)              |
| **Computes per pass**       | One column of the Jacobian            | One row of the Jacobian               |
| **Cost for $f: \mathbb{R}^n \to \mathbb{R}$** | $n$ passes             | **1 pass**                            |
| **Cost for $f: \mathbb{R} \to \mathbb{R}^m$** | **1 pass**              | $m$ passes                            |
| **Memory**                  | $O(1)$ extra (no tape needed)         | $O(T)$ — must store entire tape       |
| **Implementation**          | Operator overloading (simple)         | Tape + reverse traversal (complex)    |
| **Algebraic basis**         | Dual numbers ($\varepsilon^2 = 0$)    | Chain rule on computation graph       |
| **Best for**                | Few inputs, many outputs              | Many inputs, few outputs (e.g., ML)   |
| **Used in**                 | Sensitivity analysis, ODE solvers     | PyTorch, TensorFlow, JAX              |

### Why the tangent and the gradient look the same for scalar functions

For $f : \mathbb{R} \to \mathbb{R}$ the two quantities both reduce to the ordinary derivative $f'(x)$, which is why they appear identical in single-variable examples.

**Forward mode — tangent** is a *directional derivative*. You inject a seed $\dot{x}$ into the input's dual part and the chain rule pushes it forward:

$$
\dot{y} = \frac{\partial f}{\partial x} \cdot \dot{x}
$$

Setting $\dot{x} = 1$ gives `y.dual` $= f'(x)$.

**Reverse mode — adjoint** is a *sensitivity*. You seed the output $\bar{y} = 1$ and the tape propagates it backward:

$$
\bar{x} = \bar{y} \cdot \frac{\partial f}{\partial x}
$$

Setting $\bar{y} = 1$ gives `x.adjoint()` $= f'(x)$.

Same number, different mathematical objects. The difference becomes clear for $f : \mathbb{R}^n \to \mathbb{R}^m$, where the Jacobian $J$ is $m \times n$:

| Mode    | One pass computes         | Passes for full $J$ | Cheap when        |
| ------- | ------------------------- | ------------------- | ----------------- |
| Forward | One **column** of $J$ (fix one $\dot{x}_i = 1$) | $n$ | $n \ll m$ |
| Reverse | One **row** of $J$ (fix one $\bar{y}_j = 1$)    | $m$ | $m \ll n$ |

For the common ML case $f : \mathbb{R}^n \to \mathbb{R}$ ($m = 1$), reverse mode delivers the full gradient $\nabla f \in \mathbb{R}^n$ in a **single pass**, while forward mode would require $n$ passes — one per input dimension.

### The general case: $f : \mathbb{R}^n \to \mathbb{R}^m$

Let $\mathbf{x} \in \mathbb{R}^n$ and $\mathbf{y} = f(\mathbf{x}) \in \mathbb{R}^m$. The full Jacobian is:

$$
J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix} \in \mathbb{R}^{m \times n}
$$

The two modes compute fundamentally different matrix-vector products involving $J$.

#### Forward mode — Jacobian-Vector Product (JVP)

You seed a tangent vector $\dot{\mathbf{x}} \in \mathbb{R}^n$ on the inputs. The forward pass propagates it to a tangent vector $\dot{\mathbf{y}} \in \mathbb{R}^m$:

$$
\dot{\mathbf{y}} = J\,\dot{\mathbf{x}}
$$

This is a **left-multiplication** of $J$ by a column vector. Each component of $\dot{\mathbf{y}}$ is:

$$
\dot{y}_j = \sum_{i=1}^{n} \frac{\partial y_j}{\partial x_i}\,\dot{x}_i
$$

To extract the $k$-th **column** of $J$, set $\dot{\mathbf{x}} = \mathbf{e}_k$ (the $k$-th standard basis vector, i.e. $\dot{x}_k = 1$, all others $0$). The output $\dot{\mathbf{y}}$ is then $J_{:,k}$ — the partial derivatives of every output with respect to $x_k$.

Recovering the full $J$ therefore requires $n$ forward passes, one per input:

$$
J = \bigl[\; J\mathbf{e}_1 \;\big|\; J\mathbf{e}_2 \;\big|\; \cdots \;\big|\; J\mathbf{e}_n \;\bigr]
$$

#### Reverse mode — Vector-Jacobian Product (VJP)

You seed an adjoint (co-tangent) vector $\bar{\mathbf{y}} \in \mathbb{R}^m$ on the outputs. The backward pass propagates it to an adjoint vector $\bar{\mathbf{x}} \in \mathbb{R}^n$:

$$
\bar{\mathbf{x}} = J^{\top}\,\bar{\mathbf{y}}
$$

This is a **right-multiplication** of $J^{\top}$ by a column vector (equivalently, $\bar{\mathbf{y}}^{\top} J$). Each component is:

$$
\bar{x}_i = \sum_{j=1}^{m} \frac{\partial y_j}{\partial x_i}\,\bar{y}_j
$$

To extract the $k$-th **row** of $J$, set $\bar{\mathbf{y}} = \mathbf{e}_k$ (i.e. $\bar{y}_k = 1$, all others $0$). The output $\bar{\mathbf{x}}$ is then $J_{k,:}$ — the partial derivatives of $y_k$ with respect to every input.

Recovering the full $J$ therefore requires $m$ reverse passes, one per output:

$$
J^{\top} = \bigl[\; J^{\top}\mathbf{e}_1 \;\big|\; J^{\top}\mathbf{e}_2 \;\big|\; \cdots \;\big|\; J^{\top}\mathbf{e}_m \;\bigr]
$$

#### Concrete example: $f : \mathbb{R}^2 \to \mathbb{R}^2$

Let $f(x_1, x_2) = (x_1^2 + x_2,\; x_1 x_2)$ at $(x_1, x_2) = (2, 3)$.

The Jacobian is:

$$
J = \begin{pmatrix} 2x_1 & 1 \\ x_2 & x_1 \end{pmatrix} = \begin{pmatrix} 4 & 1 \\ 3 & 2 \end{pmatrix}
$$

**Forward pass 1** — seed $\dot{\mathbf{x}} = (1, 0)^{\top}$ (perturb $x_1$):

$$
\dot{\mathbf{y}} = J\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}4\\3\end{pmatrix} \quad \leftarrow \text{column 1 of } J
$$

**Forward pass 2** — seed $\dot{\mathbf{x}} = (0, 1)^{\top}$ (perturb $x_2$):

$$
\dot{\mathbf{y}} = J\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}1\\2\end{pmatrix} \quad \leftarrow \text{column 2 of } J
$$

Two forward passes → both columns → full $J$. ✓

**Reverse pass 1** — seed $\bar{\mathbf{y}} = (1, 0)^{\top}$ (sensitivity of $y_1$):

$$
\bar{\mathbf{x}} = J^{\top}\begin{pmatrix}1\\0\end{pmatrix} = \begin{pmatrix}4\\1\end{pmatrix} \quad \leftarrow \text{row 1 of } J \text{ (transposed)}
$$

**Reverse pass 2** — seed $\bar{\mathbf{y}} = (0, 1)^{\top}$ (sensitivity of $y_2$):

$$
\bar{\mathbf{x}} = J^{\top}\begin{pmatrix}0\\1\end{pmatrix} = \begin{pmatrix}3\\2\end{pmatrix} \quad \leftarrow \text{row 2 of } J \text{ (transposed)}
$$

Two reverse passes → both rows → full $J$. ✓

Here $n = m = 2$ so both modes cost the same. The economics shift when $n \neq m$:

- **$n \gg m$** (many inputs, few outputs — e.g. a scalar loss over millions of parameters): reverse mode wins. One reverse pass gives $\nabla f$ directly; forward mode needs $n$ passes.
- **$m \gg n$** (few inputs, many outputs — e.g. a physics simulation outputting a large state vector from a handful of control parameters): forward mode wins. $n$ passes give the full Jacobian; reverse mode needs $m$ passes.

#### Why the seed vectors encode different things

| | Forward seed $\dot{\mathbf{x}}$ | Reverse seed $\bar{\mathbf{y}}$ |
|---|---|---|
| **Lives in** | input space $\mathbb{R}^n$ | output space $\mathbb{R}^m$ |
| **Encodes** | "which input direction to perturb" | "how much we care about each output" |
| **Result** | effect on all outputs | attribution to all inputs |
| **Computes** | $J\dot{\mathbf{x}}$ (a column-combination of $J$) | $J^{\top}\bar{\mathbf{y}}$ (a row-combination of $J$) |

This is why they are different objects even though their values coincide when $n = m = 1$: in the scalar case $J$ is a $1 \times 1$ matrix, so $J\dot{x}$ and $J^{\top}\bar{y}$ are both just multiplication by the same single number $f'(x)$.

---

## 20. Connection to Neural Network Backpropagation

Backpropagation in deep learning **is** reverse-mode AD applied to the loss function.

A neural network computes:

$$
L = \text{loss}(\text{softmax}(W_3 \cdot \sigma(W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2) + b_3), \; y_{\text{true}})
$$

This is a deeply nested composition of operations. The parameters $W_1, W_2, W_3, b_1, b_2, b_3$ may number in the millions.

- **Forward pass**: Compute $L$ given inputs and current parameters (standard inference).
- **Backward pass**: Compute $\frac{\partial L}{\partial W_i}$ and $\frac{\partial L}{\partial b_i}$ for **all** parameters in **one** reverse sweep.

The gradient $\nabla L$ is then used for gradient descent:

$$
W_i \leftarrow W_i - \eta \cdot \frac{\partial L}{\partial W_i}
$$

> Backpropagation was independently discovered multiple times. The connection to reverse-mode AD was made explicit by Speelpenning (1980) and popularized by Rumelhart, Hinton & Williams (1986).

---

## 21. Memory-Computation Tradeoff: Checkpointing

Reverse mode's main drawback is **memory**: the tape stores every intermediate value from the forward pass. For deep networks or long sequences, this can exhaust GPU memory.

**Gradient checkpointing** (also called **rematerialization**) trades memory for compute:

1. During the forward pass, only store values at selected **checkpoints** (e.g., every $k$-th layer).
2. During the backward pass, **recompute** intermediate values from the nearest checkpoint as needed.

| Strategy         | Memory   | Compute overhead |
| ---------------- | -------- | ---------------- |
| Store everything | $O(T)$  | $1\times$        |
| Checkpoint every $\sqrt{T}$ layers | $O(\sqrt{T})$ | $\leq 2\times$ |
| Store nothing (recompute all) | $O(1)$ | $O(T)\times$    |

> The $O(\sqrt{T})$ strategy (Griewank & Walther) is the sweet spot used in practice by PyTorch (`torch.utils.checkpoint`) and TensorFlow.

---

## 22. C++ Implementation Sketch: Reverse-Mode AD

```cpp
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>

struct Var;
struct TapeEntry {
    double value;
    double adjoint = 0.0;
    // Backward function: given the adjoint of this node,
    // propagate gradients to its inputs.
    std::function<void(double)> backward;
};

// Global tape
std::vector<TapeEntry> tape;

struct Var {
    int index;  // index into the tape

    double value() const { return tape[index].value; }
    double& adjoint() const { return tape[index].adjoint; }

    // Create a leaf variable (input)
    explicit Var(double val) {
        index = tape.size();
        tape.push_back({ val, 0.0, [](double) {} });
    }

    // Create a computed variable
    Var(double val, std::function<void(double)> bwd) {
        index = tape.size();
        tape.push_back({ val, 0.0, bwd });
    }
};

// Addition
Var operator+(Var a, Var b) {
    double val = a.value() + b.value();
    return Var(val, [a, b](double grad) {
        tape[a.index].adjoint += grad;       // ∂(a+b)/∂a = 1
        tape[b.index].adjoint += grad;       // ∂(a+b)/∂b = 1
    });
}

// Multiplication
Var operator*(Var a, Var b) {
    double val = a.value() * b.value();
    double a_val = a.value(), b_val = b.value();
    return Var(val, [a, b, a_val, b_val](double grad) {
        tape[a.index].adjoint += grad * b_val;  // ∂(a*b)/∂a = b
        tape[b.index].adjoint += grad * a_val;  // ∂(a*b)/∂b = a
    });
}

// Subtraction
Var operator-(Var a, Var b) {
    double val = a.value() - b.value();
    return Var(val, [a, b](double grad) {
        tape[a.index].adjoint += grad;        // ∂(a-b)/∂a = 1
        tape[b.index].adjoint += -grad;       // ∂(a-b)/∂b = -1
    });
}

// Sin
Var sin(Var a) {
    double val = std::sin(a.value());
    double a_val = a.value();
    return Var(val, [a, a_val](double grad) {
        tape[a.index].adjoint += grad * std::cos(a_val);
    });
}

// Cos
Var cos(Var a) {
    double val = std::cos(a.value());
    double a_val = a.value();
    return Var(val, [a, a_val](double grad) {
        tape[a.index].adjoint += grad * (-std::sin(a_val));
    });
}

// Exp
Var exp(Var a) {
    double val = std::exp(a.value());
    return Var(val, [a, val](double grad) {
        tape[a.index].adjoint += grad * val;
    });
}

// Log
Var log(Var a) {
    double val = std::log(a.value());
    double a_val = a.value();
    return Var(val, [a, a_val](double grad) {
        tape[a.index].adjoint += grad / a_val;
    });
}

// Backward pass: call after computing the output
void backward(Var output) {
    tape[output.index].adjoint = 1.0;  // seed
    for (int i = output.index; i >= 0; --i) {
        if (tape[i].adjoint != 0.0) {
            tape[i].backward(tape[i].adjoint);
        }
    }
}

// Usage
int main() {
    tape.clear();

    Var x1(2.0);   // tape[0]
    Var x2(3.0);   // tape[1]

    // f(x1, x2) = x1 * x2 + sin(x1)
    Var y = x1 * x2 + sin(x1);

    backward(y);

    std::cout << "f(2,3) = " << y.value() << "\n";
    std::cout << "∂f/∂x1 = " << x1.adjoint()
              << " (expected " << 3 + std::cos(2.0) << ")\n";
    std::cout << "∂f/∂x2 = " << x2.adjoint()
              << " (expected 2)\n";
    return 0;
}
```

**Output:**
```
f(2,3) = 6.9093
∂f/∂x1 = 2.58385 (expected 2.58385)
∂f/∂x2 = 2 (expected 2)
```

---

---

# Computing the Full Jacobian Matrix with AD

---

## 23. What Is the Jacobian?

For a vector-valued function $\mathbf{f} : \mathbb{R}^n \to \mathbb{R}^m$ with inputs $\mathbf{x} = (x_1, \ldots, x_n)$ and outputs $\mathbf{y} = (y_1, \ldots, y_m)$:

$$
\mathbf{f}(\mathbf{x}) = \begin{pmatrix} f_1(x_1, \ldots, x_n) \\ f_2(x_1, \ldots, x_n) \\ \vdots \\ f_m(x_1, \ldots, x_n) \end{pmatrix}
$$

The **Jacobian matrix** $J \in \mathbb{R}^{m \times n}$ contains all first-order partial derivatives:

$$
J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} =
\begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$

- Row $i$: the gradient of the $i$-th output $\nabla f_i$.
- Column $j$: the sensitivity of all outputs to the $j$-th input.

---

## 24. Jacobian-Vector Product (JVP) — Forward Mode

Forward-mode AD does **not** compute the Jacobian directly. Instead, it computes a **Jacobian-vector product** (JVP):

$$
J \cdot \mathbf{v} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \cdot \mathbf{v} \in \mathbb{R}^m
$$

where $\mathbf{v} \in \mathbb{R}^n$ is the **seed** (tangent) vector.

### How it works

Seed the inputs with dual parts set to $\mathbf{v}$:

$$
\mathbf{x} + \mathbf{v}\,\varepsilon = \begin{pmatrix} x_1 + v_1\,\varepsilon \\ x_2 + v_2\,\varepsilon \\ \vdots \\ x_n + v_n\,\varepsilon \end{pmatrix}
$$

Evaluate $\mathbf{f}(\mathbf{x} + \mathbf{v}\,\varepsilon)$. The dual parts of all outputs give:

$$
\text{dual}(\mathbf{f}(\mathbf{x} + \mathbf{v}\,\varepsilon)) = J \cdot \mathbf{v}
$$

### Extracting a column of $J$

To get the $j$-th column of $J$ (all $\frac{\partial f_i}{\partial x_j}$), set $\mathbf{v} = \mathbf{e}_j$ (the $j$-th standard basis vector):

$$
J \cdot \mathbf{e}_j = \begin{pmatrix} \frac{\partial f_1}{\partial x_j} \\ \frac{\partial f_2}{\partial x_j} \\ \vdots \\ \frac{\partial f_m}{\partial x_j} \end{pmatrix} = \text{column } j \text{ of } J
$$

> **One forward-mode pass = one column of the Jacobian.**

---

## 25. Vector-Jacobian Product (VJP) — Reverse Mode

Reverse-mode AD computes a **vector-Jacobian product** (VJP):

$$
\mathbf{u}^\top \cdot J = \mathbf{u}^\top \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^n
$$

where $\mathbf{u} \in \mathbb{R}^m$ is the **seed** (cotangent) vector.

### How it works

Run the forward pass to compute $\mathbf{f}(\mathbf{x})$ and record the tape. Then seed the output adjoints with $\mathbf{u}$:

$$
\bar{y}_i = u_i \quad \text{for } i = 1, \ldots, m
$$

Run the backward pass. The input adjoints give:

$$
(\bar{x}_1, \ldots, \bar{x}_n) = \mathbf{u}^\top \cdot J
$$

### Extracting a row of $J$

To get the $i$-th row of $J$ (the gradient $\nabla f_i$), set $\mathbf{u} = \mathbf{e}_i$:

$$
\mathbf{e}_i^\top \cdot J = \begin{pmatrix} \frac{\partial f_i}{\partial x_1} & \frac{\partial f_i}{\partial x_2} & \cdots & \frac{\partial f_i}{\partial x_n} \end{pmatrix} = \text{row } i \text{ of } J
$$

> **One reverse-mode pass = one row of the Jacobian.**

---

## 26. Building the Full Jacobian

### Strategy 1: Forward mode — $n$ passes (column by column)

```
ALGORITHM: Full Jacobian via Forward-Mode AD

INPUT:  f : R^n → R^m, evaluation point x
OUTPUT: Jacobian matrix J ∈ R^(m×n)

For j = 1 to n:
    Seed:  x_i → (x_i, δ_ij)   for all i    // δ_ij = 1 if i=j, else 0
    Evaluate f with dual numbers
    J[:, j] = dual parts of all m outputs     // column j
```

**Cost:** $n$ evaluations of $f$ with dual numbers.

**Best when:** $n \ll m$ (few inputs, many outputs).

### Strategy 2: Reverse mode — $m$ passes (row by row)

```
ALGORITHM: Full Jacobian via Reverse-Mode AD

INPUT:  f : R^n → R^m, evaluation point x
OUTPUT: Jacobian matrix J ∈ R^(m×n)

1. FORWARD PASS: Evaluate f(x), record tape (done once)

2. For i = 1 to m:
     Reset all adjoints to 0
     Seed:  ȳ_k = δ_ik   for all k    // 1 for output i, 0 elsewhere
     Run backward pass
     J[i, :] = (x̄_1, ..., x̄_n)       // row i

```

**Cost:** 1 forward pass + $m$ backward passes.

**Best when:** $m \ll n$ (few outputs, many inputs).

### Strategy 3: Choose the cheaper mode

$$
\text{Total cost} = \begin{cases}
n \times \text{cost}(f) & \text{forward mode} \\
(1 + m) \times \text{cost}(f) & \text{reverse mode}
\end{cases}
$$

| Scenario   | $n$ (inputs) | $m$ (outputs) | Best mode | Passes |
| ---------- | ------------ | ------------- | --------- | ------ |
| Gradient   | 1000         | 1             | Reverse   | 1+1=2  |
| Sensitivity| 3            | 100           | Forward   | 3      |
| Square     | 50           | 50            | Either    | ~50    |

---

## 27. Worked Example: Full Jacobian

### Function: $\mathbf{f}(x_1, x_2) = \begin{pmatrix} x_1 \cdot x_2 \\ \sin(x_1) + x_2^2 \end{pmatrix}$ at $(x_1, x_2) = (2, 3)$

The true Jacobian is:

$$
J = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2}
\end{pmatrix}
= \begin{pmatrix}
x_2 & x_1 \\
\cos(x_1) & 2x_2
\end{pmatrix}
= \begin{pmatrix}
3 & 2 \\
\cos(2) & 6
\end{pmatrix}
\approx \begin{pmatrix}
3 & 2 \\
-0.416 & 6
\end{pmatrix}
$$

### Via forward mode (2 passes for 2 inputs)

**Pass 1:** Seed $\mathbf{v} = (1, 0)$ — get column 1:

$$
x_1 = 2 + 1\cdot\varepsilon, \quad x_2 = 3 + 0\cdot\varepsilon
$$

$$
f_1 = x_1 \cdot x_2 = (2 + \varepsilon)(3) = 6 + 3\varepsilon \quad \Rightarrow \quad J_{11} = 3
$$

$$
f_2 = \sin(x_1) + x_2^2 = \sin(2 + \varepsilon) + 9 = (\sin 2 + \cos 2 \cdot \varepsilon) + 9 \quad \Rightarrow \quad J_{21} = \cos(2)
$$

$$
J[:, 1] = \begin{pmatrix} 3 \\ \cos(2) \end{pmatrix} ✓
$$

**Pass 2:** Seed $\mathbf{v} = (0, 1)$ — get column 2:

$$
x_1 = 2 + 0\cdot\varepsilon, \quad x_2 = 3 + 1\cdot\varepsilon
$$

$$
f_1 = (2)(3 + \varepsilon) = 6 + 2\varepsilon \quad \Rightarrow \quad J_{12} = 2
$$

$$
f_2 = \sin(2) + (3 + \varepsilon)^2 = \sin(2) + 9 + 6\varepsilon \quad \Rightarrow \quad J_{22} = 6
$$

$$
J[:, 2] = \begin{pmatrix} 2 \\ 6 \end{pmatrix} ✓
$$

**Full Jacobian assembled:**

$$
J = \begin{pmatrix} 3 & 2 \\ \cos(2) & 6 \end{pmatrix} ✓
$$

### Via reverse mode (2 passes for 2 outputs)

**Forward pass:** Evaluate at $(2, 3)$, record tape.

**Backward pass 1:** Seed $\bar{y}_1 = 1, \bar{y}_2 = 0$ — get row 1:

Only $f_1 = x_1 \cdot x_2$ contributes.

$$
\bar{x}_1 = x_2 = 3, \quad \bar{x}_2 = x_1 = 2
$$

$$
J[1, :] = (3, 2) ✓
$$

**Backward pass 2:** Seed $\bar{y}_1 = 0, \bar{y}_2 = 1$ — get row 2:

Only $f_2 = \sin(x_1) + x_2^2$ contributes.

$$
\bar{x}_1 = \cos(x_1) = \cos(2), \quad \bar{x}_2 = 2x_2 = 6
$$

$$
J[2, :] = (\cos(2), 6) ✓
$$

**Same Jacobian, different traversal order.**

---

## 28. JVP and VJP: The Dual Operations

JVP and VJP are **transpose** operations of each other. They are the fundamental building blocks:

$$
\underbrace{J \cdot \mathbf{v}}_{\text{JVP (forward)}} \in \mathbb{R}^m
\qquad \qquad
\underbrace{\mathbf{u}^\top \cdot J}_{\text{VJP (reverse)}} \in \mathbb{R}^n
$$

### Relationship to the Jacobian

| Operation | Input          | Output         | What it computes                    | AD mode |
| --------- | -------------- | -------------- | ----------------------------------- | ------- |
| JVP       | $\mathbf{v} \in \mathbb{R}^n$ | $\mathbb{R}^m$ | Directional derivative along $\mathbf{v}$ | Forward |
| VJP       | $\mathbf{u} \in \mathbb{R}^m$ | $\mathbb{R}^n$ | Gradient weighted by $\mathbf{u}$          | Reverse |

### Why you rarely need the full Jacobian

In most applications, you need $J \cdot \mathbf{v}$ or $\mathbf{u}^\top \cdot J$, **not** $J$ itself:

| Application                     | What's needed                 | AD mode   |
| ------------------------------- | ----------------------------- | --------- |
| Gradient descent (ML)           | $\nabla L = J^\top \cdot 1$  | Reverse   |
| Newton's method                 | Solve $J \cdot \Delta x = -\mathbf{f}$ | JVPs (Krylov methods) |
| Sensitivity analysis            | $J \cdot \mathbf{v}$         | Forward   |
| Hessian-vector products         | $\nabla^2 f \cdot \mathbf{v}$ = forward-over-reverse | Both |
| Gauss-Newton                    | $J^\top J \cdot \mathbf{v}$  | VJP of JVP |

> Computing $J$ explicitly is $O(nm)$ in space and $O(\min(n,m) \cdot \text{cost}(f))$ in time. Matrix-free methods using JVP/VJP are often preferred for large systems.

---

## 29. Jacobian Sparsity and Efficient Computation

For many practical problems, the Jacobian is **sparse** — most entries are zero. Exploiting sparsity dramatically reduces the cost.

### Graph coloring for sparse Jacobians

If column $j$ and column $k$ of $J$ have **no overlapping nonzero rows**, they can be computed in a **single** forward-mode pass by seeding $\mathbf{v} = \mathbf{e}_j + \mathbf{e}_k$. The nonzero entries won't interfere.

This is formalized as a **graph coloring** problem:
1. Build the **column intersection graph**: columns $j$ and $k$ share an edge if they have a nonzero in the same row.
2. **Color** the graph with the minimum number of colors (each color = one AD pass).
3. Seed each pass with the sum of basis vectors for all columns of that color.
4. Extract individual column entries from the combined result using the known sparsity pattern.

| Scenario                          | Full Jacobian passes | With coloring          |
| --------------------------------- | -------------------- | ---------------------- |
| Dense $J \in \mathbb{R}^{100 \times 100}$ | 100 (forward)   | 100 (no benefit)       |
| Tridiagonal $J \in \mathbb{R}^{1000 \times 1000}$ | 1000 (forward) | **3 passes!**          |
| Banded (bandwidth $b$) $J$       | $n$ (forward)        | $2b + 1$ passes        |

### Example: Tridiagonal Jacobian

$$
J = \begin{pmatrix}
* & * & 0 & 0 & 0 \\
* & * & * & 0 & 0 \\
0 & * & * & * & 0 \\
0 & 0 & * & * & * \\
0 & 0 & 0 & * & *
\end{pmatrix}
$$

Coloring: columns $\{1, 4\}$ get color 1, $\{2, 5\}$ get color 2, $\{3\}$ gets color 3.

- **Pass 1:** Seed $\mathbf{v} = (1, 0, 0, 1, 0)$ → recovers columns 1 and 4 simultaneously.
- **Pass 2:** Seed $\mathbf{v} = (0, 1, 0, 0, 1)$ → recovers columns 2 and 5 simultaneously.
- **Pass 3:** Seed $\mathbf{v} = (0, 0, 1, 0, 0)$ → recovers column 3.

**3 passes instead of 5** — and the savings grow as $n$ increases (always 3 passes for any tridiagonal system, regardless of size).

---

## 30. Higher-Order Derivatives: Hessians and Beyond

The Jacobian gives first-order derivatives. For second-order derivatives (Hessians), we can **nest** AD modes.

### The Hessian

For a scalar function $f : \mathbb{R}^n \to \mathbb{R}$, the Hessian $H \in \mathbb{R}^{n \times n}$ is:

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j}
$$

### Hessian-vector product via forward-over-reverse

To compute $H \cdot \mathbf{v}$ without forming $H$ explicitly:

1. **Inner (reverse):** Define $g(\mathbf{x}) = \nabla f(\mathbf{x})$ using reverse-mode AD.
2. **Outer (forward):** Compute $J_g \cdot \mathbf{v}$ using forward-mode AD on $g$.

Since $J_g = H$ (the Jacobian of the gradient is the Hessian):

$$
J_g \cdot \mathbf{v} = H \cdot \mathbf{v}
$$

**Cost:** One forward + one reverse + one forward pass ≈ constant multiple of evaluating $f$.

### Full Hessian via $n$ Hessian-vector products

Seed $\mathbf{v} = \mathbf{e}_j$ for $j = 1, \ldots, n$:

$$
H[:, j] = H \cdot \mathbf{e}_j
$$

**Cost:** $n$ forward-over-reverse passes.

### Mode nesting combinations

| Method                | Computes          | Cost                   |
| --------------------- | ----------------- | ---------------------- |
| Forward-over-forward  | $\frac{\partial^2 f}{\partial x_i \partial x_j}$ one entry | $O(n^2)$ for full $H$ |
| Forward-over-reverse  | $H \cdot \mathbf{v}$ (Hessian-vector product) | $O(n)$ for full $H$   |
| Reverse-over-forward  | $\mathbf{u}^\top \cdot H$ (vector-Hessian product) | $O(n)$ for full $H$   |
| Reverse-over-reverse  | Possible but complex; rarely used                | —                      |

> **Forward-over-reverse** is the standard method for Hessian-vector products, used extensively in second-order optimization (Newton's method, conjugate gradients on the Hessian).

---

## 31. C++ Implementation: Full Jacobian

### Forward-mode (column-by-column)

```cpp
#include <cmath>
#include <vector>
#include <iostream>

struct Dual {
    double real;
    double dual;
    Dual(double r = 0.0, double d = 0.0) : real(r), dual(d) {}
};

Dual operator+(Dual a, Dual b) { return { a.real + b.real, a.dual + b.dual }; }
Dual operator*(Dual a, Dual b) {
    return { a.real * b.real, a.real * b.dual + a.dual * b.real };
}
Dual sin(Dual x) { return { std::sin(x.real), x.dual * std::cos(x.real) }; }

// f : R^2 → R^2
// f(x1, x2) = ( x1*x2, sin(x1) + x2*x2 )
std::vector<Dual> f(Dual x1, Dual x2) {
    return { x1 * x2, sin(x1) + x2 * x2 };
}

// Compute full Jacobian at (a1, a2) via forward-mode AD
void jacobian_forward(double a1, double a2) {
    int n = 2;  // inputs
    int m = 2;  // outputs
    std::vector<std::vector<double>> J(m, std::vector<double>(n));

    for (int j = 0; j < n; ++j) {
        // Seed: e_j (1 in position j, 0 elsewhere)
        Dual x1(a1, j == 0 ? 1.0 : 0.0);
        Dual x2(a2, j == 1 ? 1.0 : 0.0);

        auto result = f(x1, x2);

        for (int i = 0; i < m; ++i) {
            J[i][j] = result[i].dual;  // column j
        }
    }

    std::cout << "Jacobian (forward-mode):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << J[i][j] << "\t";
        std::cout << "\n";
    }
}

int main() {
    jacobian_forward(2.0, 3.0);
    // Expected:
    //   3         2
    //   cos(2)    6
    return 0;
}
```

### Reverse-mode (row-by-row)

```cpp
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>

struct TapeEntry {
    double value;
    double adjoint = 0.0;
    std::function<void(double)> backward;
};

std::vector<TapeEntry> tape;

struct Var {
    int index;
    double value() const { return tape[index].value; }
    double& adjoint() const { return tape[index].adjoint; }

    explicit Var(double val) {
        index = tape.size();
        tape.push_back({ val, 0.0, [](double) {} });
    }
    Var(double val, std::function<void(double)> bwd) {
        index = tape.size();
        tape.push_back({ val, 0.0, bwd });
    }
};

Var operator+(Var a, Var b) {
    return Var(a.value() + b.value(), [a, b](double g) {
        tape[a.index].adjoint += g;
        tape[b.index].adjoint += g;
    });
}
Var operator*(Var a, Var b) {
    double av = a.value(), bv = b.value();
    return Var(av * bv, [a, b, av, bv](double g) {
        tape[a.index].adjoint += g * bv;
        tape[b.index].adjoint += g * av;
    });
}
Var sin(Var a) {
    double av = a.value();
    return Var(std::sin(av), [a, av](double g) {
        tape[a.index].adjoint += g * std::cos(av);
    });
}

void backward_from(int idx) {
    tape[idx].adjoint = 1.0;
    for (int i = idx; i >= 0; --i) {
        if (tape[i].adjoint != 0.0) {
            tape[i].backward(tape[i].adjoint);
        }
    }
}

void reset_adjoints() {
    for (auto& entry : tape)
        entry.adjoint = 0.0;
}

// f : R^2 → R^2
// f(x1, x2) = ( x1*x2, sin(x1) + x2*x2 )
void jacobian_reverse(double a1, double a2) {
    tape.clear();

    Var x1(a1);  // tape[0]
    Var x2(a2);  // tape[1]

    // Compute both outputs (shared tape)
    Var y1 = x1 * x2;
    Var y2 = sin(x1) + x2 * x2;

    int n = 2, m = 2;
    std::vector<std::vector<double>> J(m, std::vector<double>(n));

    // Row 0: backward from y1
    reset_adjoints();
    backward_from(y1.index);
    J[0][0] = x1.adjoint();
    J[0][1] = x2.adjoint();

    // Row 1: backward from y2
    reset_adjoints();
    backward_from(y2.index);
    J[1][0] = x1.adjoint();
    J[1][1] = x2.adjoint();

    std::cout << "Jacobian (reverse-mode):\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << J[i][j] << "\t";
        std::cout << "\n";
    }
}

int main() {
    jacobian_reverse(2.0, 3.0);
    // Expected:
    //   3         2
    //   cos(2)    6
    return 0;
}
```

**Both produce:**
```
Jacobian:
3       2
-0.416147       6
```

---

## 32. When to Use What: Decision Flowchart

```
Need derivatives of f : R^n → R^m ?
│
├── Need full Jacobian?
│   ├── J is dense:
│   │   ├── n < m  → Forward mode (n passes)
│   │   ├── n > m  → Reverse mode (m passes)
│   │   └── n ≈ m  → Either (or mixed)
│   │
│   └── J is sparse:
│       └── Use graph coloring + forward mode
│           (often ≪ n passes)
│
├── Need gradient (m = 1)?
│   └── Reverse mode (1 pass) ← backpropagation
│
├── Need J·v (JVP)?
│   └── Forward mode (1 pass)
│
├── Need uᵀ·J (VJP)?
│   └── Reverse mode (1 pass)
│
└── Need Hessian-vector product H·v?
    └── Forward-over-reverse (1 pass each)
```

---

## 33. Summary

| Concept              | Detail                                                                                   |
| -------------------- | ---------------------------------------------------------------------------------------- |
| **Definition**       | $a + b\varepsilon$ where $\varepsilon^2 = 0$, $\varepsilon \neq 0$                      |
| **Key property**     | $f(a + b\varepsilon) = f(a) + b \cdot f'(a)\,\varepsilon$                                |
| **Real part**        | The function value $f(a)$                                                                |
| **Dual part**        | The exact derivative $b \cdot f'(a)$                                                     |
| **Chain rule**       | Handled automatically by composition of dual arithmetic                                  |
| **Forward-mode AD**  | Seed input with $\varepsilon$, evaluate, read off derivative from dual part              |
| **Reverse-mode AD**  | Record tape in forward pass, propagate adjoints backward to get all gradients            |
| **JVP**              | $J \cdot \mathbf{v}$ via one forward pass — gives one column of $J$ when $\mathbf{v} = \mathbf{e}_j$ |
| **VJP**              | $\mathbf{u}^\top \cdot J$ via one reverse pass — gives one row of $J$ when $\mathbf{u} = \mathbf{e}_i$ |
| **Full Jacobian**    | Forward: $n$ passes (column by column). Reverse: $m$ passes (row by row)                |
| **Sparse Jacobian**  | Graph coloring reduces passes from $n$ to number of colors (often $\ll n$)              |
| **Hessian-vector**   | Forward-over-reverse: nest forward inside reverse for $H \cdot \mathbf{v}$              |
| **Backpropagation**  | Reverse-mode AD applied to neural network loss functions                                 |
| **Checkpointing**    | Trade recomputation for memory: $O(\sqrt{T})$ memory at $\leq 2\times$ compute cost     |