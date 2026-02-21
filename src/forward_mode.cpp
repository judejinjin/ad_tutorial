// forward_mode.cpp — Forward-mode AD examples from ad.md
// Demonstrates dual numbers computing exact derivatives.
//
// Covers:
//   - Basic: x², x³+2x, sin(x)
//   - Composition (chain rule): sin(x²)
//   - Multi-layer: e^(sin(x²))
//   - All math functions: exp, log, sqrt, tan, pow
//   - Quotient rule
//   - Multi-variable partial derivatives

#include "dual.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

// ── Helpers ─────────────────────────────────────────────────────────

static void header(const char* title) {
    std::cout << "\n━━━ " << title << " ━━━\n";
}

static void check(const char* label, double got, double expected) {
    bool ok = std::abs(got - expected) < 1e-10;
    std::cout << std::fixed << std::setprecision(6)
              << "  " << label << " = " << got
              << "  (expected " << expected << ") "
              << (ok ? "✓" : "✗ MISMATCH") << "\n";
}

// ── Examples from ad.md ─────────────────────────────────────────────

// Example 1 (§5): f(x) = x² at x = 3
void example_x_squared() {
    header("f(x) = x² at x = 3");
    Dual x(3.0, 1.0);
    Dual y = x * x;
    check("f(3) ", y.real, 9.0);
    check("f'(3)", y.dual, 6.0);   // f'(x) = 2x → 6
}

// Example 2 (§5): f(x) = x³ + 2x at x = 2
void example_cubic() {
    header("f(x) = x³ + 2x at x = 2");
    Dual x(2.0, 1.0);
    Dual y = x * x * x + 2.0 * x;
    check("f(2) ", y.real, 12.0);
    check("f'(2)", y.dual, 14.0);  // f'(x) = 3x² + 2 → 14
}

// Example 3 (§5): f(x) = sin(x) at x = 0
void example_sin() {
    header("f(x) = sin(x) at x = 0");
    Dual x(0.0, 1.0);
    Dual y = sin(x);
    check("f(0) ", y.real, 0.0);
    check("f'(0)", y.dual, 1.0);   // f'(x) = cos(x) → cos(0) = 1
}

// Example 4 (§5): f(x) = sin(x²) at x = 3  — chain rule
void example_sin_x_squared() {
    header("f(x) = sin(x²) at x = 3  [chain rule]");
    Dual x(3.0, 1.0);
    Dual y = sin(x * x);
    check("f(3) ", y.real, std::sin(9.0));
    check("f'(3)", y.dual, 6.0 * std::cos(9.0));  // 2x·cos(x²)
}

// Example from §8: f(x) = sin(x²) via the C++ sketch
void example_cpp_sketch() {
    header("C++ sketch (§8): f(x) = sin(x²) at x = 3");
    Dual x(3.0, 1.0);
    Dual result = sin(x * x);
    check("value   ", result.real, std::sin(9.0));
    check("deriv   ", result.dual, 6.0 * std::cos(9.0));
}

// Example from §15: f(x) = e^(sin(x²)) at x = 1
void example_exp_sin_x_squared() {
    header("f(x) = e^(sin(x²)) at x = 1  [multi-layer composition]");
    Dual x(1.0, 1.0);
    Dual y = exp(sin(x * x));
    double expected_val = std::exp(std::sin(1.0));
    double expected_der = std::exp(std::sin(1.0)) * std::cos(1.0) * 2.0;
    check("f(1) ", y.real, expected_val);
    check("f'(1)", y.dual, expected_der);
}

// Additional: all math functions
void example_all_functions() {
    header("All math functions at x = 2");
    Dual x(2.0, 1.0);

    // exp
    Dual y_exp = exp(x);
    check("exp(2)  val", y_exp.real, std::exp(2.0));
    check("exp(2)  der", y_exp.dual, std::exp(2.0));

    // log
    Dual y_log = log(x);
    check("log(2)  val", y_log.real, std::log(2.0));
    check("log(2)  der", y_log.dual, 1.0 / 2.0);

    // sqrt
    Dual y_sqrt = sqrt(x);
    check("sqrt(2) val", y_sqrt.real, std::sqrt(2.0));
    check("sqrt(2) der", y_sqrt.dual, 1.0 / (2.0 * std::sqrt(2.0)));

    // tan
    Dual y_tan = tan(x);
    check("tan(2)  val", y_tan.real, std::tan(2.0));
    check("tan(2)  der", y_tan.dual, 1.0 / (std::cos(2.0) * std::cos(2.0)));

    // cos
    Dual y_cos = cos(x);
    check("cos(2)  val", y_cos.real, std::cos(2.0));
    check("cos(2)  der", y_cos.dual, -std::sin(2.0));

    // pow (x^3)
    Dual y_pow = pow(x, 3.0);
    check("2^3     val", y_pow.real, 8.0);
    check("2^3     der", y_pow.dual, 12.0);  // 3x² = 12
}

// Quotient rule: f(x) = sin(x) / x at x = π
void example_quotient() {
    header("Quotient rule: f(x) = sin(x)/x at x = π");
    Dual x(M_PI, 1.0);
    Dual y = sin(x) / x;
    double expected_val = std::sin(M_PI) / M_PI;  // ≈ 0
    // f'(x) = (cos(x)·x - sin(x)) / x²
    double expected_der = (std::cos(M_PI) * M_PI - std::sin(M_PI)) / (M_PI * M_PI);
    check("f(π) ", y.real, expected_val);
    check("f'(π)", y.dual, expected_der);  // ≈ -1/π
}

// Multi-variable partial derivatives: f(x,y) = x·y + sin(x) at (2,3)
void example_partial_derivatives() {
    header("Partial derivatives: f(x₁,x₂) = x₁·x₂ + sin(x₁) at (2,3)");

    // ∂f/∂x₁: seed x₁ with ε, x₂ with 0
    {
        Dual x1(2.0, 1.0), x2(3.0, 0.0);
        Dual y = x1 * x2 + sin(x1);
        check("f(2,3)    ", y.real, 2.0 * 3.0 + std::sin(2.0));
        check("∂f/∂x₁    ", y.dual, 3.0 + std::cos(2.0));
    }

    // ∂f/∂x₂: seed x₂ with ε, x₁ with 0
    {
        Dual x1(2.0, 0.0), x2(3.0, 1.0);
        Dual y = x1 * x2 + sin(x1);
        check("∂f/∂x₂    ", y.dual, 2.0);
    }
}

// Fan-out: f(x) = x·x (same variable used twice)
void example_fan_out() {
    header("Fan-out: f(x) = x·x at x = 5");
    Dual x(5.0, 1.0);
    Dual y = x * x;  // product rule handles fan-out automatically
    check("f(5) ", y.real, 25.0);
    check("f'(5)", y.dual, 10.0);  // 2x = 10
}

// ════════════════════════════════════════════════════════════════════
// Multi-variable examples that reverse mode computes in ONE pass
// but forward mode requires MULTIPLE passes (one per input variable).
// Compare with reverse_mode.cpp where each of these is a single
// backward() call.
// ════════════════════════════════════════════════════════════════════

// From reverse_mode.cpp example_complex_expression:
// f(x,y) = x·sin(y) + y·cos(x)  at (π/4, π/3)
// Reverse mode: 1 backward pass → both ∂f/∂x and ∂f/∂y
// Forward mode: 2 passes (seed x, then seed y)
void example_multi_var_2inputs() {
    header("Forward needs 2 passes: f(x,y) = x·sin(y) + y·cos(x) at (π/4, π/3)");

    double xv = M_PI / 4.0, yv = M_PI / 3.0;
    double expected_val = xv * std::sin(yv) + yv * std::cos(xv);
    double expected_dfdx = std::sin(yv) - yv * std::sin(xv);
    double expected_dfdy = xv * std::cos(yv) + std::cos(xv);

    std::cout << "  (Reverse mode gets both partials in 1 pass;\n"
              << "   forward mode needs 2 separate passes.)\n";

    // Pass 1: seed x with ε → get ∂f/∂x
    {
        Dual x(xv, 1.0), y(yv, 0.0);
        Dual f = x * sin(y) + y * cos(x);
        check("  Pass 1 — f       ", f.real, expected_val);
        check("  Pass 1 — ∂f/∂x   ", f.dual, expected_dfdx);
    }

    // Pass 2: seed y with ε → get ∂f/∂y
    {
        Dual x(xv, 0.0), y(yv, 1.0);
        Dual f = x * sin(y) + y * cos(x);
        check("  Pass 2 — ∂f/∂y   ", f.dual, expected_dfdy);
    }
}

// From reverse_mode.cpp example_three_vars:
// f(a,b,c) = a·b·c  at (2,3,4)
// Reverse mode: 1 backward pass → all 3 gradients
// Forward mode: 3 passes (seed a, then b, then c)
void example_multi_var_3inputs() {
    header("Forward needs 3 passes: f(a,b,c) = a·b·c at (2,3,4)");

    double av = 2.0, bv = 3.0, cv = 4.0;
    double expected_val = av * bv * cv;              // 24
    double expected_dfda = bv * cv;                  // 12
    double expected_dfdb = av * cv;                  // 8
    double expected_dfdc = av * bv;                  // 6

    std::cout << "  (Reverse mode gets all 3 gradients in 1 pass;\n"
              << "   forward mode needs 3 separate passes.)\n";

    // Pass 1: seed a → ∂f/∂a
    {
        Dual a(av, 1.0), b(bv, 0.0), c(cv, 0.0);
        Dual f = a * b * c;
        check("  Pass 1 — f       ", f.real, expected_val);
        check("  Pass 1 — ∂f/∂a   ", f.dual, expected_dfda);
    }

    // Pass 2: seed b → ∂f/∂b
    {
        Dual a(av, 0.0), b(bv, 1.0), c(cv, 0.0);
        Dual f = a * b * c;
        check("  Pass 2 — ∂f/∂b   ", f.dual, expected_dfdb);
    }

    // Pass 3: seed c → ∂f/∂c
    {
        Dual a(av, 0.0), b(bv, 0.0), c(cv, 1.0);
        Dual f = a * b * c;
        check("  Pass 3 — ∂f/∂c   ", f.dual, expected_dfdc);
    }
}

// From reverse_mode.cpp gradient_example (in jacobian_reverse.cpp):
// L(w₁,w₂,w₃) = (w₁·w₂ + w₃)²
// Reverse mode: 1 backward pass → all 3 gradients
// Forward mode: 3 passes
void example_loss_gradient() {
    header("Forward needs 3 passes: L(w₁,w₂,w₃) = (w₁·w₂ + w₃)² at (1,2,3)");

    double w1v = 1.0, w2v = 2.0, w3v = 3.0;
    double linear = w1v * w2v + w3v;              // 5
    double expected_val = linear * linear;        // 25
    // ∂L/∂w₁ = 2(w₁w₂+w₃)·w₂ = 2·5·2 = 20
    // ∂L/∂w₂ = 2(w₁w₂+w₃)·w₁ = 2·5·1 = 10
    // ∂L/∂w₃ = 2(w₁w₂+w₃)·1  = 2·5·1 = 10
    double expected_dLdw1 = 2.0 * linear * w2v;  // 20
    double expected_dLdw2 = 2.0 * linear * w1v;  // 10
    double expected_dLdw3 = 2.0 * linear * 1.0;  // 10

    std::cout << "  (Reverse mode gets all 3 gradients in 1 pass;\n"
              << "   forward mode needs 3 separate passes.)\n"
              << "  This is the typical ML scenario — many weights, scalar loss.\n";

    // Pass 1: seed w₁ → ∂L/∂w₁
    {
        Dual w1(w1v, 1.0), w2(w2v, 0.0), w3(w3v, 0.0);
        Dual lin = w1 * w2 + w3;
        Dual L = lin * lin;
        check("  Pass 1 — L       ", L.real, expected_val);
        check("  Pass 1 — ∂L/∂w₁  ", L.dual, expected_dLdw1);
    }

    // Pass 2: seed w₂ → ∂L/∂w₂
    {
        Dual w1(w1v, 0.0), w2(w2v, 1.0), w3(w3v, 0.0);
        Dual lin = w1 * w2 + w3;
        Dual L = lin * lin;
        check("  Pass 2 — ∂L/∂w₂  ", L.dual, expected_dLdw2);
    }

    // Pass 3: seed w₃ → ∂L/∂w₃
    {
        Dual w1(w1v, 0.0), w2(w2v, 0.0), w3(w3v, 1.0);
        Dual lin = w1 * w2 + w3;
        Dual L = lin * lin;
        check("  Pass 3 — ∂L/∂w₃  ", L.dual, expected_dLdw3);
    }
}

// ── Main ────────────────────────────────────────────────────────────

int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n"
              << "║   Forward-Mode AD — Dual Number Examples     ║\n"
              << "╚══════════════════════════════════════════════╝\n";

    // Examples from ad.md §5
    example_x_squared();
    example_cubic();
    example_sin();
    example_sin_x_squared();

    // C++ sketch from §8
    example_cpp_sketch();

    // Multi-layer from §15
    example_exp_sin_x_squared();

    // All math functions from §6 table
    example_all_functions();

    // Quotient rule from §3
    example_quotient();

    // Multi-variable from §7
    example_partial_derivatives();

    // Fan-out from §18
    example_fan_out();

    // ── Multi-variable: forward needs N passes, reverse needs 1 ──
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n"
              << "║   Forward vs Reverse: multi-variable cost comparison ║\n"
              << "╚══════════════════════════════════════════════════════╝\n";

    example_multi_var_2inputs();
    example_multi_var_3inputs();
    example_loss_gradient();

    std::cout << "\n══════════════════════════════════════════════\n"
              << "All forward-mode examples completed.\n";
    return 0;
}
