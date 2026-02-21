// reverse_mode.cpp — Reverse-mode AD examples from ad.md
// Demonstrates tape-based backpropagation for gradient computation.
//
// Covers:
//   - Basic gradient: f(x₁,x₂) = x₁·x₂ + sin(x₁)  (§13, §22)
//   - Multi-layer: f(x) = e^(sin(x²))                (§15)
//   - Fan-out: f(x) = x·x                            (§18)
//   - Multiple outputs

#include "var.hpp"
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

// ── Examples ────────────────────────────────────────────────────────

// §13/§22: f(x₁,x₂) = x₁·x₂ + sin(x₁) at (2,3)
// Both partial derivatives in a single backward pass
void example_basic_gradient() {
    header("f(x₁,x₂) = x₁·x₂ + sin(x₁) at (2,3)  [§13, §22]");
    clear_tape();

    Var x1(2.0);
    Var x2(3.0);
    Var y = x1 * x2 + sin(x1);

    backward(y);

    check("f(2,3) ", y.value(), 6.0 + std::sin(2.0));
    check("∂f/∂x₁", x1.adjoint(), 3.0 + std::cos(2.0));
    check("∂f/∂x₂", x2.adjoint(), 2.0);
}

// §15: f(x) = e^(sin(x²)) at x = 1
void example_multi_layer() {
    header("f(x) = e^(sin(x²)) at x = 1  [§15]");
    clear_tape();

    Var x(1.0);
    Var y = exp(sin(x * x));

    backward(y);

    double expected_val = std::exp(std::sin(1.0));
    double expected_der = std::exp(std::sin(1.0)) * std::cos(1.0) * 2.0;
    check("f(1)  ", y.value(), expected_val);
    check("df/dx ", x.adjoint(), expected_der);
}

// §18: Fan-out — f(x) = x·x (adjoint must accumulate with +=)
void example_fan_out() {
    header("Fan-out: f(x) = x·x at x = 5  [§18]");
    clear_tape();

    Var x(5.0);
    Var y = x * x;

    backward(y);

    check("f(5)  ", y.value(), 25.0);
    check("df/dx ", x.adjoint(), 10.0);  // 2x = 10 (accumulated from both uses)
}

// Division & subtraction
void example_quotient() {
    header("f(x) = (x² - 1) / (x + 1) at x = 3  [should equal x - 1 = 2]");
    clear_tape();

    Var x(3.0);
    Var num = x * x - Var(1.0);  // x² - 1
    Var den = x + Var(1.0);      // x + 1
    Var y = num / den;           // simplifies to x - 1

    backward(y);

    check("f(3)  ", y.value(), 2.0);
    check("df/dx ", x.adjoint(), 1.0);  // d/dx(x-1) = 1
}

// Nested: f(x) = log(1 + exp(x))  — softplus function
void example_softplus() {
    header("Softplus: f(x) = log(1 + exp(x)) at x = 2");
    clear_tape();

    Var x(2.0);
    Var y = log(Var(1.0) + exp(x));

    backward(y);

    double expected_val = std::log(1.0 + std::exp(2.0));
    double expected_der = std::exp(2.0) / (1.0 + std::exp(2.0));  // sigmoid
    check("f(2)  ", y.value(), expected_val);
    check("df/dx ", x.adjoint(), expected_der);
    std::cout << "  (df/dx is the sigmoid function σ(x))\n";
}

// Complex expression: f(x,y) = x·sin(y) + y·cos(x)
void example_complex_expression() {
    header("f(x,y) = x·sin(y) + y·cos(x) at (π/4, π/3)");
    clear_tape();

    double xv = M_PI / 4.0, yv = M_PI / 3.0;
    Var x(xv);
    Var y(yv);
    Var f = x * sin(y) + y * cos(x);

    backward(f);

    double expected_val = xv * std::sin(yv) + yv * std::cos(xv);
    double expected_dfdx = std::sin(yv) - yv * std::sin(xv);
    double expected_dfdy = xv * std::cos(yv) + std::cos(xv);

    check("f      ", f.value(), expected_val);
    check("∂f/∂x  ", x.adjoint(), expected_dfdx);
    check("∂f/∂y  ", y.adjoint(), expected_dfdy);
}

// Sqrt and chain rule: f(x) = sqrt(x² + 1) at x = 3
void example_sqrt_chain() {
    header("f(x) = sqrt(x² + 1) at x = 3");
    clear_tape();

    Var x(3.0);
    Var y = sqrt(x * x + Var(1.0));

    backward(y);

    double expected_val = std::sqrt(10.0);
    double expected_der = 3.0 / std::sqrt(10.0);  // x / sqrt(x²+1)
    check("f(3)  ", y.value(), expected_val);
    check("df/dx ", x.adjoint(), expected_der);
}

// Three variables: f(a,b,c) = a·b·c
void example_three_vars() {
    header("f(a,b,c) = a·b·c at (2,3,4)  [all gradients in one pass]");
    clear_tape();

    Var a(2.0), b(3.0), c(4.0);
    Var y = a * b * c;

    backward(y);

    check("f(2,3,4)", y.value(), 24.0);
    check("∂f/∂a   ", a.adjoint(), 12.0);  // b·c = 12
    check("∂f/∂b   ", b.adjoint(), 8.0);   // a·c = 8
    check("∂f/∂c   ", c.adjoint(), 6.0);   // a·b = 6
}

// ── Main ────────────────────────────────────────────────────────────

int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n"
              << "║   Reverse-Mode AD — Tape-Based Examples      ║\n"
              << "╚══════════════════════════════════════════════╝\n";

    example_basic_gradient();
    example_multi_layer();
    example_fan_out();
    example_quotient();
    example_softplus();
    example_complex_expression();
    example_sqrt_chain();
    example_three_vars();

    std::cout << "\n══════════════════════════════════════════════\n"
              << "All reverse-mode examples completed.\n";
    return 0;
}
