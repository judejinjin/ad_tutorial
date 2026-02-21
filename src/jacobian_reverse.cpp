// jacobian_reverse.cpp — Full Jacobian via reverse-mode AD (row by row)
// From ad.md §27 and §31.
//
// f : R² → R²
// f(x₁,x₂) = ( x₁·x₂,  sin(x₁) + x₂² )
//
// Jacobian:  J = [ x₂      x₁    ]
//                [ cos(x₁)  2·x₂  ]
//
// At (2,3):  J = [ 3         2      ]
//                [ cos(2)    6      ]

#include "var.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// ── Reverse-mode Jacobian (m passes for m outputs) ──────────────────

void jacobian_reverse(double a1, double a2) {
    clear_tape();

    // Forward pass: record the computation on one shared tape
    Var x1(a1);   // tape[0]
    Var x2(a2);   // tape[1]

    Var y1 = x1 * x2;                // output 1: x₁·x₂
    Var y2 = sin(x1) + x2 * x2;     // output 2: sin(x₁) + x₂²

    const int n = 2, m = 2;
    std::vector<std::vector<double>> J(m, std::vector<double>(n));

    // Row 0: backward from y1 (seed ȳ₁=1, ȳ₂=0)
    reset_adjoints();
    backward_from(y1.index);
    J[0][0] = x1.adjoint();
    J[0][1] = x2.adjoint();

    // Row 1: backward from y2 (seed ȳ₁=0, ȳ₂=1)
    reset_adjoints();
    backward_from(y2.index);
    J[1][0] = x1.adjoint();
    J[1][1] = x2.adjoint();

    // Print
    std::cout << "Jacobian (reverse-mode, row by row):\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < m; ++i) {
        std::cout << "  [ ";
        for (int j = 0; j < n; ++j)
            std::cout << std::setw(12) << J[i][j] << " ";
        std::cout << "]\n";
    }

    // Verify
    std::cout << "\nExpected:\n";
    std::cout << "  [ " << std::setw(12) << a2 << " "
              << std::setw(12) << a1 << " ]\n";
    std::cout << "  [ " << std::setw(12) << std::cos(a1) << " "
              << std::setw(12) << 2.0 * a2 << " ]\n";
}

// ── VJP (vector-Jacobian product) ───────────────────────────────────

void vjp_example(double a1, double a2, double u1, double u2) {
    std::cout << "\n━━━ VJP: uᵀ·J with u = (" << u1 << ", " << u2 << ") ━━━\n";
    clear_tape();

    Var x1(a1);
    Var x2(a2);
    Var y1 = x1 * x2;
    Var y2 = sin(x1) + x2 * x2;

    // Seed both output adjoints with u
    reset_adjoints();
    auto& tape = get_tape();
    tape[y1.index].adjoint = u1;
    tape[y2.index].adjoint = u2;

    // Backward pass
    for (int i = static_cast<int>(tape.size()) - 1; i >= 0; --i) {
        if (tape[i].adjoint != 0.0) {
            tape[i].backward(tape[i].adjoint);
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  uᵀ·J = [ " << x1.adjoint() << ", " << x2.adjoint() << " ]\n";

    // Manual: uᵀ·J = [ u1·x2 + u2·cos(x1),  u1·x1 + u2·2x2 ]
    double expected0 = u1 * a2 + u2 * std::cos(a1);
    double expected1 = u1 * a1 + u2 * 2.0 * a2;
    std::cout << "  Expected: [ " << expected0 << ", " << expected1 << " ]\n";
}

// ── Gradient of a scalar function (the common ML case) ──────────────

void gradient_example() {
    std::cout << "\n━━━ Gradient: L(w₁,w₂,w₃) = (w₁·w₂ + w₃)² ━━━\n";
    clear_tape();

    Var w1(1.0), w2(2.0), w3(3.0);
    Var linear = w1 * w2 + w3;
    Var loss = linear * linear;  // L = (w₁w₂ + w₃)²

    backward(loss);

    // L = (1·2 + 3)² = 25
    // ∂L/∂w₁ = 2(w₁w₂+w₃)·w₂ = 2·5·2 = 20
    // ∂L/∂w₂ = 2(w₁w₂+w₃)·w₁ = 2·5·1 = 10
    // ∂L/∂w₃ = 2(w₁w₂+w₃)·1  = 2·5·1 = 10

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  L        = " << loss.value()  << "  (expected 25)\n";
    std::cout << "  ∂L/∂w₁   = " << w1.adjoint()  << "  (expected 20)\n";
    std::cout << "  ∂L/∂w₂   = " << w2.adjoint()  << "  (expected 10)\n";
    std::cout << "  ∂L/∂w₃   = " << w3.adjoint()  << "  (expected 10)\n";
    std::cout << "  → All 3 gradients from a SINGLE backward pass!\n";
}

// ── Main ────────────────────────────────────────────────────────────

int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n"
              << "║   Jacobian via Reverse-Mode AD (Row by Row)      ║\n"
              << "╚══════════════════════════════════════════════════╝\n\n";

    std::cout << "f(x₁,x₂) = ( x₁·x₂,  sin(x₁) + x₂² )\n";
    std::cout << "Evaluated at (x₁,x₂) = (2, 3)\n\n";

    jacobian_reverse(2.0, 3.0);

    // VJP examples
    vjp_example(2.0, 3.0, 1.0, 0.0);  // row 1
    vjp_example(2.0, 3.0, 0.0, 1.0);  // row 2
    vjp_example(2.0, 3.0, 1.0, 1.0);  // weighted sum of rows

    // Scalar gradient (the ML use case)
    gradient_example();

    std::cout << "\n══════════════════════════════════════════════\n"
              << "Jacobian (reverse) examples completed.\n";
    return 0;
}
