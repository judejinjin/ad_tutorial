// jacobian_forward.cpp — Full Jacobian via forward-mode AD (column by column)
// From ad.md §27 and §31.
//
// f : R² → R²
// f(x₁,x₂) = ( x₁·x₂,  sin(x₁) + x₂² )
//
// Jacobian:  J = [ ∂f₁/∂x₁  ∂f₁/∂x₂ ] = [ x₂      x₁    ]
//                [ ∂f₂/∂x₁  ∂f₂/∂x₂ ]   [ cos(x₁)  2·x₂  ]
//
// At (2,3):  J = [ 3         2      ]
//                [ cos(2)    6      ]

#include "dual.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// ── The function ────────────────────────────────────────────────────

std::vector<Dual> f(Dual x1, Dual x2) {
    return { x1 * x2, sin(x1) + x2 * x2 };
}

// ── Forward-mode Jacobian (n passes for n inputs) ───────────────────

void jacobian_forward(double a1, double a2) {
    const int n = 2;  // inputs
    const int m = 2;  // outputs
    std::vector<std::vector<double>> J(m, std::vector<double>(n));

    for (int j = 0; j < n; ++j) {
        // Seed with standard basis vector e_j
        Dual x1(a1, j == 0 ? 1.0 : 0.0);
        Dual x2(a2, j == 1 ? 1.0 : 0.0);

        auto result = f(x1, x2);

        for (int i = 0; i < m; ++i) {
            J[i][j] = result[i].dual;  // fill column j
        }
    }

    // Print
    std::cout << "Jacobian (forward-mode, column by column):\n";
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

// ── JVP (Jacobian-vector product) ───────────────────────────────────

void jvp_example(double a1, double a2, double v1, double v2) {
    std::cout << "\n━━━ JVP: J·v with v = (" << v1 << ", " << v2 << ") ━━━\n";
    Dual x1(a1, v1);
    Dual x2(a2, v2);
    auto result = f(x1, x2);

    std::cout << "  J·v = [ " << result[0].dual << ", "
              << result[1].dual << " ]\n";

    // Manual: J·v = [x2·v1 + x1·v2,  cos(x1)·v1 + 2x2·v2]
    double expected0 = a2 * v1 + a1 * v2;
    double expected1 = std::cos(a1) * v1 + 2.0 * a2 * v2;
    std::cout << "  Expected: [ " << expected0 << ", " << expected1 << " ]\n";
}

// ── Main ────────────────────────────────────────────────────────────

int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n"
              << "║   Jacobian via Forward-Mode AD (Column by Column)║\n"
              << "╚══════════════════════════════════════════════════╝\n\n";

    std::cout << "f(x₁,x₂) = ( x₁·x₂,  sin(x₁) + x₂² )\n";
    std::cout << "Evaluated at (x₁,x₂) = (2, 3)\n\n";

    jacobian_forward(2.0, 3.0);

    // JVP examples
    jvp_example(2.0, 3.0, 1.0, 0.0);  // column 1
    jvp_example(2.0, 3.0, 0.0, 1.0);  // column 2
    jvp_example(2.0, 3.0, 1.0, 1.0);  // arbitrary direction

    std::cout << "\n══════════════════════════════════════════════\n"
              << "Jacobian (forward) examples completed.\n";
    return 0;
}
