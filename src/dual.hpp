#pragma once
// dual.hpp — Forward-mode AD via dual numbers
// ε² = 0, so f(a + bε) = f(a) + b·f'(a)·ε

#include <cmath>
#include <iostream>

struct Dual {
    double real;  // f(x)   — the value (primal)
    double dual;  // f'(x)  — the derivative (tangent)

    Dual(double r = 0.0, double d = 0.0) : real(r), dual(d) {}
};

// ── Arithmetic operators ────────────────────────────────────────────

inline Dual operator+(Dual a, Dual b) {
    return { a.real + b.real, a.dual + b.dual };
}

inline Dual operator-(Dual a, Dual b) {
    return { a.real - b.real, a.dual - b.dual };
}

inline Dual operator*(Dual a, Dual b) {
    // product rule: (a + a'ε)(b + b'ε) = ab + (ab' + a'b)ε
    return { a.real * b.real, a.real * b.dual + a.dual * b.real };
}

inline Dual operator/(Dual a, Dual b) {
    // quotient rule: (a/b)' = (a'b - ab') / b²
    return { a.real / b.real,
             (a.dual * b.real - a.real * b.dual) / (b.real * b.real) };
}

// Unary minus
inline Dual operator-(Dual a) {
    return { -a.real, -a.dual };
}

// Scalar-Dual mixed operations
inline Dual operator+(double s, Dual a) { return Dual(s) + a; }
inline Dual operator+(Dual a, double s) { return a + Dual(s); }
inline Dual operator-(double s, Dual a) { return Dual(s) - a; }
inline Dual operator-(Dual a, double s) { return a - Dual(s); }
inline Dual operator*(double s, Dual a) { return Dual(s) * a; }
inline Dual operator*(Dual a, double s) { return a * Dual(s); }
inline Dual operator/(double s, Dual a) { return Dual(s) / a; }
inline Dual operator/(Dual a, double s) { return a / Dual(s); }

// ── Math functions ──────────────────────────────────────────────────
// Each uses: f(a + bε) = f(a) + b·f'(a)·ε

inline Dual sin(Dual x) {
    return { std::sin(x.real), x.dual * std::cos(x.real) };
}

inline Dual cos(Dual x) {
    return { std::cos(x.real), -x.dual * std::sin(x.real) };
}

inline Dual tan(Dual x) {
    double c = std::cos(x.real);
    return { std::tan(x.real), x.dual / (c * c) };
}

inline Dual exp(Dual x) {
    double ex = std::exp(x.real);
    return { ex, x.dual * ex };
}

inline Dual log(Dual x) {
    return { std::log(x.real), x.dual / x.real };
}

inline Dual sqrt(Dual x) {
    double s = std::sqrt(x.real);
    return { s, x.dual / (2.0 * s) };
}

inline Dual pow(Dual x, double n) {
    double p = std::pow(x.real, n);
    return { p, x.dual * n * std::pow(x.real, n - 1.0) };
}

inline Dual abs(Dual x) {
    return { std::abs(x.real), x.dual * (x.real >= 0 ? 1.0 : -1.0) };
}

// ── Output ──────────────────────────────────────────────────────────

inline std::ostream& operator<<(std::ostream& os, Dual d) {
    os << d.real;
    if (d.dual >= 0) os << " + " << d.dual << "ε";
    else             os << " - " << -d.dual << "ε";
    return os;
}
