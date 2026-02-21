#pragma once
// var.hpp — Reverse-mode AD via a Wengert tape
// Records operations during a forward pass, then propagates
// adjoints (sensitivities) backward to compute gradients.

#include <cmath>
#include <vector>
#include <functional>
#include <iostream>

// ── Tape ────────────────────────────────────────────────────────────

struct TapeEntry {
    double value;
    double adjoint = 0.0;
    std::function<void(double)> backward;
};

// Global tape — all Var operations append here
inline std::vector<TapeEntry>& get_tape() {
    static std::vector<TapeEntry> tape;
    return tape;
}

// ── Var type ────────────────────────────────────────────────────────

struct Var {
    int index;  // index into the global tape

    double value()   const { return get_tape()[index].value; }
    double& adjoint() const { return get_tape()[index].adjoint; }

    // Create a leaf variable (input)
    explicit Var(double val) {
        auto& tape = get_tape();
        index = static_cast<int>(tape.size());
        tape.push_back({ val, 0.0, [](double) {} });
    }

    // Create a computed variable (internal node)
    Var(double val, std::function<void(double)> bwd) {
        auto& tape = get_tape();
        index = static_cast<int>(tape.size());
        tape.push_back({ val, 0.0, std::move(bwd) });
    }
};

// ── Arithmetic operators ────────────────────────────────────────────

inline Var operator+(Var a, Var b) {
    double val = a.value() + b.value();
    return Var(val, [a, b](double grad) {
        auto& tape = get_tape();
        tape[a.index].adjoint += grad;       // ∂(a+b)/∂a = 1
        tape[b.index].adjoint += grad;       // ∂(a+b)/∂b = 1
    });
}

inline Var operator-(Var a, Var b) {
    double val = a.value() - b.value();
    return Var(val, [a, b](double grad) {
        auto& tape = get_tape();
        tape[a.index].adjoint += grad;       // ∂(a-b)/∂a = 1
        tape[b.index].adjoint += -grad;      // ∂(a-b)/∂b = -1
    });
}

inline Var operator*(Var a, Var b) {
    double val = a.value() * b.value();
    double a_val = a.value(), b_val = b.value();
    return Var(val, [a, b, a_val, b_val](double grad) {
        auto& tape = get_tape();
        tape[a.index].adjoint += grad * b_val;  // ∂(a*b)/∂a = b
        tape[b.index].adjoint += grad * a_val;  // ∂(a*b)/∂b = a
    });
}

inline Var operator/(Var a, Var b) {
    double val = a.value() / b.value();
    double a_val = a.value(), b_val = b.value();
    return Var(val, [a, b, a_val, b_val](double grad) {
        auto& tape = get_tape();
        tape[a.index].adjoint += grad / b_val;
        tape[b.index].adjoint += -grad * a_val / (b_val * b_val);
    });
}

// ── Math functions ──────────────────────────────────────────────────

inline Var sin(Var a) {
    double val = std::sin(a.value());
    double a_val = a.value();
    return Var(val, [a, a_val](double grad) {
        get_tape()[a.index].adjoint += grad * std::cos(a_val);
    });
}

inline Var cos(Var a) {
    double val = std::cos(a.value());
    double a_val = a.value();
    return Var(val, [a, a_val](double grad) {
        get_tape()[a.index].adjoint += grad * (-std::sin(a_val));
    });
}

inline Var exp(Var a) {
    double val = std::exp(a.value());
    return Var(val, [a, val](double grad) {
        get_tape()[a.index].adjoint += grad * val;
    });
}

inline Var log(Var a) {
    double val = std::log(a.value());
    double a_val = a.value();
    return Var(val, [a, a_val](double grad) {
        get_tape()[a.index].adjoint += grad / a_val;
    });
}

inline Var sqrt(Var a) {
    double val = std::sqrt(a.value());
    return Var(val, [a, val](double grad) {
        get_tape()[a.index].adjoint += grad / (2.0 * val);
    });
}

// ── Tape operations ─────────────────────────────────────────────────

// Run backward pass from a given output node
inline void backward(Var output) {
    auto& tape = get_tape();
    tape[output.index].adjoint = 1.0;
    for (int i = output.index; i >= 0; --i) {
        if (tape[i].adjoint != 0.0) {
            tape[i].backward(tape[i].adjoint);
        }
    }
}

// Run backward from a specific tape index (for Jacobian row computation)
inline void backward_from(int idx) {
    auto& tape = get_tape();
    tape[idx].adjoint = 1.0;
    for (int i = idx; i >= 0; --i) {
        if (tape[i].adjoint != 0.0) {
            tape[i].backward(tape[i].adjoint);
        }
    }
}

// Reset all adjoints to zero (for multiple backward passes)
inline void reset_adjoints() {
    for (auto& entry : get_tape())
        entry.adjoint = 0.0;
}

// Clear the entire tape (call before a new computation)
inline void clear_tape() {
    get_tape().clear();
}
