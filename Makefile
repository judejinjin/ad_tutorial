# Makefile for Automatic Differentiation examples from ad.md
#
# Targets:
#   make          — build all examples
#   make run      — build and run all examples
#   make clean    — remove build artifacts
#
#   make forward_mode      — build forward-mode example
#   make reverse_mode      — build reverse-mode example
#   make jacobian_forward  — build Jacobian (forward) example
#   make jacobian_reverse  — build Jacobian (reverse) example

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
SRCDIR   := src
BUILDDIR := build

SOURCES := forward_mode reverse_mode jacobian_forward jacobian_reverse
TARGETS := $(addprefix $(BUILDDIR)/,$(SOURCES))

.PHONY: all run clean help

all: $(TARGETS)

# ── Build rules ──────────────────────────────────────────────────────

$(BUILDDIR)/forward_mode: $(SRCDIR)/forward_mode.cpp $(SRCDIR)/dual.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(BUILDDIR)/reverse_mode: $(SRCDIR)/reverse_mode.cpp $(SRCDIR)/var.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(BUILDDIR)/jacobian_forward: $(SRCDIR)/jacobian_forward.cpp $(SRCDIR)/dual.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(BUILDDIR)/jacobian_reverse: $(SRCDIR)/jacobian_reverse.cpp $(SRCDIR)/var.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# ── Convenience targets ─────────────────────────────────────────────

forward_mode: $(BUILDDIR)/forward_mode
reverse_mode: $(BUILDDIR)/reverse_mode
jacobian_forward: $(BUILDDIR)/jacobian_forward
jacobian_reverse: $(BUILDDIR)/jacobian_reverse

# ── Run all examples ─────────────────────────────────────────────────

run: all
	@echo ""
	@echo "========================================"
	@echo " Running: Forward-Mode AD"
	@echo "========================================"
	@$(BUILDDIR)/forward_mode
	@echo ""
	@echo "========================================"
	@echo " Running: Reverse-Mode AD"
	@echo "========================================"
	@$(BUILDDIR)/reverse_mode
	@echo ""
	@echo "========================================"
	@echo " Running: Jacobian (Forward-Mode)"
	@echo "========================================"
	@$(BUILDDIR)/jacobian_forward
	@echo ""
	@echo "========================================"
	@echo " Running: Jacobian (Reverse-Mode)"
	@echo "========================================"
	@$(BUILDDIR)/jacobian_reverse

# ── Clean ────────────────────────────────────────────────────────────

clean:
	rm -rf $(BUILDDIR)

# ── Help ─────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo "  all              — build all examples (default)"
	@echo "  run              — build and run all examples"
	@echo "  forward_mode     — forward-mode AD with dual numbers"
	@echo "  reverse_mode     — reverse-mode AD with tape"
	@echo "  jacobian_forward — full Jacobian via forward mode"
	@echo "  jacobian_reverse — full Jacobian via reverse mode"
	@echo "  clean            — remove build artifacts"
	@echo "  help             — show this message"
