#!/usr/bin/env bash
set -e

# ---- 0. Submodules ----
# xsimd (CPU SIMD) and cutlass (GPU) are header-only and just need to be
# checked out. blis must additionally be *compiled* before the BLIS
# micro-kernel tests; that build is driven on demand by test/Makefile
# (`make blis` / `make test_blis_lof`), not here.
echo "==> Updating git submodules..."
git submodule update --init --recursive

# ---- 1. C++ headers via CMake ----
echo "==> Installing C++ headers..."
cmake -B build
cmake --build build

# ---- 2. Python/Torch extension ----
echo "==> Installing Python/Torch extension..."

# Check for venv
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_PREFIX" ]]; then
    echo "WARNING: No active virtual environment detected."
    read -p "Continue anyway? [y/N] " yn
    [[ "$yn" =~ ^[Yy]$ ]] || exit 1
fi

pip install -e .
# --no-build-isolation

echo "==> Done. Headers installed system-wide; Python package installed in editable mode."
