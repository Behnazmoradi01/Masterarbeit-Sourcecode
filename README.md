# Masterarbeit Sourcecode

Source code for my master's thesis on hardware-oriented division algorithms.

The thesis compares four approaches for computing reciprocals and divisions
in floating-point hardware, focusing on iteration count and numerical error
at different precision levels (FP32, BF16, FP8):

- Newton-Raphson (reciprocal: 1/b)
- Goldschmidt classic (division: a/b)
- Goldschmidt binomial (division: a/b, order 2 and 3)
- Restoring long division (integer, for operation-count baseline)

The simulations are mostly in Python. There is also a small C++ prototype
in main.cpp that shows quantized Newton-Raphson steps at reduced precision.

## Folder structure

- `src/` — algorithm implementations
- `scripts/` — experiment scripts and table generators
- `results/` — CSV output files and stress test results
- `tests/` — test suite (Python + C++)

## How to run

Install the package first:

```bash
pip install -e .
```

Then run the demo scripts:

```bash
python demo_newton_range.py
python demo_goldschmidt_debug.py
python plot_test_chart.py
python plot_thesis_charts.py
```

Charts are saved as PNG files in the project root.

To run the C++ tests:

```bash
g++ -std=c++17 -o test_cpp tests/test_algorithms.cpp -lm
./test_cpp
```

The C++ tests cover the core algorithm logic (NR, Goldschmidt classic, binomial).
The fixed-point simulation tests are only in Python for now.

To reproduce the result tables:

```bash
python scripts/nr_random_sweep.py --out results/nr_random_sweep_summary.csv
python scripts/gs_random_sweep.py --out results/gs_random_sweep_summary.csv
python scripts/gb_random_sweep.py --out results/gb_random_sweep_summary.csv
python scripts/generate_tables.py
python scripts/fp_stress_sweep.py
python scripts/validate_magic_constants.py
```
