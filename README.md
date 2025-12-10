# BFGS in Practice: Convergence, Efficiency, and Line Search Strategies

**AM 205 Final Project**
**Authors:** Lixuan Wei, Feiyang Wu
**Date:** December 2025

## Overview

This project provides a comprehensive numerical analysis of the BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton optimization method. We investigate its convergence properties, computational efficiency, and the impact of different line search strategies through carefully designed experiments.

## Repository Structure

```
.
├── exp1/                    # Experiment 1: Basic convergence verification
│   ├── *.png               # Trajectory and convergence plots
│   └── *.csv               # Numerical results and iteration data
│
├── exp3/                    # Experiment 3: Line search strategy comparison
│   ├── fig*.png            # Comparative visualizations
│   └── *.csv               # Performance metrics
│
├── exp4/                    # Experiment 4: Computational efficiency analysis
│   ├── *.png               # Efficiency comparison plots
│   └── *.csv               # Timing data
│
├── experiment1_simple.py           # Convergence verification implementation
├── experiment3_line_search.py      # Line search comparison implementation
└── time_consumption.py             # Computational cost analysis
```

## Key Experiments

### Experiment 1: Convergence Analysis
Compares BFGS with gradient descent and Newton's method on the Rosenbrock function, demonstrating:
- Superlinear convergence rate (order ~1.25)
- Performance between linear (gradient descent) and quadratic (Newton's method)
- Convergence to machine precision without Hessian computations

### Experiment 3: Line Search Strategies
Investigates the impact of different line search methods:
- No line search (fixed step α=1)
- Armijo condition (sufficient decrease only)
- Strong Wolfe conditions (sufficient decrease + curvature)

### Experiment 4: Computational Efficiency
Analyzes the O(n²) vs O(n³) complexity difference between BFGS and Newton's method:
- Time per iteration scaling with problem dimension
- Crossover point analysis
- Speedup ratios demonstrating BFGS's advantage for large-scale problems

## Test Functions

- **Rosenbrock Function:** Primary test case for 2D convergence analysis
- **LogSumExp + L2 Regularization:** Dense Hessian test for efficiency comparison

## Key Results

- **BFGS achieves superlinear convergence** without computing Hessians
- **O(n²) complexity per iteration** provides decisive advantage over Newton's O(n³) for dimensions n > 50
- **Strong Wolfe conditions** guarantee positive definiteness of Hessian approximation
- **Speedup of 520×** observed at n=200 compared to Newton's method

## Course Information

**Course:** AM 205 - Advanced Scientific Computing: Numerical Methods
**Institution:** Harvard University
**Semester:** Fall 2025
