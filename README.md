# qcostas

Quantum algorithm for **generating Costas arrays** with **Grover’s search** (Qiskit).  
It builds a Boolean oracle for the Costas constraints, runs Grover, decodes a permutation, verifies the Costas property, and plots the result. Best suited for **small n** (the oracle grows quickly).

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from qcostas import generate_costas, plot_costas
import matplotlib.pyplot as plt

n = 3
res = generate_costas(n)

print("Found permutation:", res.permutation, "Costas?", res.is_costas)
plot_costas(res.permutation)
plt.show()
```

You can also open and run **`example.ipynb`**.

## What you get

`generate_costas(n)` returns a `CostasResult` dataclass:

- `permutation: list[int]` - 1‑indexed columns (one per row)
- `is_costas: bool` - sanity check of the Costas property
- `N, M, r: int` - search space size, target count estimate, Grover iterations
- `grover, problem, result` - Qiskit objects if you want to inspect the circuit/results

To visualize, call `plot_costas(permutation)`.

## Notes & limits

- This is a **research/educational** demo. Grover offers quadratic speedup, but the oracle (Costas constraints) makes circuits large; use **small n**.
- By default it uses Qiskit’s `StatevectorSampler` (simulator). You can swap in another sampler if you want hardware/backend runs.

## License

MIT — see `LICENSE`.
