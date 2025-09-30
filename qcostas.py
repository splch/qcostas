from dataclasses import dataclass
from functools import reduce
from typing import cast
import matplotlib.pyplot as plt
from qiskit_algorithms import Grover, GroverResult, AmplificationProblem
from qiskit.circuit.library import PhaseOracleGate
from qiskit.primitives import StatevectorSampler


@dataclass(frozen=True)
class CostasResult:
    order: int
    qubits: int
    grover: Grover
    problem: AmplificationProblem
    result: GroverResult
    iterations: int
    permutation: list[int]
    is_costas: bool


def costas_oracle(n: int) -> tuple[PhaseOracleGate, int]:
    m = (n - 1).bit_length()
    V = lambda i, k: f"x{i:03d}_{k:02d}"  # zero-padded names -> natural variable order
    R = [[V(i, k) for k in range(m)] for i in range(n)]  # row registers (little-endian)
    XNOR = lambda a, b: f"~({a}^{b})"  # logical equality
    NE = (
        lambda A, B: "(" + "|".join(f"({a}^{b})" for a, b in zip(A, B)) + ")"
    )  # not equal
    EQ = (
        lambda A, B: "("
        + "&".join((A[k] if (B >> k) & 1 else f"~{A[k]}") for k in range(len(A)))
        + ")"
    )  # equal to integer B
    LT = (
        lambda A, B: "("
        + "|".join(
            (
                f"(({ '&'.join(XNOR(A[j], B[j]) for j in range(k+1, len(A))) })&(~{A[k]}&{B[k]}))"
                if k < len(A) - 1
                else f"(~{A[k]}&{B[k]})"
            )
            for k in range(len(A) - 1, -1, -1)
        )
        + ")"
    )  # less than integer B
    ADD = lambda A, B: (lambda S_c: S_c[0] + [S_c[1]])(
        reduce(
            lambda acc, ab: (
                acc[0]
                + [
                    (
                        f"(({ab[0]}^{ab[1]})^{acc[1]})"
                        if acc[1] != "0"
                        else f"({ab[0]}^{ab[1]})"
                    )
                ],
                (
                    f"(({ab[0]}&{ab[1]})|(({ab[0]}^{ab[1]})&{acc[1]}))"
                    if acc[1] != "0"
                    else f"({ab[0]}&{ab[1]})"
                ),
            ),
            zip(A, B),
            cast(tuple[list[str], str], ([], "0")),
        )
    )  # add two binary numbers, return sum bits (little-endian)
    clauses = []
    # domain clamp if n not a power of two
    if (1 << m) != n:
        clauses += [f"~{EQ(R[i], v)}" for i in range(n) for v in range(n, 1 << m)]
    # all different
    clauses += [NE(R[i], R[j]) for i in range(n) for j in range(i + 1, n)]
    # Costas cross-sum inequality at each lag d
    for d in range(1, n):
        for i in range(0, n - d):
            for j in range(i + 1, n - d):
                L = ADD(R[i], R[j + d])  # f(i) + f(j+d)
                Rhs = ADD(R[j], R[i + d])  # f(j) + f(i+d)
                clauses.append(NE(L, Rhs))
    # symmetry halving: p[1] < p[n]
    if n >= 2:
        clauses.append(LT(R[0], R[n - 1]))
    expr = " & ".join(clauses)
    return PhaseOracleGate(expr, label="CostasOracle"), m


def decode(n: int, m: int, bitstr: str) -> list[int]:
    bits = bitstr[::-1]  # little-endian measurement
    return [1 + sum((bits[i * m + k] == "1") << k for k in range(m)) for i in range(n)]


def is_costas(p: list[int]) -> bool:
    seen = set()
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            t = (j - i, p[j] - p[i])
            if t in seen:
                return False
            seen.add(t)
    return True


def plot_costas(p: list[int]):
    n = len(p)
    ax = plt.imshow(
        [[j + 1 == p[n - 1 - i] for j in range(n)] for i in range(n)],
        cmap="Greys",
        vmin=0,
        vmax=1,
    ).axes
    ax.set_xticks(range(n), range(1, n + 1))
    ax.set_yticks(range(n), range(n, 0, -1))
    ax.set_xticks([i - 0.5 for i in range(1, n)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, n)], minor=True)
    ax.grid(which="minor", c="#ccc")


def generate_costas(n: int) -> CostasResult:
    oracle, qubits = costas_oracle(n)
    grover = Grover(sampler=StatevectorSampler(), sample_from_iterations=True)
    problem = AmplificationProblem(oracle)
    result = grover.amplify(problem)
    permutation = decode(n, qubits, result.top_measurement)  # type: ignore
    return CostasResult(
        order=n,
        qubits=qubits,
        grover=grover,
        problem=problem,
        result=result,
        iterations=result.iterations[-1],
        permutation=permutation,
        is_costas=is_costas(permutation),
    )
