from dataclasses import dataclass
import matplotlib.pyplot as plt
from qiskit.circuit.library import PhaseOracleGate
from qiskit_algorithms import Grover, AmplificationProblem, GroverResult
from qiskit.primitives import StatevectorSampler


@dataclass(frozen=True)
class CostasResult:
    N: int
    iterations: int
    permutation: list[int]
    is_costas: bool
    grover: Grover
    problem: AmplificationProblem
    result: GroverResult


def costas_oracle(n: int) -> tuple[PhaseOracleGate, int]:
    m = (n - 1).bit_length()
    V = lambda i, b: f"x{i}_{b}"
    XNOR = lambda a, b: f"~({a}^{b})"
    ne = lambda A, B: "(" + "|".join(f"({a}^{b})" for a, b in zip(A, B)) + ")"

    def const_eq(A, v: int) -> str:
        return (
            "("
            + "&".join((A[k] if ((v >> k) & 1) else f"~{A[k]}") for k in range(len(A)))
            + ")"
        )

    def lt(A, B) -> str:
        m = len(A)
        terms = []
        for k in range(m - 1, -1, -1):
            eq_pref = (
                "&".join(XNOR(A[j], B[j]) for j in range(k + 1, m))
                if k < m - 1
                else "1"
            )
            terms.append(
                f"(({eq_pref})&(~{A[k]}&{B[k]}))"
                if eq_pref != "1"
                else f"(~{A[k]}&{B[k]})"
            )
        return "(" + "|".join(terms) + ")"

    def add(A, B) -> list[str]:
        c = "0"
        S = []
        for k in range(m):
            ax, bx = A[k], B[k]
            S.append(f"(({ax}^{bx})^{c})" if c != "0" else f"({ax}^{bx})")
            c = f"(({ax}&{bx})|(({ax}^{bx})&{c}))" if c != "0" else f"({ax}&{bx})"
        S.append(c)
        return S  # length m+1

    R = [[V(i, k) for k in range(m)] for i in range(n)]
    clauses = []
    # domain r_i in {0,..,n-1}
    if (1 << m) != n:
        for i in range(n):
            for v in range(n, 1 << m):
                clauses.append(f"~{const_eq(R[i], v)}")
    # all-different rows
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append(ne(R[i], R[j]))
    # Costas: differences unique at each lag d (use a+d == b+c equality test)
    for d in range(1, n):
        for i in range(0, n - d):
            for j in range(i + 1, n - d):
                S_left = add(R[i], R[j + d])  # f(i) + f(j+d)
                S_right = add(R[j], R[i + d])  # f(j) + f(i+d)
                clauses.append(
                    "(" + "|".join(f"({a}^{b})" for a, b in zip(S_left, S_right)) + ")"
                )
    # symmetry break (horizontal reflection): p[1] < p[n]
    if n >= 2:
        clauses.append(lt(R[0], R[n - 1]))
    expr = " & ".join(clauses)
    var_order = [v for row in R for v in row]
    return PhaseOracleGate(expr, var_order=var_order, label="CostasOracle"), m


def decode(n: int, m: int, bitstr: str) -> list[int]:
    bits = bitstr[::-1]  # little-endian
    cols = []  # decode per-row m-bit int -> 1..n
    for i in range(n):
        v = 0
        for k in range(m):
            v |= (bits[i * m + k] == "1") << k
        cols.append(v + 1)
    return cols


def is_costas(p: list[int]) -> bool:
    s = set()
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            t = (j - i, p[j] - p[i])
            if t in s:
                return False
            s.add(t)
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
    oracle, m = costas_oracle(n)
    N = 2 ** (m * n)
    grover = Grover(sampler=StatevectorSampler(), sample_from_iterations=True)
    problem = AmplificationProblem(oracle)
    result: GroverResult = grover.amplify(problem)
    permutation = decode(n, m, result.top_measurement)  # type: ignore
    return CostasResult(
        N=N,
        iterations=result.iterations[-1],
        grover=grover,
        problem=problem,
        result=result,
        permutation=permutation,
        is_costas=is_costas(permutation),
    )
