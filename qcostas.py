import math
from qiskit.circuit.library import PhaseOracle
from qiskit_algorithms import Grover, AmplificationProblem
from qiskit.primitives import StatevectorSampler


def costas_oracle(n):
    m = (n - 1).bit_length()
    V = lambda i, b: f"x{i}_{b}"
    XNOR = lambda a, b: f"~({a}^{b})"
    eq = lambda A, B: "(" + "&".join(XNOR(a, b) for a, b in zip(A, B)) + ")"
    ne = lambda A, B: "(" + "|".join(f"({a}^{b})" for a, b in zip(A, B)) + ")"

    def const_eq(A, v):
        bits = [(v >> k) & 1 for k in range(m)]
        return "(" + "&".join(A[k] if bits[k] else f"~{A[k]}" for k in range(m)) + ")"

    def add(A, B):
        c = "0"
        S = []
        for k in range(m):
            ax, bx = A[k], B[k]
            s = f"(({ax}^{bx})^{c})" if c != "0" else f"({ax}^{bx})"
            S.append(s)
            c = f"(({ax}&{bx})|(({ax}^{bx})&{c}))" if c != "0" else f"({ax}&{bx})"
        S.append(c)
        return S  # length m+1

    R = [[V(i, k) for k in range(m)] for i in range(n)]
    clauses = []
    # domain r_i in {0,..,n-1}
    for i in range(n):
        for v in range(n, 1 << m):
            clauses.append(f"~{const_eq(R[i],v)}")
    # all-different rows
    for i in range(n):
        for j in range(i + 1, n):
            clauses.append(ne(R[i], R[j]))
    # Costas: differences unique at each lag d (use a+d == b+c equality test)
    for d in range(1, n):
        P = [(i, i + d) for i in range(n - d)]
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                i, j = P[a]
                k, l = P[b]
                clauses.append(f"~{eq(add(R[i],R[l]),add(R[j],R[k]))}")
    expr = " & ".join(clauses)
    var_order = [v for row in R for v in row]
    return PhaseOracle(expr, var_order=var_order), m


def decode(n, m, bitstr):
    bits = bitstr[::-1]  # little-endian
    cols = []  # decode per-row m-bit int -> 1..n
    for i in range(n):
        v = 0
        for k in range(m):
            v |= (bits[i * m + k] == "1") << k
        cols.append(v + 1)
    return cols


def is_costas(p):
    s = set()
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            t = (j - i, p[j] - p[i])
            if t in s:
                return False
            s.add(t)
    return True


def generate_costas(n):
    oracle, m = costas_oracle(n)
    N = 2 ** (m * n)
    # approximate upper bound (doi:10.1109/TIT.2022.3202507)
    M = round(math.factorial(n) * math.exp(-0.135 * n))
    theta = math.asin(math.sqrt(M / N))
    r = max(1, round(math.pi / (4 * theta)))  # near-optimal Grover iterations
    grover = Grover(iterations=r, sampler=StatevectorSampler())
    problem = AmplificationProblem(oracle)
    result = grover.amplify(problem)
    permutation = decode(n, m, result.top_measurement)
    return permutation, {
        "grover": grover,
        "problem": problem,
        "result": result,
        "is_costas": is_costas(permutation),
        "N": N,
        "M": M,
        "r": r,
    }
