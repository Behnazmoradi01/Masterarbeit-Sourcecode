from __future__ import annotations
from typing import Dict, Tuple
def long_divide_unsigned(a: int, b: int, *, bits: int) -> Tuple[int, int, int, Dict[str, int]]:
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("a and b must be int")
    if not isinstance(bits, int):
        raise TypeError("bits must be int")
    if bits <= 0:
        raise ValueError("bits must be > 0")
    mask = (1 << bits) - 1
    a = a & mask
    b = b & mask
    if b == 0:
        raise ValueError("b must be > 0")
    q = 0
    r = 0
    ops: Dict[str, int] = {
        "shifts": 0,
        "comparisons": 0,
        "subtractions": 0,
        "additions_restores": 0,
        "bitwise_and_or": 0,
        "assignments": 0,
        "loop_iterations": 0,
    }
    ops["assignments"] += 2
    for i in range(bits - 1, -1, -1):
        ops["loop_iterations"] += 1
        bit = (a >> i) & 1
        ops["shifts"] += 1
        ops["bitwise_and_or"] += 1
        r = (r << 1) | bit
        ops["shifts"] += 1
        ops["bitwise_and_or"] += 1
        ops["assignments"] += 1
        t = r - b
        ops["subtractions"] += 1
        ops["assignments"] += 1
        ops["comparisons"] += 1
        if t >= 0:
            r = t
            ops["assignments"] += 1
            q = q | (1 << i)
            ops["shifts"] += 1
            ops["bitwise_and_or"] += 1
            ops["assignments"] += 1
    iterations = bits
    return q & mask, r & mask, iterations, ops
