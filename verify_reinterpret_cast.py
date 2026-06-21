"""Verify that the reinterpret cast (float <-> uint32) is correct.

This script proves that _float32_to_uint32 uses struct.pack/unpack
(bit-level reinterpretation), NOT int(b) arithmetic conversion.

Expected IEEE 754 single-precision encodings:
    1.0  -> 0x3F800000
    2.0  -> 0x40000000
    0.5  -> 0x3F000000
    5.0  -> 0x40A00000
    pi   -> 0x40490FDB

If int(b) were used instead, 1.0 would give 0x00000001 — clearly wrong.
"""

import math
from nr_reciprocal.basic import (
    _float32_to_uint32,
    _uint32_to_float32,
    MAGIC_CONST_FP32,
)

KNOWN_ENCODINGS = {
    1.0: 0x3F800000,
    2.0: 0x40000000,
    0.5: 0x3F000000,
    5.0: 0x40A00000,
}

print("=" * 72)
print("  REINTERPRET CAST VERIFICATION  (struct.pack/unpack, NOT int(b))")
print("=" * 72)

# --- Step 1: Verify known IEEE 754 bit patterns ---
print("\n--- Step 1: IEEE 754 bit-pattern check ---")
all_ok = True
for val, expected_hex in KNOWN_ENCODINGS.items():
    actual = _float32_to_uint32(val)
    ok = actual == expected_hex
    status = "OK" if ok else "FAIL"
    print(f"  {val:6.1f}  ->  0x{actual:08X}  (expected 0x{expected_hex:08X})  [{status}]")
    if not ok:
        all_ok = False

# --- Step 2: Full magic-seed table ---
print("\n--- Step 2: Magic-constant seed table ---")
print(f"  MAGIC_CONST_FP32 = 0x{MAGIC_CONST_FP32:08X}")
print()
print(f"  {'b':>10s}  {'b_bits':>12s}  {'seed_int':>12s}  {'x0':>10s}  {'exact':>10s}  {'rel_err':>10s}")
print("  " + "-" * 68)

for b in [1.0, 2.0, 0.5, 5.0, math.pi]:
    b_int = _float32_to_uint32(b)
    seed_int = (MAGIC_CONST_FP32 - b_int) & 0xFFFFFFFF
    x0 = _uint32_to_float32(seed_int)
    exact = 1.0 / b
    rel_err = abs(x0 / exact - 1.0)
    print(f"  {b:10.4f}  0x{b_int:08X}  0x{seed_int:08X}  {x0:10.6f}  {exact:10.6f}  {rel_err:.4e}")

# --- Step 3: Round-trip check ---
print("\n--- Step 3: Round-trip float -> uint32 -> float ---")
for val in [1.0, 2.0, 0.5, 5.0, math.pi, 42.0, 0.1]:
    bits = _float32_to_uint32(val)
    back = _uint32_to_float32(bits)
    # Note: 0.1 cannot be exactly represented in FP32, so round-trip
    # goes through FP32 precision (not FP64).  We check FP32 fidelity.
    import struct
    val_as_f32 = struct.unpack("<f", struct.pack("<f", val))[0]
    ok = back == val_as_f32
    print(f"  {val:10.6f}  ->  0x{bits:08X}  ->  {back:10.6f}  [{'OK' if ok else 'FAIL'}]")

# --- Final verdict ---
print("\n" + "=" * 72)
if all_ok:
    print("  RESULT: Reinterpret cast is CORRECT (struct-based, not int(b)).")
else:
    print("  RESULT: FAILURE — bit patterns do not match IEEE 754!")
print("=" * 72)
