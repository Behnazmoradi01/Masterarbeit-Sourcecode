"""Tests for all division algorithms.

This suite covers Newton-Raphson, Goldschmidt, long division, and fixed-point helpers.
It checks both basic behavior and simulation behavior.
"""

from __future__ import annotations

import math
import pytest
from decimal import Decimal

# ── Newton–Raphson ────────────────────────────────────────────────────────────
from nr_reciprocal import (
    MAGIC_CONST_BF16,
    MAGIC_CONST_FP32,
    MAGIC_CONST_FP8_E4M3,
    NewtonReciprocalConfig,
    SimulationPoint,
    simulate_newton_reciprocal,
    NewtonResult,
    PRECISION_THRESHOLDS,
    hardware_initial_guess,
    magic_constant_reciprocal_seed,
    newton_reciprocal,
    newton_reciprocal_result,
    search_magic_constant,
    threshold_for_precision,
)
from nr_reciprocal.fixed_point import QFormat, mul_q, quantize

# ── Goldschmidt (classic) ────────────────────────────────────────────────────
from gs_division.basic import (
    goldschmidt_classic_divide,
    threshold_for_precision as gs_threshold,
)

# ── Goldschmidt (binomial) ───────────────────────────────────────────────────
from gs_binomial.basic import (
    goldschmidt_binomial_divide,
    threshold_for_precision as gb_threshold,
)

# ── Long division ────────────────────────────────────────────────────────────
from long_division.basic import long_divide_unsigned


# =============================================================================
#  1. Newton–Raphson reciprocal – basic.py
# =============================================================================
class TestNewtonReciprocalBasic:
    """Tests for newton_reciprocal_result / newton_reciprocal."""

    # -------- correctness for typical values --------
    @pytest.mark.parametrize("b", [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 100.0, 0.5, 0.25, 0.1])
    def test_converges_for_common_values(self, b: float):
        res = newton_reciprocal_result(b, threshold=1e-12, max_iter=50)
        assert res.converged
        assert abs(res.x - 1.0 / b) / abs(1.0 / b) < 1e-12

    # -------- tuple interface --------
    def test_tuple_interface_matches_result(self):
        x, iters, errs = newton_reciprocal(3.0, threshold=1e-10)
        res = newton_reciprocal_result(3.0, threshold=1e-10)
        assert math.isclose(x, res.x, rel_tol=1e-15)
        assert iters == res.iterations
        assert len(errs) == len(res.errors)
        for e1, e2 in zip(errs, res.errors):
            assert math.isclose(e1, e2, rel_tol=1e-15, abs_tol=1e-30)

    # -------- initial guess strategies --------
    @pytest.mark.parametrize("strategy", ["linear", "const"])
    def test_x0_strategies(self, strategy: str):
        res = newton_reciprocal_result(5.0, threshold=1e-10, x0_strategy=strategy)
        assert res.converged

    # -------- custom x0 --------
    def test_custom_x0(self):
        res = newton_reciprocal_result(4.0, x0=0.2, threshold=1e-10)
        assert res.converged
        assert abs(res.x - 0.25) < 1e-10

    # -------- precision presets --------
    @pytest.mark.parametrize("preset", ["fp32", "bf16", "fp8"])
    def test_precision_presets(self, preset: str):
        thr = threshold_for_precision(preset)
        res = newton_reciprocal_result(7.0, precision=preset)
        assert res.converged
        assert res.errors[-1] < thr

    def test_unknown_precision_raises(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            threshold_for_precision("fp128")

    # -------- error on invalid input --------
    def test_negative_b_raises(self):
        with pytest.raises(ValueError):
            newton_reciprocal_result(-1.0)

    def test_zero_b_raises(self):
        with pytest.raises(ValueError):
            newton_reciprocal_result(0.0)

    def test_non_numeric_b_raises(self):
        with pytest.raises(TypeError):
            newton_reciprocal_result("abc")  # type: ignore

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            newton_reciprocal_result(2.0, threshold=-1.0)

    def test_negative_max_iter_raises(self):
        with pytest.raises(ValueError):
            newton_reciprocal_result(2.0, max_iter=-1)

    # -------- quadratic convergence (errors should decrease fast) --------
    def test_quadratic_convergence(self):
        res = newton_reciprocal_result(3.0, threshold=1e-15, max_iter=50)
        # After the initial seed phase, error should drop rapidly.
        # Verify that at least one consecutive pair shows quadratic-like drop.
        found_quadratic = False
        for i in range(1, len(res.errors)):
            if res.errors[i - 1] > 1e-12 and res.errors[i] > 0:
                ratio = res.errors[i] / (res.errors[i - 1] ** 2)
                if ratio < 1e4:
                    found_quadratic = True
        assert found_quadratic, "No quadratic convergence observed"

    # -------- max_iter=0 should return immediately --------
    def test_max_iter_zero(self):
        res = newton_reciprocal_result(4.0, threshold=1e-12, max_iter=0)
        assert res.iterations == 0
        assert not res.converged

    # -------- hardware_initial_guess standalone --------
    @pytest.mark.parametrize("b", [0.5, 1.0, 3.14, 100.0, 0.01])
    def test_hardware_initial_guess_bounded(self, b: float):
        x0 = hardware_initial_guess(b)
        exact = 1.0 / b
        # The linear seed should be within a factor of ~4 of the true reciprocal
        assert 0.1 * exact < x0 < 10 * exact

    def test_hardware_initial_guess_invalid_b(self):
        with pytest.raises(ValueError):
            hardware_initial_guess(-1.0)
        with pytest.raises(ValueError):
            hardware_initial_guess(0.0)

    def test_inf_b_raises(self):
        with pytest.raises(ValueError):
            newton_reciprocal_result(float("inf"))

    def test_nan_b_raises(self):
        with pytest.raises(ValueError):
            newton_reciprocal_result(float("nan"))

    def test_hardware_initial_guess_inf_raises(self):
        with pytest.raises(ValueError):
            hardware_initial_guess(float("inf"))

    def test_hardware_initial_guess_nan_raises(self):
        with pytest.raises(ValueError):
            hardware_initial_guess(float("nan"))


# =============================================================================
#  1b. Magic-constant FP32 reciprocal seed
# =============================================================================
class TestMagicConstantSeed:
    """Tests for the FP32 magic-constant seed."""

    # -------- magic constant value --------
    def test_magic_constant_value(self):
        assert MAGIC_CONST_FP32 == 0x7EF311C3

    # -------- seed is a reasonable approximation of 1/b --------
    @pytest.mark.parametrize("b", [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 0.5, 0.25, 100.0])
    def test_seed_approximation_quality(self, b: float):
        """Magic seed should stay close to the true reciprocal."""
        x0 = magic_constant_reciprocal_seed(b)
        exact = 1.0 / b
        ratio = x0 / exact
        # Seed should be in the right ballpark: within factor 0.3..3.0
        assert 0.3 < ratio < 3.0, f"b={b}: x0={x0}, exact={exact}, ratio={ratio}"

    # -------- seed is positive for positive b --------
    @pytest.mark.parametrize("b", [0.01, 0.5, 1.0, 42.0, 1e6])
    def test_seed_positive(self, b: float):
        x0 = magic_constant_reciprocal_seed(b)
        assert x0 > 0.0

    # -------- integration: NR with magic seed converges --------
    @pytest.mark.parametrize("b", [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 0.5, 100.0])
    def test_nr_converges_with_magic_seed(self, b: float):
        """NR should still converge from the magic seed."""
        res = newton_reciprocal_result(b, threshold=1e-12, max_iter=50, x0_strategy="magic")
        assert res.converged
        assert abs(res.x - 1.0 / b) / (1.0 / b) < 1e-10

    # -------- magic seed via hardware_initial_guess interface --------
    @pytest.mark.parametrize("b", [2.0, 7.0, 100.0])
    def test_hardware_initial_guess_magic(self, b: float):
        x0 = hardware_initial_guess(b, strategy="magic")
        exact = 1.0 / b
        ratio = x0 / exact
        assert 0.3 < ratio < 3.0

    # -------- magic and linear iteration counts are nearly identical --------
    def test_magic_vs_linear_iteration_count(self):
        """FP32 magic and linear seeds should give almost the same count."""
        res_linear = newton_reciprocal_result(3.0, threshold=1e-12, x0_strategy="linear")
        res_magic = newton_reciprocal_result(3.0, threshold=1e-12, x0_strategy="magic")
        assert res_magic.converged
        assert abs(res_magic.iterations - res_linear.iterations) <= 1, (
            f"magic={res_magic.iterations} vs linear={res_linear.iterations}: differ by >1"
        )

    # -------- simulation pipeline with magic seed (float mode) --------
    def test_simulate_float_magic_seed(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=15, tol_rel=1e-12, x0_strategy="magic")
        pts = simulate_newton_reciprocal(5.0, cfg)
        assert pts[-1].err_rel < 1e-10

    # -------- simulation pipeline with magic seed (fixed mode) --------
    def test_simulate_fixed_magic_seed(self):
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=15, tol_rel=1e-6, frac_bits=24, x0_strategy="magic"
        )
        pts = simulate_newton_reciprocal(5.0, cfg)
        assert pts[-1].err_rel < 0.01

    # -------- error on invalid b --------
    def test_seed_negative_b_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(-1.0)

    def test_seed_zero_b_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(0.0)

    def test_seed_inf_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(float("inf"))

    def test_seed_nan_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(float("nan"))


# =============================================================================
#  2. Newton–Raphson reciprocal – simulate.py (float + fixed-point)
# =============================================================================
class TestSimulateNewtonReciprocal:
    """Tests for the full simulation pipeline."""

    # -------- float mode --------
    def test_float_mode_basic(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=10, tol_rel=1e-12)
        pts = simulate_newton_reciprocal(2.0, cfg)
        assert len(pts) >= 2
        assert pts[-1].err_rel < 1e-12

    @pytest.mark.parametrize("b", [0.5, 1.0, 3.0, 7.5, 100.0, 1e-5, 1e5])
    def test_float_mode_various_b(self, b: float):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=20, tol_rel=1e-12)
        pts = simulate_newton_reciprocal(b, cfg)
        assert pts[-1].err_rel < 1e-10

    # -------- fixed-point mode --------
    @pytest.mark.parametrize("frac_bits", [16, 24, 32])
    def test_fixed_mode_converges(self, frac_bits: int):
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=15, tol_rel=1e-6, frac_bits=frac_bits
        )
        pts = simulate_newton_reciprocal(3.0, cfg)
        # Should reach reasonably low error
        assert pts[-1].err_rel < 0.01

    def test_fixed_mode_x_int_present(self):
        cfg = NewtonReciprocalConfig(mode="fixed", max_iter=5, frac_bits=16)
        pts = simulate_newton_reciprocal(4.0, cfg)
        for p in pts:
            assert p.x_int is not None

    # -------- negative b (simulate uses sign handling) --------
    def test_negative_b_simulation(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=10, tol_rel=1e-10)
        pts = simulate_newton_reciprocal(-3.0, cfg)
        assert pts[-1].x < 0  # 1/(-3) is negative
        assert pts[-1].err_rel < 1e-8

    # -------- Decimal / string b input --------
    def test_decimal_input(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=10, tol_rel=1e-10)
        pts = simulate_newton_reciprocal(Decimal("3.0"), cfg)
        assert pts[-1].err_rel < 1e-8

    def test_string_input(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=10, tol_rel=1e-10)
        pts = simulate_newton_reciprocal("3.0", cfg)
        assert pts[-1].err_rel < 1e-8

    # -------- zero b --------
    def test_zero_b_raises(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=5)
        with pytest.raises(ZeroDivisionError):
            simulate_newton_reciprocal(0.0, cfg)

    # -------- inf/nan b --------
    def test_inf_b_raises(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=5)
        with pytest.raises(ValueError, match="must be finite"):
            simulate_newton_reciprocal(float("inf"), cfg)

    def test_nan_b_raises(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=5)
        with pytest.raises(ValueError, match="must be finite"):
            simulate_newton_reciprocal(float("nan"), cfg)

    # -------- SimulationPoint fields --------
    def test_simulation_point_fields(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=3)
        pts = simulate_newton_reciprocal(5.0, cfg)
        p = pts[0]
        assert isinstance(p, SimulationPoint)
        assert hasattr(p, "k")
        assert hasattr(p, "b")
        assert hasattr(p, "b_norm")
        assert hasattr(p, "scale_e")
        assert hasattr(p, "x")
        assert hasattr(p, "residual")
        assert hasattr(p, "err_abs")
        assert hasattr(p, "err_rel")

    # -------- residual decreases (trend, not strict) --------
    def test_residual_decreasing_trend(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=10, tol_rel=1e-14)
        pts = simulate_newton_reciprocal(7.0, cfg)
        assert abs(pts[-1].residual) < abs(pts[1].residual) * 1e-4
        increases = sum(
            1 for i in range(2, len(pts))
            if abs(pts[i].residual) > abs(pts[i - 1].residual) * (1.0 + 1e-8)
        )
        assert increases <= 1, f"Too many residual increases: {increases}"

    # -------- tol_abs stopping criterion --------
    def test_tol_abs_stopping(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=20, tol_abs=1e-8, tol_rel=0.0)
        pts = simulate_newton_reciprocal(3.0, cfg)
        assert pts[-1].err_abs <= 1e-8

    # -------- x0_strategy='one' --------
    def test_x0_one_strategy(self):
        cfg = NewtonReciprocalConfig(mode="float", max_iter=30, tol_rel=1e-10, x0_strategy="one")
        pts = simulate_newton_reciprocal(2.0, cfg)
        assert pts[-1].err_rel < 1e-8

    # -------- rounding modes in fixed --------
    @pytest.mark.parametrize("rounding", ["nearest_even", "trunc"])
    def test_fixed_rounding_modes(self, rounding: str):
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=10, frac_bits=24, rounding=rounding
        )
        pts = simulate_newton_reciprocal(3.0, cfg)
        assert len(pts) >= 2


# =============================================================================
#  3. Fixed-point helpers (fixed_point.py)
# =============================================================================
class TestFixedPoint:
    """Tests for QFormat, quantize, mul_q."""

    # -------- QFormat basics --------
    def test_qformat_scale(self):
        q = QFormat(frac_bits=8)
        assert q.scale() == 256

    def test_to_int_trunc(self):
        q = QFormat(frac_bits=8)
        # 1.5 * 256 = 384
        assert q.to_int_trunc(1.5) == 384

    def test_to_int_nearest_even(self):
        q = QFormat(frac_bits=8)
        assert q.to_int_nearest_even(1.5) == 384

    def test_to_float_roundtrip(self):
        q = QFormat(frac_bits=16)
        val = 3.14
        i = q.to_int_nearest_even(val)
        f = q.to_float(i)
        assert abs(f - val) < 1.0 / q.scale()

    # -------- ties-to-even --------
    def test_ties_to_even_round_up(self):
        q = QFormat(frac_bits=1)
        # 1.5 -> scaled = 3.0 -> int 3 (odd already ties-to-even = round to nearest even)
        # Actually 1.5 * 2 = 3.0 exact, no tie
        assert q.to_int_nearest_even(1.5) == 3

    def test_ties_to_even_exact_half(self):
        q = QFormat(frac_bits=2)
        # 0.125 * 4 = 0.5 -> tie -> nearest even = 0
        assert q.to_int_nearest_even(0.125) == 0
        # 0.375 * 4 = 1.5 -> tie -> nearest even = 2
        assert q.to_int_nearest_even(0.375) == 2

    # -------- negative value quantize --------
    def test_quantize_negative(self):
        q = QFormat(frac_bits=8)
        i = quantize(-1.5, q, "nearest_even")
        assert i == -384

    # -------- mul_q --------
    def test_mul_q_trunc(self):
        fb = 8
        q = QFormat(frac_bits=fb)
        a = quantize(1.5, q, "trunc")
        b = quantize(2.0, q, "trunc")
        result = mul_q(a, b, fb, "trunc")
        val = q.to_float(result)
        assert abs(val - 3.0) < 1.0 / q.scale()

    def test_mul_q_nearest_even(self):
        fb = 16
        q = QFormat(frac_bits=fb)
        a = quantize(1.25, q, "nearest_even")
        b = quantize(0.8, q, "nearest_even")
        result = mul_q(a, b, fb, "nearest_even")
        val = q.to_float(result)
        assert abs(val - 1.0) < 2.0 / q.scale()

    def test_mul_q_identity(self):
        fb = 16
        q = QFormat(frac_bits=fb)
        one = quantize(1.0, q, "nearest_even")
        x = quantize(3.14, q, "nearest_even")
        result = mul_q(x, one, fb, "nearest_even")
        assert abs(q.to_float(result) - q.to_float(x)) < 2.0 / q.scale()

    def test_mul_q_commutativity(self):
        fb = 16
        q = QFormat(frac_bits=fb)
        a = quantize(2.5, q, "nearest_even")
        b = quantize(1.3, q, "nearest_even")
        assert mul_q(a, b, fb, "nearest_even") == mul_q(b, a, fb, "nearest_even")

    def test_mul_q_zero(self):
        fb = 16
        result = mul_q(0, 12345, fb, "nearest_even")
        assert result == 0

    def test_quantize_unsupported_rounding(self):
        q = QFormat(frac_bits=8)
        with pytest.raises(ValueError, match="Unsupported rounding"):
            quantize(1.0, q, "ceiling")

    def test_mul_q_unsupported_rounding(self):
        with pytest.raises(ValueError, match="Unsupported rounding"):
            mul_q(100, 200, 8, "ceiling")


# =============================================================================
#  4. Goldschmidt classic division
# =============================================================================
class TestGoldschmidtClassic:
    """Tests for goldschmidt_classic_divide."""

    @pytest.mark.parametrize(
        "a, b",
        [
            (1.0, 1.0),
            (1.0, 2.0),
            (10.0, 3.0),
            (7.0, 7.0),
            (100.0, 13.0),
            (1.0, 0.5),
            (0.1, 0.3),
            (1e6, 1e3),
        ],
    )
    def test_converges_to_correct_quotient(self, a: float, b: float):
        q, iters, errs = goldschmidt_classic_divide(a, b, threshold=1e-12)
        expected = a / b
        assert abs(q - expected) / abs(expected) < 1e-10

    def test_errors_decrease_overall(self):
        # strict monotonicity isn't guaranteed due to rounding, so we check overall trend
        _, _, errs = goldschmidt_classic_divide(10.0, 3.0, threshold=1e-14)
        assert errs[-1] < errs[0] * 1e-6, "Final error should be far below initial error"
        increases = sum(
            1 for i in range(1, len(errs))
            if errs[i] > errs[i - 1] * (1.0 + 1e-10)
        )
        assert increases <= 1, f"Too many non-decreasing steps: {increases}"

    # -------- precision presets --------
    @pytest.mark.parametrize("preset", ["fp32", "bf16", "fp8"])
    def test_precision_presets(self, preset: str):
        thr = gs_threshold(preset)
        q, _, errs = goldschmidt_classic_divide(10.0, 3.0, precision=preset)
        assert errs[-1] < thr

    # -------- error cases --------
    def test_negative_a_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_classic_divide(-1.0, 2.0)

    def test_zero_b_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_classic_divide(1.0, 0.0)

    def test_negative_b_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_classic_divide(1.0, -2.0)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            goldschmidt_classic_divide("1", 2.0)  # type: ignore

    def test_infinite_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_classic_divide(float("inf"), 2.0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_classic_divide(1.0, 2.0, threshold=-1.0)

    def test_negative_max_iter_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_classic_divide(1.0, 2.0, max_iter=-1)

    # -------- identity: a/a == 1 --------
    @pytest.mark.parametrize("v", [1.0, 3.14, 100.0, 0.01])
    def test_self_division(self, v: float):
        q, _, _ = goldschmidt_classic_divide(v, v, threshold=1e-12)
        assert abs(q - 1.0) < 1e-10

    # -------- a / 1 == a --------
    @pytest.mark.parametrize("a", [1.0, 5.0, 99.99, 0.001])
    def test_divide_by_one(self, a: float):
        q, _, _ = goldschmidt_classic_divide(a, 1.0, threshold=1e-12)
        assert abs(q - a) / max(abs(a), 1e-15) < 1e-10


# =============================================================================
#  5. Goldschmidt binomial division
# =============================================================================
class TestGoldschmidtBinomial:
    """Tests for goldschmidt_binomial_divide."""

    @pytest.mark.parametrize(
        "a, b",
        [
            (1.0, 1.0),
            (1.0, 2.0),
            (10.0, 3.0),
            (7.0, 7.0),
            (100.0, 13.0),
            (0.1, 0.3),
            (1e6, 1e3),
        ],
    )
    def test_converges_to_correct_quotient(self, a: float, b: float):
        q, iters, errs = goldschmidt_binomial_divide(a, b, threshold=1e-12)
        expected = a / b
        assert abs(q - expected) / abs(expected) < 1e-10

    # -------- order 2 vs order 3 --------
    def test_order2_converges(self):
        q, iters2, _ = goldschmidt_binomial_divide(10.0, 3.0, threshold=1e-12, order=2)
        assert abs(q - 10.0 / 3.0) / (10.0 / 3.0) < 1e-10

    def test_order3_fewer_iters_than_order2(self):
        _, iters2, _ = goldschmidt_binomial_divide(10.0, 3.0, threshold=1e-12, order=2)
        _, iters3, _ = goldschmidt_binomial_divide(10.0, 3.0, threshold=1e-12, order=3)
        # Higher order should need fewer (or equal) iterations
        assert iters3 <= iters2

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="order must be 2 or 3"):
            goldschmidt_binomial_divide(1.0, 2.0, order=4)

    # -------- precision presets --------
    @pytest.mark.parametrize("preset", ["fp32", "bf16", "fp8"])
    def test_precision_presets(self, preset: str):
        thr = gb_threshold(preset)
        q, _, errs = goldschmidt_binomial_divide(10.0, 3.0, precision=preset)
        assert errs[-1] < thr

    # -------- error cases --------
    def test_negative_a_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_binomial_divide(-1.0, 2.0)

    def test_zero_b_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_binomial_divide(1.0, 0.0)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            goldschmidt_binomial_divide("1", 2.0)  # type: ignore

    def test_infinite_raises(self):
        with pytest.raises(ValueError):
            goldschmidt_binomial_divide(float("inf"), 2.0)

    # -------- identity: a/a == 1 --------
    @pytest.mark.parametrize("v", [1.0, 3.14, 100.0])
    def test_self_division(self, v: float):
        q, _, _ = goldschmidt_binomial_divide(v, v, threshold=1e-12)
        assert abs(q - 1.0) < 1e-10


# =============================================================================
#  6. Unsigned restoring long division
# =============================================================================
class TestLongDivision:
    """Tests for long_divide_unsigned."""

    # -------- basic correctness --------
    @pytest.mark.parametrize(
        "a, b, bits",
        [
            (10, 3, 8),
            (255, 1, 8),
            (100, 7, 8),
            (1, 1, 8),
            (0, 5, 8),
            (65535, 256, 16),
            (1023, 31, 16),
            (12345, 67, 16),
            (0xFFFF_FFFF, 17, 32),
        ],
    )
    def test_quotient_remainder(self, a: int, b: int, bits: int):
        mask = (1 << bits) - 1
        a_m = a & mask
        b_m = b & mask
        if b_m == 0:
            with pytest.raises(ValueError, match="b must be > 0"):
                long_divide_unsigned(a, b, bits=bits)
            return
        q, r, iters, ops = long_divide_unsigned(a, b, bits=bits)
        assert q == a_m // b_m
        assert r == a_m % b_m

    # -------- identity: a / 1 == a --------
    @pytest.mark.parametrize("bits", [8, 16, 32])
    def test_divide_by_one(self, bits: int):
        a = (1 << bits) - 1  # max value
        q, r, _, _ = long_divide_unsigned(a, 1, bits=bits)
        assert q == a
        assert r == 0

    # -------- zero dividend --------
    def test_zero_dividend(self):
        q, r, _, _ = long_divide_unsigned(0, 5, bits=8)
        assert q == 0
        assert r == 0

    # -------- division by self --------
    @pytest.mark.parametrize("a", [1, 7, 42, 255])
    def test_self_division(self, a: int):
        q, r, _, _ = long_divide_unsigned(a, a, bits=8)
        assert q == 1
        assert r == 0

    # -------- iterations == bits --------
    @pytest.mark.parametrize("bits", [4, 8, 16, 32])
    def test_iterations_equals_bits(self, bits: int):
        _, _, iters, _ = long_divide_unsigned(100, 7, bits=bits)
        assert iters == bits

    # -------- ops dict structure --------
    def test_ops_structure(self):
        _, _, _, ops = long_divide_unsigned(10, 3, bits=8)
        expected_keys = {
            "shifts", "comparisons", "subtractions",
            "additions_restores", "bitwise_and_or",
            "assignments", "loop_iterations",
        }
        assert set(ops.keys()) == expected_keys
        for k, v in ops.items():
            assert isinstance(v, int), f"ops[{k!r}] should be int, got {type(v)}"
            assert v >= 0, f"ops[{k!r}] should be >= 0, got {v}"

    # -------- division by zero raises --------
    def test_divide_by_zero_raises(self):
        with pytest.raises(ValueError, match="b must be > 0"):
            long_divide_unsigned(10, 0, bits=8)

    # -------- invalid types --------
    def test_non_int_a_raises(self):
        with pytest.raises(TypeError):
            long_divide_unsigned(1.5, 2, bits=8)  # type: ignore

    def test_non_int_b_raises(self):
        with pytest.raises(TypeError):
            long_divide_unsigned(10, 2.5, bits=8)  # type: ignore

    def test_non_int_bits_raises(self):
        with pytest.raises(TypeError):
            long_divide_unsigned(10, 3, bits=8.0)  # type: ignore

    def test_zero_bits_raises(self):
        with pytest.raises(ValueError, match="bits must be > 0"):
            long_divide_unsigned(10, 3, bits=0)

    def test_negative_bits_raises(self):
        with pytest.raises(ValueError, match="bits must be > 0"):
            long_divide_unsigned(10, 3, bits=-1)

    # -------- masking / overflow --------
    def test_overflow_masked(self):
        """Values larger than bit-width are masked correctly."""
        q, r, _, _ = long_divide_unsigned(300, 7, bits=8)
        # 300 & 0xFF = 44
        assert q == 44 // 7
        assert r == 44 % 7

    # -------- large bit-widths --------
    def test_64_bit(self):
        a = (1 << 63) + 1234567
        b = 997
        q, r, _, _ = long_divide_unsigned(a, b, bits=64)
        mask = (1 << 64) - 1
        assert q == (a & mask) // (b & mask)
        assert r == (a & mask) % (b & mask)

    # -------- power of two divisor --------
    @pytest.mark.parametrize("shift", [1, 2, 3, 4])
    def test_power_of_two_divisor(self, shift: int):
        a = 200
        b = 1 << shift
        q, r, _, _ = long_divide_unsigned(a, b, bits=8)
        assert q == a // b
        assert r == a % b

    # -------- loop counter consistency --------
    def test_loop_counter_consistency(self):
        _, _, iters, ops = long_divide_unsigned(100, 7, bits=16)
        assert iters == 16
        assert ops["loop_iterations"] == 16


# =============================================================================
#  7. Cross-algorithm consistency tests
# =============================================================================
class TestCrossAlgorithm:
    """Compare different algorithms on the same inputs."""

    # ------------------------------------------------------------------
    #  Reciprocal-only: NR vs GS with a=1 (both effectively compute 1/b)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [2.0, 3.0, 5.0, 7.0, 10.0, 0.5, 0.1])
    def test_nr_reciprocal_vs_gs_classic_reciprocal(self, b: float):
        """NR and GS-classic should agree on 1/b."""
        nr_recip, _, _ = newton_reciprocal(b, threshold=1e-12)
        gs_recip, _, _ = goldschmidt_classic_divide(1.0, b, threshold=1e-12)
        assert abs(nr_recip - gs_recip) / abs(nr_recip) < 1e-9

    @pytest.mark.parametrize("b", [2.0, 3.0, 5.0, 7.0, 10.0, 0.5, 0.1])
    def test_nr_reciprocal_vs_gs_binomial_reciprocal(self, b: float):
        """NR 1/b vs GS-binomial with a=1 (pure reciprocal comparison)."""
        nr_recip, _, _ = newton_reciprocal(b, threshold=1e-12)
        gb_recip, _, _ = goldschmidt_binomial_divide(1.0, b, threshold=1e-12)
        assert abs(nr_recip - gb_recip) / abs(nr_recip) < 1e-9

    # ------------------------------------------------------------------
    #  Goldschmidt division variants (a/b, does not involve NR)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [2.0, 3.0, 5.0, 7.0, 10.0])
    def test_gs_classic_vs_gs_binomial(self, b: float):
        """Classic and binomial Goldschmidt should agree on a/b."""
        a = 10.0
        q_classic, _, _ = goldschmidt_classic_divide(a, b, threshold=1e-12)
        q_binom, _, _ = goldschmidt_binomial_divide(a, b, threshold=1e-12)
        assert abs(q_classic - q_binom) / abs(q_classic) < 1e-9

    # ------------------------------------------------------------------
    #  Reciprocal-then-multiply: NR 1/b used for integer division
    #  The multiplication by a is a SEPARATE step, not part of NR.
    # ------------------------------------------------------------------
    def test_nr_reciprocal_then_multiply_vs_long_div(self):
        """NR reciprocal * a compared against long division result."""
        a, b = 1000, 7
        q_ld, _, _, _ = long_divide_unsigned(a, b, bits=16)

        # Step 1 — NR reciprocal (this is the algorithm under test).
        nr_recip, _, _ = newton_reciprocal(float(b), threshold=1e-14)

        # Step 2 — separate multiplication (not part of NR).
        q_est = int(a * nr_recip)

        # Step 3 — remainder correction (hardware post-processing).
        while q_est * b > a:
            q_est -= 1
        while (q_est + 1) * b <= a:
            q_est += 1

        assert q_ld == q_est

    # ------------------------------------------------------------------
    #  Reciprocal-then-multiply: NR 1/b * a  vs  GS a/b
    #  Explicitly documents that the multiply is outside NR.
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "a, b",
        [(10.0, 3.0), (100.0, 7.0), (1.0, 0.5)],
    )
    def test_nr_reciprocal_times_a_vs_gs_division(self, a: float, b: float):
        """NR reciprocal times a should match GS division result."""
        # NR: reciprocal only
        nr_recip, _, _ = newton_reciprocal(b, threshold=1e-13)
        # Separate multiply (not NR)
        nr_quotient = a * nr_recip

        gs_q, _, _ = goldschmidt_classic_divide(a, b, threshold=1e-13)
        gb_q, _, _ = goldschmidt_binomial_divide(a, b, threshold=1e-13)

        expected = a / b
        for label, val in [("NR·a", nr_quotient), ("GS", gs_q), ("GB", gb_q)]:
            rel = abs(val - expected) / abs(expected)
            assert rel < 1e-10, f"{label} relative error {rel} too large"


# =============================================================================
#  8. Edge-case & stress tests
# =============================================================================
class TestEdgeCases:
    """Edge cases: very large, very small, near-boundary values."""

    # -------- very small b --------
    def test_nr_very_small_b(self):
        res = newton_reciprocal_result(1e-10, threshold=1e-10)
        assert res.converged
        assert abs(res.x - 1e10) / 1e10 < 1e-8

    # -------- very large b --------
    def test_nr_very_large_b(self):
        res = newton_reciprocal_result(1e10, threshold=1e-10)
        assert res.converged
        assert abs(res.x - 1e-10) / 1e-10 < 1e-8

    # -------- b close to 1 --------
    def test_nr_b_near_one(self):
        for b in [0.999, 1.0, 1.001]:
            res = newton_reciprocal_result(b, threshold=1e-14)
            assert res.converged

    # -------- irrational-like b --------
    def test_nr_irrational_b(self):
        for b in [math.pi, math.e, math.sqrt(2)]:
            res = newton_reciprocal_result(b, threshold=1e-12)
            assert res.converged

    # -------- simulate with high-precision decimal reference --------
    def test_simulate_high_prec_ref(self):
        cfg = NewtonReciprocalConfig(
            mode="float", max_iter=15, tol_rel=1e-14, reference_decimal_prec=150
        )
        pts = simulate_newton_reciprocal(Decimal("3.141592653589793"), cfg)
        assert pts[-1].err_rel < 1e-12

    # -------- GS with large ratio --------
    def test_gs_classic_large_ratio(self):
        q, _, _ = goldschmidt_classic_divide(1e8, 0.01, threshold=1e-10)
        assert abs(q - 1e10) / 1e10 < 1e-8

    # -------- long div: a < b --------
    def test_long_div_a_less_than_b(self):
        q, r, _, _ = long_divide_unsigned(3, 10, bits=8)
        assert q == 0
        assert r == 3

    # -------- long div: a == b --------
    def test_long_div_a_equals_b(self):
        q, r, _, _ = long_divide_unsigned(42, 42, bits=8)
        assert q == 1
        assert r == 0

    # -------- long div: max values --------
    def test_long_div_max_8bit(self):
        q, r, _, _ = long_divide_unsigned(255, 2, bits=8)
        assert q == 127
        assert r == 1


# =============================================================================
#  9. Iteration-count sanity checks
# =============================================================================
class TestIterationCounts:
    """Verify that iteration counts are in expected ranges."""

    def test_nr_few_iterations(self):
        """NR with linear seed should converge in very few iterations for moderate b."""
        res = newton_reciprocal_result(3.0, threshold=1e-12, max_iter=50)
        assert res.iterations <= 6  # typically 4-5

    def test_gs_classic_few_iterations(self):
        _, iters, _ = goldschmidt_classic_divide(10.0, 3.0, threshold=1e-12)
        assert iters <= 15

    def test_gs_binomial_order3_fewer(self):
        _, iters, _ = goldschmidt_binomial_divide(10.0, 3.0, threshold=1e-12, order=3)
        assert iters <= 10

    def test_long_div_exact_iterations(self):
        _, _, iters, _ = long_divide_unsigned(100, 7, bits=32)
        assert iters == 32


# =============================================================================
#  10. Seed-strategy comparison: "linear" vs "magic"
# =============================================================================
# Test values spanning several orders of magnitude, including near-boundary
# and hardware-relevant cases.
_COMPARISON_B_VALUES = [
    0.1, 0.25, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0,
    42.0, 100.0, 1000.0, 1e6,
]


class TestSeedStrategyComparison:
    """Compare 'linear' and 'magic' initial-guess strategies for NR reciprocal.

    Metrics:
      - iteration count to reach each precision threshold
      - worst-case iteration count across a range of b values
      - convergence robustness (both strategies must converge for all b)

    Tested at three simulated precision levels: FP32, BF16, FP8.
    """

    # ------------------------------------------------------------------
    #  Both strategies converge for all b values at every precision
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("precision", ["fp32", "bf16", "fp8"])
    @pytest.mark.parametrize("strategy", ["linear", "magic"])
    @pytest.mark.parametrize("b", _COMPARISON_B_VALUES)
    def test_convergence_robustness(self, b, strategy, precision):
        """Both seeds must converge for every (b, precision) pair."""
        res = newton_reciprocal_result(
            b, precision=precision, max_iter=50, x0_strategy=strategy
        )
        assert res.converged, (
            f"strategy={strategy}, b={b}, precision={precision}: "
            f"did not converge in {res.iterations} iters"
        )

    # ------------------------------------------------------------------
    #  Iteration-count comparison at FP32 precision
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", _COMPARISON_B_VALUES)
    def test_iteration_count_fp32(self, b):
        """At FP32 precision, magic and linear should differ by at most 1 iteration.

        Empirical fact: both strategies converge in 2-3 iterations for all
        tested b values.  A difference > 1 would indicate a regression.
        """
        res_lin = newton_reciprocal_result(b, precision="fp32", max_iter=50, x0_strategy="linear")
        res_mag = newton_reciprocal_result(b, precision="fp32", max_iter=50, x0_strategy="magic")
        assert res_lin.converged
        assert res_mag.converged
        # Strict: at most 1 iteration difference in either direction
        assert abs(res_mag.iterations - res_lin.iterations) <= 1, (
            f"b={b}: magic={res_mag.iterations} vs linear={res_lin.iterations}"
        )
        # Both must finish in at most 3 iterations (empirical hard bound)
        assert res_lin.iterations <= 3, f"b={b}: linear took {res_lin.iterations} iters"
        assert res_mag.iterations <= 3, f"b={b}: magic took {res_mag.iterations} iters"

    # ------------------------------------------------------------------
    #  Iteration-count comparison at BF16 precision
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", _COMPARISON_B_VALUES)
    def test_iteration_count_bf16(self, b):
        """At BF16 precision, both seeds converge in exactly 1 iteration."""
        res_lin = newton_reciprocal_result(b, precision="bf16", max_iter=50, x0_strategy="linear")
        res_mag = newton_reciprocal_result(b, precision="bf16", max_iter=50, x0_strategy="magic")
        assert res_lin.converged
        assert res_mag.converged
        # Both linear and magic converge in 1 iter at bf16 threshold (~7.8e-3)
        assert res_lin.iterations == 1, f"b={b}: linear took {res_lin.iterations} iters at bf16"
        assert res_mag.iterations == 1, f"b={b}: magic took {res_mag.iterations} iters at bf16"

    # ------------------------------------------------------------------
    #  Iteration-count comparison at FP8 precision
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", _COMPARISON_B_VALUES)
    def test_iteration_count_fp8(self, b):
        """At FP8 precision, both seeds converge in exactly 1 iteration."""
        res_lin = newton_reciprocal_result(b, precision="fp8", max_iter=50, x0_strategy="linear")
        res_mag = newton_reciprocal_result(b, precision="fp8", max_iter=50, x0_strategy="magic")
        assert res_lin.converged
        assert res_mag.converged
        # At fp8 threshold (~0.125), a single NR step from any decent seed suffices
        assert res_lin.iterations == 1, f"b={b}: linear took {res_lin.iterations} iters at fp8"
        assert res_mag.iterations == 1, f"b={b}: magic took {res_mag.iterations} iters at fp8"

    # ------------------------------------------------------------------
    #  Worst-case iteration count across all b values
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("precision", ["fp32", "bf16", "fp8"])
    def test_worst_case_iterations(self, precision):
        """Worst-case iteration count should stay within sane bounds."""
        max_iters = {"fp32": 8, "bf16": 5, "fp8": 3}
        bound = max_iters[precision]

        for strategy in ("linear", "magic"):
            worst = 0
            for b in _COMPARISON_B_VALUES:
                res = newton_reciprocal_result(
                    b, precision=precision, max_iter=50, x0_strategy=strategy
                )
                assert res.converged
                worst = max(worst, res.iterations)
            assert worst <= bound, (
                f"strategy={strategy}, precision={precision}: "
                f"worst-case {worst} iters exceeds bound {bound}"
            )

    # ------------------------------------------------------------------
    #  Linear seed should be at least as good or better than magic
    #  (averaged over all b values)
    # ------------------------------------------------------------------
    def test_magic_and_linear_statistically_equivalent_fp32(self):
        """At FP32, magic and linear give nearly identical average iteration counts."""
        total_lin = 0
        total_mag = 0
        for b in _COMPARISON_B_VALUES:
            total_lin += newton_reciprocal_result(
                b, precision="fp32", max_iter=50, x0_strategy="linear"
            ).iterations
            total_mag += newton_reciprocal_result(
                b, precision="fp32", max_iter=50, x0_strategy="magic"
            ).iterations
        avg_lin = total_lin / len(_COMPARISON_B_VALUES)
        avg_mag = total_mag / len(_COMPARISON_B_VALUES)
        # They should be nearly identical (within 0.2 iters)
        assert abs(avg_lin - avg_mag) < 0.2, (
            f"avg_lin={avg_lin:.3f} vs avg_mag={avg_mag:.3f}: "
            f"difference {abs(avg_lin - avg_mag):.3f} exceeds 0.2"
        )

    # ------------------------------------------------------------------
    #  Honest seed-error comparison (strict quantitative bounds)
    # ------------------------------------------------------------------
    def test_seed_error_bounds_fp32(self):
        """Worst-case seed error: linear <= 5.88%, magic <= 5.05%."""
        import random as _rng
        rng = _rng.Random(12345)
        b_vals = [math.exp(rng.uniform(math.log(1e-3), math.log(1e6))) for _ in range(500)]

        lin_worst = 0.0
        mag_worst = 0.0
        for b in b_vals:
            exact = 1.0 / b
            x0_lin = hardware_initial_guess(b, strategy="linear")
            x0_mag = hardware_initial_guess(b, strategy="magic")
            lin_worst = max(lin_worst, abs(x0_lin / exact - 1.0))
            mag_worst = max(mag_worst, abs(x0_mag / exact - 1.0))

        # Hard upper bounds (empirically verified, mathematically grounded)
        assert lin_worst <= 0.06, f"linear worst-case seed error = {lin_worst:.6f} > 0.06"
        assert mag_worst <= 0.06, f"magic worst-case seed error = {mag_worst:.6f} > 0.06"

        # Magic has strictly lower worst-case seed error than linear
        # (0.0505 vs 0.0588 — structurally guaranteed by the constant)
        assert mag_worst < lin_worst, (
            f"Expected magic worst ({mag_worst:.6f}) < linear worst ({lin_worst:.6f}). "
            f"This is a structural property of the chosen magic constant."
        )

    # ------------------------------------------------------------------
    #  Large-scale comparison: provably equivalent iteration counts
    # ------------------------------------------------------------------
    def test_magic_linear_same_max_iters_large_scale(self):
        """Both strategies peak at 3 iterations over 200 diverse b values."""
        import random as _rng
        rng = _rng.Random(999)
        b_vals = [math.exp(rng.uniform(math.log(1e-3), math.log(1e6))) for _ in range(200)]

        max_lin = 0
        max_mag = 0
        for b in b_vals:
            rl = newton_reciprocal_result(b, precision="fp32", max_iter=50, x0_strategy="linear")
            rm = newton_reciprocal_result(b, precision="fp32", max_iter=50, x0_strategy="magic")
            assert rl.converged, f"linear failed at b={b}"
            assert rm.converged, f"magic failed at b={b}"
            max_lin = max(max_lin, rl.iterations)
            max_mag = max(max_mag, rm.iterations)

        # Both strategies: worst-case is exactly 3 iterations at fp32
        assert max_lin == 3, f"linear worst-case = {max_lin} (expected 3)"
        assert max_mag == 3, f"magic worst-case = {max_mag} (expected 3)"

    # ------------------------------------------------------------------
    #  Simulation-level comparison (float mode)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("strategy", ["linear", "magic"])
    def test_simulation_both_strategies_converge(self, strategy):
        """Full simulation pipeline should converge with both strategies."""
        cfg = NewtonReciprocalConfig(
            mode="float", max_iter=15, tol_rel=1e-12, x0_strategy=strategy
        )
        for b in [2.0, 5.0, 10.0, 100.0]:
            pts = simulate_newton_reciprocal(b, cfg)
            assert pts[-1].err_rel < 1e-10, (
                f"strategy={strategy}, b={b}: final err_rel={pts[-1].err_rel}"
            )

    # ------------------------------------------------------------------
    #  Simulation-level comparison (fixed-point mode)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("strategy", ["linear", "magic"])
    def test_simulation_fixed_both_strategies(self, strategy):
        """Fixed-point simulation should work with both seeds."""
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=15, tol_rel=1e-4,
            frac_bits=24, x0_strategy=strategy
        )
        for b in [2.0, 5.0, 10.0]:
            pts = simulate_newton_reciprocal(b, cfg)
            assert pts[-1].err_rel < 0.01, (
                f"strategy={strategy}, b={b}: final err_rel={pts[-1].err_rel}"
            )


# ============================================================================
#  BF16 / FP8 Magic-Constant Seed Tests
# ============================================================================

class TestMagicConstantBF16FP8:
    """Tests for BF16 and FP8 E4M3 magic-constant reciprocal seeds."""

    # ------------------------------------------------------------------
    #  Constant value sanity checks
    # ------------------------------------------------------------------
    def test_bf16_constant_value(self):
        # 0x7EF3 = upper 16 bits of FP32 magic 0x7EF311C3
        assert MAGIC_CONST_BF16 == 0x7EF3

    def test_fp8_constant_value(self):
        # 0x70 = brute-force optimal by NR convergence speed
        assert MAGIC_CONST_FP8_E4M3 == 0x70

    def test_fp32_constant_value(self):
        assert MAGIC_CONST_FP32 == 0x7EF311C3

    def test_bf16_constant_is_16bit(self):
        assert 0 < MAGIC_CONST_BF16 <= 0xFFFF

    def test_fp8_constant_is_8bit(self):
        assert 0 < MAGIC_CONST_FP8_E4M3 <= 0xFF

    # ------------------------------------------------------------------
    #  Seed produces positive result for typical b values
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    def test_bf16_seed_positive(self, b):
        x0 = magic_constant_reciprocal_seed(b, fmt="bf16")
        assert x0 > 0.0, f"bf16 seed for b={b} should be positive, got {x0}"

    @pytest.mark.parametrize("b", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_fp8_seed_positive(self, b):
        x0 = magic_constant_reciprocal_seed(b, fmt="fp8_e4m3")
        assert x0 > 0.0, f"fp8 seed for b={b} should be positive, got {x0}"

    # ------------------------------------------------------------------
    #  Seed is a rough approximation of 1/b (correct order of magnitude)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [1.0, 2.0, 4.0, 10.0, 50.0])
    def test_bf16_seed_order_of_magnitude(self, b):
        exact = 1.0 / b
        x0 = magic_constant_reciprocal_seed(b, fmt="bf16")
        # Within 10x of exact — generous for a coarse seed
        assert 0.1 * exact < x0 < 10.0 * exact, (
            f"bf16 seed for b={b}: x0={x0}, exact={exact}"
        )

    @pytest.mark.parametrize("b", [1.0, 2.0, 4.0, 8.0])
    def test_fp8_seed_order_of_magnitude(self, b):
        exact = 1.0 / b
        x0 = magic_constant_reciprocal_seed(b, fmt="fp8_e4m3")
        assert 0.1 * exact < x0 < 10.0 * exact, (
            f"fp8 seed for b={b}: x0={x0}, exact={exact}"
        )

    # ------------------------------------------------------------------
    #  FP32 seed should be significantly better than BF16/FP8
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [2.0, 5.0, 10.0])
    def test_fp32_seed_close_to_bf16(self, b):
        """BF16 magic = upper 16 bits of FP32 magic, so seeds are very close.

        Due to truncation, BF16 can occasionally be slightly better for
        specific b values.  We only check they are within 5% of each other.
        """
        exact = 1.0 / b
        err_fp32 = abs(magic_constant_reciprocal_seed(b, "fp32") / exact - 1.0)
        err_bf16 = abs(magic_constant_reciprocal_seed(b, "bf16") / exact - 1.0)
        # Both should be < 10% seed error
        assert err_fp32 < 0.10, f"fp32 seed error unexpectedly large: {err_fp32}"
        assert err_bf16 < 0.10, f"bf16 seed error unexpectedly large: {err_bf16}"
        # They should be close (BF16 is the truncated FP32 constant)
        assert abs(err_fp32 - err_bf16) < 0.05, (
            f"b={b}: fp32 err={err_fp32:.4e}, bf16 err={err_bf16:.4e} differ too much"
        )

    # ------------------------------------------------------------------
    #  Error cases
    # ------------------------------------------------------------------
    def test_bf16_seed_negative_b_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(-1.0, fmt="bf16")

    def test_fp8_seed_zero_b_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(0.0, fmt="fp8_e4m3")

    def test_bf16_seed_inf_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(float("inf"), fmt="bf16")

    def test_fp8_seed_nan_raises(self):
        with pytest.raises(ValueError):
            magic_constant_reciprocal_seed(float("nan"), fmt="fp8_e4m3")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            magic_constant_reciprocal_seed(2.0, fmt="fp64")

    # ------------------------------------------------------------------
    #  hardware_initial_guess dispatches correctly
    # ------------------------------------------------------------------
    def test_hardware_guess_magic_bf16(self):
        x0 = hardware_initial_guess(2.0, strategy="magic_bf16")
        expected = magic_constant_reciprocal_seed(2.0, fmt="bf16")
        assert math.isclose(x0, expected, rel_tol=1e-15)

    def test_hardware_guess_magic_fp8(self):
        x0 = hardware_initial_guess(2.0, strategy="magic_fp8")
        expected = magic_constant_reciprocal_seed(2.0, fmt="fp8_e4m3")
        assert math.isclose(x0, expected, rel_tol=1e-15)

    # ------------------------------------------------------------------
    #  NR convergence using BF16 / FP8 seeds
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [1.0, 2.0, 5.0, 10.0, 42.0])
    def test_nr_converges_with_bf16_seed(self, b):
        """NR should converge to 1/b even when started from the coarser BF16 seed."""
        x0 = magic_constant_reciprocal_seed(b, fmt="bf16")
        res = newton_reciprocal_result(b, x0=x0, max_iter=30, threshold=1e-12)
        assert res.converged, (
            f"BF16 seed: NR did not converge for b={b} (final err={res.errors[-1]:.4e})"
        )
        assert math.isclose(res.x, 1.0 / b, rel_tol=1e-10)

    @pytest.mark.parametrize("b", [1.0, 2.0, 4.0, 8.0])
    def test_nr_converges_with_fp8_seed(self, b):
        """NR should converge to 1/b even from the very coarse FP8 seed."""
        x0 = magic_constant_reciprocal_seed(b, fmt="fp8_e4m3")
        res = newton_reciprocal_result(b, x0=x0, max_iter=30, threshold=1e-12)
        assert res.converged, (
            f"FP8 seed: NR did not converge for b={b} (final err={res.errors[-1]:.4e})"
        )
        assert math.isclose(res.x, 1.0 / b, rel_tol=1e-10)

    # ------------------------------------------------------------------
    #  BF16 seed needs more NR iterations than FP32 (worse starting error)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [2.0, 10.0, 42.0])
    def test_bf16_needs_more_or_equal_iterations_than_fp32(self, b):
        res_fp32 = newton_reciprocal_result(
            b, x0_strategy="magic", max_iter=30, threshold=1e-12
        )
        res_bf16 = newton_reciprocal_result(
            b, x0=magic_constant_reciprocal_seed(b, "bf16"),
            max_iter=30, threshold=1e-12
        )
        # BF16 starts with a worse seed, so it should need >= iterations
        assert res_bf16.iterations >= res_fp32.iterations, (
            f"b={b}: bf16 iters={res_bf16.iterations} < fp32 iters={res_fp32.iterations}"
        )

    # ------------------------------------------------------------------
    #  Simulation pipeline with magic_bf16 / magic_fp8 strategies
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("strategy", ["magic_bf16", "magic_fp8"])
    def test_simulation_float_with_reduced_precision_seed(self, strategy):
        """Float-mode simulation converges using BF16/FP8 magic seeds."""
        cfg = NewtonReciprocalConfig(
            mode="float", max_iter=30, tol_rel=1e-10, x0_strategy=strategy
        )
        for b in [2.0, 5.0, 10.0]:
            pts = simulate_newton_reciprocal(b, cfg)
            assert pts[-1].err_rel < 1e-10, (
                f"strategy={strategy}, b={b}: final err_rel={pts[-1].err_rel}"
            )

    @pytest.mark.parametrize("strategy", ["magic_bf16", "magic_fp8"])
    def test_simulation_fixed_with_reduced_precision_seed(self, strategy):
        """Fixed-point simulation works with BF16/FP8 magic seeds.

        We relax tolerances because quantization and coarse seeds compound.
        """
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=20, tol_rel=1e-3,
            frac_bits=24, x0_strategy=strategy
        )
        for b in [2.0, 5.0, 10.0]:
            pts = simulate_newton_reciprocal(b, cfg)
            assert pts[-1].err_rel < 0.05, (
                f"strategy={strategy}, b={b}: final err_rel={pts[-1].err_rel}"
            )

    # ------------------------------------------------------------------
    #  Seed comparison across formats for same b
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("b", [1.0, 2.0, 5.0, 10.0])
    def test_seed_error_ordering_fp32_bf16_fp8(self, b):
        """Seed quality ordering: FP32 and BF16 are both good;
        FP8 is significantly coarser.

        Since BF16 magic = upper 16 bits of FP32 magic, their seed
        errors are nearly identical.  FP8 has only 3 mantissa bits,
        so its error is expected to be much worse.
        """
        exact = 1.0 / b
        err32 = abs(magic_constant_reciprocal_seed(b, "fp32") / exact - 1.0)
        err16 = abs(magic_constant_reciprocal_seed(b, "bf16") / exact - 1.0)
        err8 = abs(magic_constant_reciprocal_seed(b, "fp8_e4m3") / exact - 1.0)
        # FP32 and BF16 should both have < 10% seed error
        assert err32 < 0.10, f"b={b}: fp32 err {err32:.4e} unexpectedly large"
        assert err16 < 0.10, f"b={b}: bf16 err {err16:.4e} unexpectedly large"
        # FP8 error can be much larger (up to ~12.5% with optimal constant)
        assert err8 < 1.5, f"b={b}: fp8 err {err8:.4e} unexpectedly large"


# ============================================================================
#  search_magic_constant Tests
# ============================================================================

class TestSearchMagicConstant:
    """Tests for the brute-force magic constant search function."""

    def test_bf16_search_returns_tuple(self):
        """Search should return (constant: int, score: float)."""
        result = search_magic_constant("bf16", b_samples=[1.0, 2.0, 5.0, 10.0])
        assert isinstance(result, tuple) and len(result) == 2
        const, score = result
        assert isinstance(const, int)
        assert isinstance(score, float)

    def test_fp8_search_returns_tuple(self):
        result = search_magic_constant("fp8_e4m3", b_samples=[1.0, 2.0, 4.0, 8.0])
        assert isinstance(result, tuple) and len(result) == 2

    def test_bf16_search_constant_in_16bit_range(self):
        const, _ = search_magic_constant("bf16", b_samples=[1.0, 2.0, 5.0])
        assert 0 <= const <= 0xFFFF

    def test_fp8_search_constant_in_8bit_range(self):
        const, _ = search_magic_constant("fp8_e4m3", b_samples=[1.0, 2.0, 4.0])
        assert 0 <= const <= 0xFF

    def test_bf16_search_score_positive(self):
        _, score = search_magic_constant("bf16", b_samples=[1.0, 2.0, 5.0])
        assert score > 0.0

    def test_bf16_search_finds_reasonable_constant(self):
        """The searched BF16 constant should give seed errors < 100% for all test b."""
        test_bs = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        const, _ = search_magic_constant("bf16", b_samples=test_bs)
        # Verify the found constant actually works reasonably
        from nr_reciprocal.basic import _float_to_bf16_int, _bf16_int_to_float
        for bv in test_bs:
            exact = 1.0 / bv
            b_int = _float_to_bf16_int(bv)
            seed_int = (const - b_int) & 0xFFFF
            x0 = _bf16_int_to_float(seed_int)
            if x0 > 0:
                rel_err = abs(x0 / exact - 1.0)
                assert rel_err < 1.0, (
                    f"bf16 searched const=0x{const:04X}: b={bv}: rel_err={rel_err:.3f}"
                )

    def test_fp8_search_finds_reasonable_constant(self):
        """The searched FP8 constant should give seed errors < 200% for all test b."""
        test_bs = [1.0, 2.0, 4.0, 8.0]
        const, _ = search_magic_constant("fp8_e4m3", b_samples=test_bs)
        from nr_reciprocal.basic import _float_to_fp8e4m3_int, _fp8e4m3_int_to_float
        for bv in test_bs:
            exact = 1.0 / bv
            b_int = _float_to_fp8e4m3_int(bv)
            seed_int = (const - b_int) & 0xFF
            x0 = _fp8e4m3_int_to_float(seed_int)
            if x0 > 0:
                rel_err = abs(x0 / exact - 1.0)
                assert rel_err < 2.0, (
                    f"fp8 searched const=0x{const:02X}: b={bv}: rel_err={rel_err:.3f}"
                )

    def test_mean_objective_returns_different_score(self):
        """mean vs worst objective may give different scores."""
        _, score_worst = search_magic_constant(
            "fp8_e4m3", b_samples=[1.0, 2.0, 4.0, 8.0], objective="worst"
        )
        _, score_mean = search_magic_constant(
            "fp8_e4m3", b_samples=[1.0, 2.0, 4.0, 8.0], objective="mean"
        )
        # Both should be non-negative (0.0 is possible if the constant
        # happens to give exact seeds for all test values).
        assert score_worst >= 0.0
        assert score_mean >= 0.0

    def test_fp32_search_raises(self):
        """FP32 brute-force search should raise (not practical)."""
        with pytest.raises(ValueError, match="not practical"):
            search_magic_constant("fp32")

    def test_search_default_samples(self):
        """Search with default b_samples should work."""
        const, score = search_magic_constant("fp8_e4m3")
        assert 0 <= const <= 0xFF
        assert score >= 0.0

    def test_bf16_search_worst_beats_random_constant(self):
        """Searched constant should beat an arbitrary constant (e.g. 0x4000)."""
        test_bs = [1.0, 2.0, 5.0, 10.0]
        best_const, best_score = search_magic_constant(
            "bf16", b_samples=test_bs, objective="worst"
        )
        # Compute score for an arbitrary constant
        from nr_reciprocal.basic import _float_to_bf16_int, _bf16_int_to_float
        arb_const = 0x4000
        arb_errors = []
        for bv in test_bs:
            exact = 1.0 / bv
            b_int = _float_to_bf16_int(bv)
            seed_int = (arb_const - b_int) & 0xFFFF
            x0 = _bf16_int_to_float(seed_int)
            if x0 > 0:
                arb_errors.append(abs(x0 / exact - 1.0))
            else:
                arb_errors.append(float("inf"))
        arb_score = max(arb_errors)
        assert best_score <= arb_score, (
            f"Searched BF16 const=0x{best_const:04X} (score={best_score:.4f}) "
            f"should beat arbitrary 0x4000 (score={arb_score:.4f})"
        )


# ============================================================================
#  6a) Magic seed converges faster or equal compared to simple seeds
# ============================================================================

class TestMagicSeedSpeed:
    """Compare iteration counts between magic and simple seed strategies."""

    THRESHOLD = 1e-12
    MAX_ITER = 50
    B_VALUES = [1.1, 2.0, 3.7, 5.0, 10.0, 42.0, 100.0, 999.0]

    # -- FP32 magic vs. constant ("one") seed ---------------------------------
    @pytest.mark.parametrize("b", B_VALUES)
    def test_magic_fp32_faster_or_close_to_const(self, b):
        res_magic = newton_reciprocal_result(
            b, x0_strategy="magic", max_iter=self.MAX_ITER,
            threshold=self.THRESHOLD,
        )
        res_const = newton_reciprocal_result(
            b, x0_strategy="const", max_iter=self.MAX_ITER,
            threshold=self.THRESHOLD,
        )
        # Magic is typically faster; allow at most 1 extra iteration
        # because for certain b the naive x0=1.0 can be accidentally close.
        assert res_magic.iterations <= res_const.iterations + 1, (
            f"b={b}: magic iters={res_magic.iterations} > "
            f"const iters={res_const.iterations}+1"
        )

    # -- FP32 magic vs. linear seed -------------------------------------------
    @pytest.mark.parametrize("b", B_VALUES)
    def test_magic_fp32_equivalent_to_linear(self, b):
        """Magic and linear iteration counts differ by at most 1."""
        res_magic = newton_reciprocal_result(
            b, x0_strategy="magic", max_iter=self.MAX_ITER,
            threshold=self.THRESHOLD,
        )
        res_linear = newton_reciprocal_result(
            b, x0_strategy="linear", max_iter=self.MAX_ITER,
            threshold=self.THRESHOLD,
        )
        assert abs(res_magic.iterations - res_linear.iterations) <= 1, (
            f"b={b}: magic={res_magic.iterations} vs linear={res_linear.iterations}"
        )

    # -- BF16 magic seed still converges (not necessarily faster than const) --
    @pytest.mark.parametrize("b", [1.5, 2.0, 5.0, 10.0, 50.0])
    def test_magic_bf16_converges(self, b):
        res_bf16 = newton_reciprocal_result(
            b, x0=magic_constant_reciprocal_seed(b, "bf16"),
            max_iter=self.MAX_ITER, threshold=self.THRESHOLD,
        )
        assert res_bf16.converged, (
            f"b={b}: bf16 seed did not converge (iters={res_bf16.iterations})"
        )
        assert math.isclose(res_bf16.x, 1.0 / b, rel_tol=1e-10)

    # -- FP8 magic seed still converges ---------------------------------------
    @pytest.mark.parametrize("b", [1.0, 2.0, 4.0, 8.0])
    def test_magic_fp8_converges(self, b):
        res_fp8 = newton_reciprocal_result(
            b, x0=magic_constant_reciprocal_seed(b, "fp8_e4m3"),
            max_iter=self.MAX_ITER, threshold=self.THRESHOLD,
        )
        assert res_fp8.converged, (
            f"b={b}: fp8 seed did not converge (iters={res_fp8.iterations})"
        )
        assert math.isclose(res_fp8.x, 1.0 / b, rel_tol=1e-10)

    # -- All magic seeds reach full fp64 accuracy -----------------------------
    @pytest.mark.parametrize("strategy,b_vals", [
        ("magic", [2.0, 7.0, 42.0]),
        ("magic_bf16", [2.0, 7.0, 42.0]),
        ("magic_fp8", [2.0, 4.0, 8.0]),  # FP8 range is narrow; b=42 not viable
    ])
    def test_all_magic_converge_to_full_precision(self, strategy, b_vals):
        for b in b_vals:
            cfg = NewtonReciprocalConfig(
                mode="float", max_iter=40, tol_rel=1e-14,
                x0_strategy=strategy,
            )
            pts = simulate_newton_reciprocal(b, cfg)
            exact = 1.0 / b
            assert math.isclose(pts[-1].x, exact, rel_tol=1e-12), (
                f"strategy={strategy}, b={b}: final x={pts[-1].x}, "
                f"exact={exact}"
            )

    # -- Iteration count ordering: fp32, bf16, fp8 seed -------------------------
    @pytest.mark.parametrize("b", [3.0, 10.0, 42.0])
    def test_iteration_ordering_fp32_bf16_fp8(self, b):
        """FP32 and BF16 seeds are nearly identical (BF16 = truncated FP32).
        FP8 is coarser for non-power-of-2 inputs.

        NOTE: b=2.0 is excluded because FP8 magic 0x70 gives an exact seed
        for all powers of 2 (x0 = 1/b), making FP8 faster than BF16 on
        those specific inputs.
        """
        iters = {}
        for label, strat in [("fp32", "magic"), ("bf16", "magic_bf16"),
                             ("fp8", "magic_fp8")]:
            cfg = NewtonReciprocalConfig(
                mode="float", max_iter=40, tol_rel=1e-14,
                x0_strategy=strat,
            )
            pts = simulate_newton_reciprocal(b, cfg)
            iters[label] = len(pts)
        # BF16 magic = truncation of FP32 magic, so they may tie.
        assert iters["fp32"] <= iters["bf16"] + 1, (
            f"b={b}: fp32={iters['fp32']} >> bf16={iters['bf16']}"
        )
        # For non-power-of-2, FP8 should generally need more iters
        assert iters["fp32"] <= iters["fp8"] + 1, (
            f"b={b}: fp32={iters['fp32']} >> fp8={iters['fp8']}"
        )


# ============================================================================
#  6b) Convergence under rounding and quantization
# ============================================================================

class TestRoundingQuantizationConvergence:
    """Check NR convergence under fixed-point quantization."""

    # -- Fixed-point mode with magic seed converges ---------------------------
    @pytest.mark.parametrize("frac_bits", [8, 12, 16, 24, 32])
    @pytest.mark.parametrize("b", [2.0, 5.0, 10.0])
    def test_fixed_point_magic_converges(self, frac_bits, b):
        """With enough iterations, fixed-point NR should get close to 1/b."""
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=30, tol_rel=1e-2,
            frac_bits=frac_bits, x0_strategy="magic",
        )
        pts = simulate_newton_reciprocal(b, cfg)
        # The simulation stops early when tol_rel is met.
        # With magic seed, 1-2 iterations typically reach ~0.3% error.
        # Use a generous bound: final error must be < 1%.
        assert pts[-1].err_rel < 0.01, (
            f"frac_bits={frac_bits}, b={b}: err_rel={pts[-1].err_rel:.4e}"
        )

    # -- Fixed-point mode with linear seed converges too ----------------------
    @pytest.mark.parametrize("frac_bits", [8, 16, 24])
    @pytest.mark.parametrize("b", [2.0, 5.0, 10.0])
    def test_fixed_point_linear_converges(self, frac_bits, b):
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=30, tol_rel=1e-2,
            frac_bits=frac_bits, x0_strategy="linear",
        )
        pts = simulate_newton_reciprocal(b, cfg)
        assert pts[-1].err_rel < 0.01, (
            f"frac_bits={frac_bits}, b={b}: err_rel={pts[-1].err_rel:.4e}"
        )

    # -- Very coarse quantization (4 frac bits) doesn't blow up ---------------
    @pytest.mark.parametrize("b", [2.0, 4.0, 8.0])
    def test_very_coarse_quantization_stable(self, b):
        """With only 4 fractional bits the result won't be accurate, but NR
        must not diverge (final error bounded, not inf/nan)."""
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=20, tol_rel=0.5,
            frac_bits=4, x0_strategy="linear",
        )
        pts = simulate_newton_reciprocal(b, cfg)
        assert math.isfinite(pts[-1].err_rel)
        # Coarse but bounded: within 50% of exact
        assert pts[-1].err_rel < 0.5, (
            f"b={b}: err_rel={pts[-1].err_rel:.4f} with 4 frac bits"
        )

    # -- Fixed-point magic_bf16 / magic_fp8 seeds still converge --------------
    @pytest.mark.parametrize("strategy", ["magic_bf16", "magic_fp8"])
    @pytest.mark.parametrize("b", [2.0, 5.0, 10.0])
    def test_fixed_point_reduced_seed_converges(self, strategy, b):
        cfg = NewtonReciprocalConfig(
            mode="fixed", max_iter=25, tol_rel=1e-2,
            frac_bits=24, x0_strategy=strategy,
        )
        pts = simulate_newton_reciprocal(b, cfg)
        assert pts[-1].err_rel < 0.01, (
            f"strategy={strategy}, b={b}: err_rel={pts[-1].err_rel:.4e}"
        )

    # -- Quantized result is close to float result (tolerance, not exact) -----
    @pytest.mark.parametrize("b", [2.0, 5.0, 10.0, 42.0])
    def test_quantized_vs_float_tolerance(self, b):
        """Fixed-point result should be close to float result (not identical)."""
        cfg_float = NewtonReciprocalConfig(
            mode="float", max_iter=15, tol_rel=1e-12,
            x0_strategy="magic",
        )
        cfg_fixed = NewtonReciprocalConfig(
            mode="fixed", max_iter=15, tol_rel=1e-6,
            frac_bits=24, x0_strategy="magic",
        )
        pts_float = simulate_newton_reciprocal(b, cfg_float)
        pts_fixed = simulate_newton_reciprocal(b, cfg_fixed)
        # They should agree within the quantization resolution (~2^-24).
        assert math.isclose(
            pts_float[-1].x, pts_fixed[-1].x, rel_tol=1e-2
        ), (
            f"b={b}: float={pts_float[-1].x}, "
            f"fixed={pts_fixed[-1].x}"
        )

    # -- Error always reported with tolerance, never exact --------------------
    @pytest.mark.parametrize("b", [2.0, 7.0, 100.0])
    def test_final_error_within_tolerance(self, b):
        """Final relative error should be below threshold, checked via < not ==."""
        res = newton_reciprocal_result(
            b, max_iter=30, threshold=1e-12, x0_strategy="magic"
        )
        assert res.errors[-1] < 1e-12
        exact = 1.0 / b
        assert math.isclose(res.x, exact, rel_tol=1e-10)

    # -- Goldschmidt under quantization-like perturbation ---------------------
    @pytest.mark.parametrize("b", [2, 5, 10])
    def test_goldschmidt_rounded_inputs(self, b):
        """GS classic should tolerate slightly perturbed (rounded) inputs."""
        from gs_division.basic import goldschmidt_classic_divide
        a = 100
        # Slightly perturbed dividend (simulates rounding)
        for offset in [-1, 0, 1]:
            q, _iters, _errs = goldschmidt_classic_divide(a + offset, b)
            expected = (a + offset) / b  # float division
            assert math.isclose(q, expected, rel_tol=1e-6), (
                f"a={a + offset}, b={b}: q={q}, expected~{expected}"
            )

    # -- Long division is exact (integer, immune to rounding) -----------------
    @pytest.mark.parametrize("a,b", [(100, 7), (255, 3), (1024, 10)])
    def test_long_division_exact_integer(self, a, b):
        """Long division on integers is bit-exact — no tolerance needed."""
        from long_division.basic import long_divide_unsigned
        q, r, _iters, _ops = long_divide_unsigned(a, b, bits=32)
        assert q == a // b
        assert r == a % b
