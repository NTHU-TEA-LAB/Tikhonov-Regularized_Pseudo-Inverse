"""
Software model for `RTF_top` to verify hardware results.

All arithmetic is done with explicit fixed bit widths and two's‑complement
wrapping, to mimic the Verilog implementation as closely as possible.

Simplification for now:
    - Treat divider output as pure integer: inv_det = 1 / det (truncated)
      without fractional bits.
    - If det == 0, skip division and keep inv_det = 0.

Input:
    Prefer `RTF_input_from_hardware.txt`: each line
        save_exp_out_addr= N  save_exp_out_real= R  save_exp_out_imag= I
    gives one sample (real R, imag I) at index N-1. Fallback: `RTF_input_Real.txt`
    and `RTF_input_Imag.txt` (one integer per line). If no file found, testbench
    pattern: bram_rd_mem_real[i] = (i % 1000) - 500, bram_rd_mem_imag[i] = ((i+1) % 1000) - 500
    where i = freq * (MIC_NUM * SOR_NUM) + sor * MIC_NUM + mic.
"""

from __future__ import annotations
import re
from datetime import datetime
import numpy as np

# ---------------------------------------------------------------------------
# Parameters (mirror RTF_top.v - match hardware bit widths for testing)
# ---------------------------------------------------------------------------

MIC_NUM = 8
SOR_NUM = 2
FREQ_NUM = 257

DATA_WIDTH = 16
ACC_WIDTH = 33                   
INV_G_WIDTH = 104                 
DET_WIDTH = 66                    
INPUT_FRAC_BITS = 14
INPUT_Q_FORMAT = "s2.14"
DET_FRAC_BITS = 4 * INPUT_FRAC_BITS

# Divider configuration (match RTF_top.v)
DIVOUT_TDATA_WIDTH = 72
DIVOUT_F_WIDTH = 24  # match RTL: Vivado divider frac MSB is sign (1 sign + 23 value bits)
DIVOUT_GUARD_BITS = 0
DIVIDEND_TDATA_WIDTH = 48
DIVISOR_TDATA_WIDTH = 64
# Vivado Divider Generator (Radix-2 / Fixed-point fractional) truncates the fractional LSB (see doc Eq. 3-2/3-3:
# fractional output from integer remainder, no rounding). So we use truncation (floor) for the quotient.
# Set True only if you need round-to-nearest for comparison with a different IP.
DIVIDER_QUOTIENT_ROUND_TO_NEAREST = False

# Regularization lambda in same scale as G (Q4.28 when INPUT_FRAC_BITS=14).
# 0.01 in Q4.28 = round(0.01 * 2^(2*INPUT_FRAC_BITS)) = 2,684,355.
LAMBDA = 2684355

# Float comparison (set False to skip)
ENABLE_FLOAT_COMPARE = True

# Auto-tune divider and internal widths based on float error
AUTO_TUNE_DIVOUT_F_WIDTH = False
AUTO_TUNE_INTERNAL_WIDTHS = False
# Allow larger error -> smaller bit widths (e.g. 5e-2 = 5%, 1e-1 = 10%)
ERR_TRIGGER = 5e-2
ERR_TARGET  = 5e-2
DIVOUT_F_WIDTH_MAX = 63
DIVOUT_TDATA_WIDTH_MAX = 96
DIVOUT_GUARD_BITS_MAX = 8
ACC_WIDTH_MAX = 80
DET_WIDTH_MAX = 128
INV_G_WIDTH_MAX = 160

# Fixed det_shift for experiment: if set, use FIXED_DET_SHIFT instead of dynamic det_shift
USE_FIXED_DET_SHIFT = False
FIXED_DET_SHIFT = 48

# Shrink widths while keeping max_err <= ERR_TARGET
SHRINK_WIDTHS = False
# Minimum output width (INV_G_WIDTH) when shrinking; set > 1 to cap output bits (e.g. 64 or 48)
INV_G_WIDTH_FLOOR = 48
# Dump detailed info for the worst-case error
DUMP_MAX_ERROR_DETAIL = True
# Number of top errors to list (e.g. in Shrunk widths skipped / Max error vs float)
TOP_N_ERRORS_TO_PRINT = 50

PER_FREQ = MIC_NUM * SOR_NUM
TOTAL_NUM = MIC_NUM * SOR_NUM * FREQ_NUM

# Primary: single file from hardware (each line: save_exp_out_addr= N  save_exp_out_real= R  save_exp_out_imag= I)
HARDWARE_INPUT_PATH = "./RTF_input_from_hardware.txt"
# Fallback: separate real/imag files (one integer per line)
REAL_INPUT_PATH = "./RTF_input_Real.txt"
IMAG_INPUT_PATH = "./RTF_input_Imag.txt"

BRAM_REAL: list[int] | None = None
BRAM_IMAG: list[int] | None = None


# ---------------------------------------------------------------------------
# Two's‑complement helpers
# ---------------------------------------------------------------------------

def to_sint(value: int, bits: int) -> int:
    """Convert integer to signed two's‑complement with given bit width."""
    mask = (1 << bits) - 1
    value &= mask
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def add_tc(a: int, b: int, bits: int) -> int:
    """Two's‑complement add with wrapping."""
    return to_sint(a + b, bits)


def sub_tc(a: int, b: int, bits: int) -> int:
    """Two's‑complement subtract with wrapping."""
    return to_sint(a - b, bits)


def mul_tc(a: int, b: int, bits: int) -> int:
    """Two's‑complement multiply with wrapping to `bits`."""
    return to_sint(a * b, bits)


def div_tc(numer: int, denom: int, bits: int) -> int:
    """
    Integer division numer / denom with truncation toward zero,
    wrapped to two's‑complement `bits`.
    If denom == 0, return 0 (matches a safe‑guard behavior).
    """
    if denom == 0:
        return 0
    # Python's // truncates toward negative infinity, so correct sign manually.
    if numer * denom >= 0:
        q = abs(numer) // abs(denom)
    else:
        q = - (abs(numer) // abs(denom))
    return to_sint(q, bits)


def fixed_div_1_over_det(det: int) -> int:
    """
    Fixed‑point version of 1/det with DIVOUT_F_WIDTH fractional bits.

    The ideal fixed‑point value would be:
        inv_det_raw = round((1 << DIVOUT_F_WIDTH) / det)
    encoded on DIVOUT_TDATA_WIDTH bits (signed two's‑complement).

    This mimics a divider that outputs a signed Q(DIVOUT_TDATA_WIDTH-DIVOUT_F_WIDTH).DIVOUT_F_WIDTH
    fixed‑point number without extra scaling.
    """
    # numerator is 1.0 in fixed‑point with det fractional compensation
    numer = 1 << (DIVOUT_F_WIDTH + DET_FRAC_BITS + DIVOUT_GUARD_BITS)
    if det == 0:
        return 0
    # Round-to-nearest at the divider output (with guard bits).
    abs_n = abs(numer)
    abs_d = abs(det)
    q_full = (abs_n + (abs_d // 2)) // abs_d
    # Round back to DIVOUT_F_WIDTH fractional bits.
    if DIVOUT_GUARD_BITS > 0:
        q_full = (q_full + (1 << (DIVOUT_GUARD_BITS - 1))) >> DIVOUT_GUARD_BITS
    if numer * det < 0:
        q_full = -q_full
    return to_sint(q_full, DIVOUT_TDATA_WIDTH)


# ---------------------------------------------------------------------------
# Testbench‑style BRAM pattern
# ---------------------------------------------------------------------------

# Pattern for hardware dump: save_exp_out_addr= N  save_exp_out_real= R  save_exp_out_imag= I
_RE_HARDWARE_LINE = re.compile(
    r"save_exp_out_addr=\s*(-?\d+)\s+save_exp_out_real=\s*(-?\d+)\s+save_exp_out_imag=\s*(-?\d+)"
)


def load_bram_inputs_from_hardware(
    path: str, expected_len: int
) -> tuple[list[int], list[int]]:
    """
    Load BRAM inputs from hardware dump file.
    Each line: save_exp_out_addr= N  save_exp_out_real= R  save_exp_out_imag= I
    Returns (real_vals, imag_vals) of length expected_len, ordered by addr 1..N.
    """
    real_vals: list[int] = []
    imag_vals: list[int] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _RE_HARDWARE_LINE.match(line)
            if not m:
                continue
            _addr, r, i = m.groups()
            real_vals.append(int(r))
            imag_vals.append(int(i))
    if len(real_vals) < expected_len or len(imag_vals) < expected_len:
        raise ValueError(
            "Hardware input length mismatch: "
            f"got {len(real_vals)} samples, expected={expected_len}"
        )
    return real_vals[:expected_len], imag_vals[:expected_len]


def load_bram_inputs(
    real_path: str, imag_path: str, expected_len: int
) -> tuple[list[int], list[int]]:
    """Load BRAM inputs from two text files (one integer per line)."""
    with open(real_path, "r", encoding="utf-8") as f_real:
        real_vals = [int(line.strip()) for line in f_real if line.strip()]
    with open(imag_path, "r", encoding="utf-8") as f_imag:
        imag_vals = [int(line.strip()) for line in f_imag if line.strip()]

    if len(real_vals) < expected_len or len(imag_vals) < expected_len:
        raise ValueError(
            "Input length mismatch: "
            f"real={len(real_vals)}, imag={len(imag_vals)}, "
            f"expected={expected_len}"
        )

    # Keep only the first expected_len samples if files are longer.
    return real_vals[:expected_len], imag_vals[:expected_len]


def tb_bram_value(index: int) -> tuple[int, int]:
    """
    Reproduce the BRAM initialization used in RTF_top_tb.v:

        bram_rd_mem_real[i] = (i % 1000) - 500
        bram_rd_mem_imag[i] = ((i + 1) % 1000) - 500

    Returned values are clipped/wrapped to DATA_WIDTH bits.
    """
    if BRAM_REAL is not None and BRAM_IMAG is not None:
        real = BRAM_REAL[index]
        imag = BRAM_IMAG[index]
    else:
        real = (index % 1000) - 500
        imag = ((index + 1) % 1000) - 500
    return to_sint(real, DATA_WIDTH), to_sint(imag, DATA_WIDTH)


def build_af_for_freq(freq: int) -> tuple[list[complex], list[complex]]:
    """
    Build sor0 and sor1 arrays (length MIC_NUM) for a given frequency index,
    using the same indexing as the hardware BRAM.
    """
    sor0 = []
    sor1 = []
    base = freq * PER_FREQ
    for mic in range(MIC_NUM):
        idx0 = base + 0 * MIC_NUM + mic  # sor0
        idx1 = base + 1 * MIC_NUM + mic  # sor1
        r0, i0 = tb_bram_value(idx0)
        r1, i1 = tb_bram_value(idx1)
        sor0.append(complex(r0, i0))
        sor1.append(complex(r1, i1))
    return sor0, sor1


# ---------------------------------------------------------------------------
# Core algorithm: mirror RTF_top FSM computations (combinational view)
# ---------------------------------------------------------------------------

def compute_g_matrix(
    sor0: list[complex], sor1: list[complex]
) -> tuple[int, int, int, int, int, int]:
    """
    Compute g11_real_acc, g12_real_acc, g12_imag_acc, g22_real_acc
    with the same accumulation pattern and widths as hardware.

    Returns:
        (g11_before_lambda, g12_real_acc, g12_imag_acc, g22_before_lambda,
         g11_after_lambda,  g22_after_lambda)
    """
    g11_real_acc = 0
    g12_real_acc = 0
    g12_imag_acc = 0
    g22_real_acc = 0

    for mic in range(MIC_NUM):
        a0 = sor0[mic]
        a1 = sor1[mic]

        # sor0 contribution to g11
        g11_term = int(a0.real) * int(a0.real) + int(a0.imag) * int(a0.imag)
        g11_real_acc = add_tc(g11_real_acc, g11_term, ACC_WIDTH)

        # sor1 contribution to g22
        g22_term = int(a1.real) * int(a1.real) + int(a1.imag) * int(a1.imag)
        g22_real_acc = add_tc(g22_real_acc, g22_term, ACC_WIDTH)

        # g12 accumulation: conj(sor0) * sor1
        g12_r_term = int(a0.real) * int(a1.real) + int(a0.imag) * int(a1.imag)
        g12_i_term = int(a0.real) * int(a1.imag) - int(a0.imag) * int(a1.real)
        g12_real_acc = add_tc(g12_real_acc, g12_r_term, ACC_WIDTH)
        g12_imag_acc = add_tc(g12_imag_acc, g12_i_term, ACC_WIDTH)

    # Record AH * A(F) result (before adding lambda on diagonal)
    g11_before = g11_real_acc
    g22_before = g22_real_acc

    # PLUS state: add lambda to diagonal terms (G + lambda*I)
    g11_after = add_tc(g11_real_acc, LAMBDA, ACC_WIDTH)
    g22_after = add_tc(g22_real_acc, LAMBDA, ACC_WIDTH)

    return g11_before, g12_real_acc, g12_imag_acc, g22_before, g11_after, g22_after


def compute_inv_g(
    g11: int, g12_r: int, g12_i: int, g22: int
) -> tuple[int, int, int, int, int]:
    """
    Mirror S_CALDET1, S_CALDET2, S_INVDET, S_CALINVG.
    All arguments are ACC_WIDTH‑bit signed.
    """
    # det = g11 * g22  (stored in 32‑bit det register in hardware)
    det_mul = mul_tc(g11, g22, DET_WIDTH)

    # subtract |g12|^2
    g12_r_sqr = mul_tc(g12_r, g12_r, DET_WIDTH)
    g12_i_sqr = mul_tc(g12_i, g12_i, DET_WIDTH)
    det_sub = add_tc(g12_r_sqr, g12_i_sqr, DET_WIDTH)
    det = sub_tc(det_mul, det_sub, DET_WIDTH)

    # Normalize det to avoid inv_det underflow to zero.
    det_mag = abs(det)
    if det_mag == 0:
        det_shift = 0
        det_scaled = det
    else:
        if USE_FIXED_DET_SHIFT:
            det_shift = FIXED_DET_SHIFT
        else:
            det_bits = det_mag.bit_length()
            det_shift = max(0, det_bits - (DIVOUT_F_WIDTH + 1))
        det_scaled = to_sint(det >> det_shift, DET_WIDTH)

    # inv_det ≈ 1 / det_scaled in fixed‑point, with DIVOUT_F_WIDTH fractional bits
    inv_det_fp = fixed_div_1_over_det(det_scaled)

    # In hardware they multiply inv_det (divider output) with 32‑bit g* and
    # keep the result in 48‑bit registers. Here we mimic that by doing the
    # multiply and wrapping to INV_G_WIDTH bits; the common 2^DIVOUT_F_WIDTH
    # scale factor is kept implicitly (same as RTL).
    inv_g11_real = mul_tc(g22,    inv_det_fp, INV_G_WIDTH)
    inv_g12_real = mul_tc(-g12_r, inv_det_fp, INV_G_WIDTH)
    inv_g12_imag = mul_tc(-g12_i, inv_det_fp, INV_G_WIDTH)
    inv_g22_real = mul_tc(g11,    inv_det_fp, INV_G_WIDTH)

    return inv_g11_real, inv_g12_real, inv_g12_imag, inv_g22_real, det_shift


def compute_outputs_for_freq(freq: int) -> dict:
    """
    Compute the 16 complex outputs (row0 for MIC_NUM, then row1 for MIC_NUM)
    for a given frequency index, with the same per‑mic formulas as S_CALRESULT/S_WR.

    Returns a dictionary with:
        - 'sor0', 'sor1': input vectors (list of complex)
        - 'g_before': (g11, g12_real, g12_imag, g22) before adding lambda
        - 'g_after' : (g11, g12_real, g12_imag, g22) after adding lambda
        - 'inv_g'   : (inv_g11_real, inv_g12_real, inv_g12_imag, inv_g22_real)
        - 'outputs' : list of 16 (real, imag) tuples, wrapped to INV_G_WIDTH bits.
                      Order: mic 0..7 (row0), then mic 0..7 (row1).
    """
    sor0, sor1 = build_af_for_freq(freq)
    (
        g11_before,
        g12_r,
        g12_i,
        g22_before,
        g11_after,
        g22_after,
    ) = compute_g_matrix(sor0, sor1)
    # det_shift will be set after we have det; inv_g will be computed from RTL-style inv_det below.
    det_shift_dbg = 0  # set in det block below

    # Compute det and inv_det (divider output) same as RTL; then inv_g from inv_det (match hardware).
    det_mul = mul_tc(g11_after, g22_after, DET_WIDTH)
    g12_r_sqr = mul_tc(g12_r, g12_r, DET_WIDTH)
    g12_i_sqr = mul_tc(g12_i, g12_i, DET_WIDTH)
    det_sub = add_tc(g12_r_sqr, g12_i_sqr, DET_WIDTH)
    det = sub_tc(det_mul, det_sub, DET_WIDTH)
    det_mag = abs(det)
    det_bits = det_mag.bit_length() if det_mag else 0
    if det_mag == 0:
        det_shift_dbg = 0
        det_scaled = det
    else:
        if USE_FIXED_DET_SHIFT:
            det_shift_dbg = FIXED_DET_SHIFT
        else:
            det_shift_dbg = max(0, det_bits - (DIVOUT_F_WIDTH + 1))
        det_scaled = to_sint(det >> det_shift_dbg, DET_WIDTH)

    # Divider inputs (mirror RTF_top.v S_INVDET: det_scaled_div, dividend_rounded)
    det_scaled_div = to_sint(
        det_scaled & ((1 << DIVISOR_TDATA_WIDTH) - 1), DIVISOR_TDATA_WIDTH
    )
    det_scaled_abs_val = abs(det_scaled_div)
    det_half_val = det_scaled_abs_val >> 1
    # 48-bit signed: 1<<47 is -2^47 (match RTL signed dividend); quotient is negated in RTL for correct 1/det sign
    dividend_scaled_val = to_sint(1 << (DIVIDEND_TDATA_WIDTH - 1), DIVIDEND_TDATA_WIDTH)
    det_scaled_div_zero = det_scaled_div == 0
    if det_scaled_div_zero:
        dividend_rounded_val = 0
    else:
        if det_scaled_div < 0:
            dividend_rounded_val = to_sint(
                dividend_scaled_val - det_half_val, DIVIDEND_TDATA_WIDTH
            )
        else:
            dividend_rounded_val = to_sint(
                dividend_scaled_val + det_half_val, DIVIDEND_TDATA_WIDTH
            )

    inv_det_fp = fixed_div_1_over_det(det_scaled)

    # Simulate Vivado Divider Generator (PG151) Radix-2, signed, fractional: Eq. 3-1 (Quotient + IntRmd) then Eq. 3-3 (FractRmd).
    # Eq. 3-1: Dividend = Quotient * Divisor + IntRmd. Signed: remainder sign follows dividend (e.g. -6/4 = -1 REMD -2).
    # Eq. 3-3: FractRmd = (IntRmd * (2^F - 1)) / Divisor; fractional is truncated, F-bit signed (MSB = sign).
    if det_scaled_div_zero:
        inv_det_q_sw = 0
        inv_det_f_sw = 0
    else:
        den_sw = det_scaled_div
        abs_den = abs(den_sw)
        abs_dividend = abs(dividend_rounded_val)
        # Truncation toward zero for integer quotient (PG151 signed remainder convention).
        if (dividend_rounded_val >= 0) == (den_sw >= 0):
            quotient_int = abs_dividend // abs_den
        else:
            quotient_int = -(abs_dividend // abs_den)
        IntRmd = dividend_rounded_val - quotient_int * den_sw  # Eq. 3-1
        inv_det_q_sw = to_sint(quotient_int, DIVOUT_TDATA_WIDTH - DIVOUT_F_WIDTH)  # 48b signed
        # Eq. 3-3 signed: IP uses scale 2^(F-1) (not 2^F-1) so fractional magnitude fits F-bit signed; matches HW (~half of 2^F-1 scale).
        scale_f = 1 << (DIVOUT_F_WIDTH - 1)
        fract_numer = IntRmd * scale_f
        if (fract_numer >= 0) == (den_sw >= 0):
            fract_val = abs(fract_numer) // abs_den
        else:
            fract_val = -(abs(fract_numer) // abs_den)
        inv_det_f_sw = to_sint(fract_val, DIVOUT_F_WIDTH)  # 24b signed
    inv_det_raw_sw = to_sint(
        (inv_det_q_sw << (DIVOUT_F_WIDTH - 1)) + inv_det_f_sw, DIVOUT_TDATA_WIDTH
    )
    inv_det_raw_zero_sw = (inv_det_q_sw == 0) and (inv_det_f_sw == 0)
    div_shift = (DIVOUT_F_WIDTH - 1) + DET_FRAC_BITS + DIVOUT_GUARD_BITS - (DIVIDEND_TDATA_WIDTH - 1)
    # Same as RTL: inv_det = -(inv_det_raw << shift). When det > 0 the result is positive; if it exceeds
    # 72-bit signed max (2^71-1), to_sint wraps it to negative (overflow). RTL 72-bit register behaves the same.
    inv_det_hw_sw = (
        0
        if inv_det_raw_zero_sw
        else to_sint(-(inv_det_raw_sw << div_shift), DIVOUT_TDATA_WIDTH)
    )

    # Match RTL S_CALINVG: inv_g* = g* * inv_det (use RTL-style inv_det so output matches hardware).
    inv_g11 = mul_tc(g22_after, inv_det_hw_sw, INV_G_WIDTH)
    inv_g12_r = mul_tc(-g12_r, inv_det_hw_sw, INV_G_WIDTH)
    inv_g12_i = mul_tc(-g12_i, inv_det_hw_sw, INV_G_WIDTH)
    inv_g22 = mul_tc(g11_after, inv_det_hw_sw, INV_G_WIDTH)

    results: list[tuple[int, int]] = []

    # First row (result_row1 == 0 in hardware)
    for mic in range(MIC_NUM):
        a0 = sor0[mic]
        a1 = sor1[mic]
        r0 = mul_tc(inv_g11, int(a0.real), INV_G_WIDTH)
        r1 = mul_tc(inv_g12_r, int(a1.real), INV_G_WIDTH)
        r2 = mul_tc(inv_g12_i, int(a1.imag), INV_G_WIDTH)
        i0 = mul_tc(-inv_g11, int(a0.imag), INV_G_WIDTH)
        i1 = mul_tc(-inv_g12_r, int(a1.imag), INV_G_WIDTH)
        i2 = mul_tc(inv_g12_i, int(a1.real), INV_G_WIDTH)
        real_out = add_tc(add_tc(r0, r1, INV_G_WIDTH), r2, INV_G_WIDTH)
        imag_out = add_tc(add_tc(i0, i1, INV_G_WIDTH), i2, INV_G_WIDTH)
        results.append((real_out, imag_out))

    # Second row (result_row1 == 1 in hardware)
    for mic in range(MIC_NUM):
        a0 = sor0[mic]
        a1 = sor1[mic]
        r0 = mul_tc(inv_g12_r, int(a0.real), INV_G_WIDTH)
        r1 = mul_tc(-inv_g12_i, int(a0.imag), INV_G_WIDTH)
        r2 = mul_tc(inv_g22, int(a1.real), INV_G_WIDTH)
        i0 = mul_tc(-inv_g12_r, int(a0.imag), INV_G_WIDTH)
        i1 = mul_tc(-inv_g12_i, int(a0.real), INV_G_WIDTH)
        i2 = mul_tc(-inv_g22, int(a1.imag), INV_G_WIDTH)
        real_out = add_tc(add_tc(r0, r1, INV_G_WIDTH), r2, INV_G_WIDTH)
        imag_out = add_tc(add_tc(i0, i1, INV_G_WIDTH), i2, INV_G_WIDTH)
        results.append((real_out, imag_out))

    return {
        "sor0": sor0,
        "sor1": sor1,
        "g_before": (g11_before, g12_r, g12_i, g22_before),
        "g_after": (g11_after, g12_r, g12_i, g22_after),
        "det": {
            "det_mul": det_mul,
            "det_sub": det_sub,
            "det": det,
            "det_mag": det_mag,
            "det_bits": det_bits,
            "det_shift": det_shift_dbg,
            "det_scaled": det_scaled,
            "det_scaled_div": det_scaled_div,
            "det_scaled_abs": det_scaled_abs_val,
            "det_half": det_half_val,
            "dividend_scaled": dividend_scaled_val,
            "det_scaled_div_zero": det_scaled_div_zero,
            "dividend_rounded": dividend_rounded_val,
            "inv_det_fp": inv_det_fp,
            "inv_det_q_hw": inv_det_q_sw,
            "inv_det_f_hw": inv_det_f_sw,
            "inv_det_raw_hw": inv_det_raw_sw,
            "inv_det_raw_zero_hw": inv_det_raw_zero_sw,
            "inv_det_hw": inv_det_hw_sw,
        },
        "inv_g": (inv_g11, inv_g12_r, inv_g12_i, inv_g22),
        "outputs": results,
    }


def compute_float_pinv_for_freq(freq: int) -> np.ndarray:
    """
    Compute floating-point pinvA = inv(A^H A + lambda*I) * A^H for comparison.
    Inputs use Q-format scaling to float domain.
    """
    sor0, sor1 = build_af_for_freq(freq)
    scale_in = 2 ** INPUT_FRAC_BITS
    a0 = np.array([complex(int(x.real), int(x.imag)) for x in sor0], dtype=np.complex128) / scale_in
    a1 = np.array([complex(int(x.real), int(x.imag)) for x in sor1], dtype=np.complex128) / scale_in
    A = np.stack([a0, a1], axis=1)  # (MIC_NUM, SOR_NUM)
    G = A.conj().T @ A
    lambda_float = LAMBDA / (2 ** (2 * INPUT_FRAC_BITS))
    G_reg = G + lambda_float * np.eye(SOR_NUM, dtype=np.complex128)
    inv_g = np.linalg.inv(G_reg)
    pinv_a = inv_g @ A.conj().T  # (SOR_NUM, MIC_NUM)
    return pinv_a


def compute_float_stages_for_freq(freq: int) -> dict:
    """
    Compute floating-point stages for comparison:
    G (before/after lambda), det, inv_det, inv(G), and outputs (pinvA).
    """
    sor0, sor1 = build_af_for_freq(freq)
    scale_in = 2 ** INPUT_FRAC_BITS
    a0 = np.array([complex(int(x.real), int(x.imag)) for x in sor0], dtype=np.complex128) / scale_in
    a1 = np.array([complex(int(x.real), int(x.imag)) for x in sor1], dtype=np.complex128) / scale_in
    A = np.stack([a0, a1], axis=1)  # (MIC_NUM, SOR_NUM)
    G = A.conj().T @ A
    lambda_float = LAMBDA / (2 ** (2 * INPUT_FRAC_BITS))
    G_reg = G + lambda_float * np.eye(SOR_NUM, dtype=np.complex128)
    # In rare cases G_reg can be (numerically) singular; fall back to pseudoinverse
    # so that tuning/shrinking still works instead of crashing.
    try:
        det = np.linalg.det(G_reg)
        inv_g = np.linalg.inv(G_reg)
        inv_det = 0.0 if det == 0 else 1.0 / det
    except np.linalg.LinAlgError:
        det = np.linalg.det(G_reg)
        inv_g = np.linalg.pinv(G_reg)
        inv_det = 0.0
    pinv_a = inv_g @ A.conj().T  # (SOR_NUM, MIC_NUM)
    return {
        "G_before": G,
        "G_after": G_reg,
        "det": det,
        "inv_det": inv_det,
        "inv_g": inv_g,
        "pinv_a": pinv_a,
    }


def compute_max_error_for_freq(freq: int) -> tuple[float, dict]:
    """
    Compute the maximum absolute error vs float for the final outputs only (pinvA 16 complex values).
    """
    info = compute_outputs_for_freq(freq)
    float_info = compute_float_stages_for_freq(freq)
    det_info = info["det"]
    det_shift = det_info["det_shift"]
    scale_out = 2 ** (DIVOUT_F_WIDTH + det_shift + 3 * INPUT_FRAC_BITS)

    max_err = 0.0
    max_info: dict = {"stage": "output", "freq": freq, "mic": 0}

    pinv_a = float_info["pinv_a"]
    for mic in range(MIC_NUM):
        fx_row0 = info["outputs"][mic]
        fx_row1 = info["outputs"][MIC_NUM + mic]
        fx_row0_c = complex(fx_row0[0] / scale_out, fx_row0[1] / scale_out)
        fx_row1_c = complex(fx_row1[0] / scale_out, fx_row1[1] / scale_out)
        err0 = abs(pinv_a[0, mic] - fx_row0_c)
        if err0 > max_err:
            max_err = err0
            max_info = {"stage": "output", "freq": freq, "mic": mic, "row": 0}
        err1 = abs(pinv_a[1, mic] - fx_row1_c)
        if err1 > max_err:
            max_err = err1
            max_info = {"stage": "output", "freq": freq, "mic": mic, "row": 1}

    return max_err, max_info


def compute_max_error_all(max_freq: int) -> tuple[float, dict]:
    """
    Compute the maximum absolute error across all freqs and stages.
    """
    max_err = 0.0
    max_info: dict = {}
    for freq in range(max_freq):
        err, info = compute_max_error_for_freq(freq)
        if err > max_err:
            max_err = err
            max_info = info
    return max_err, max_info


def collect_all_errors_for_freq(freq: int) -> list[tuple[float, dict]]:
    """
    Collect (err, info) for final outputs only (16 complex pinvA values per freq).
    """
    info = compute_outputs_for_freq(freq)
    float_info = compute_float_stages_for_freq(freq)
    det_info = info["det"]
    det_shift = det_info["det_shift"]
    scale_out = 2 ** (DIVOUT_F_WIDTH + det_shift + 3 * INPUT_FRAC_BITS)

    out: list[tuple[float, dict]] = []
    pinv_a = float_info["pinv_a"]
    for mic in range(MIC_NUM):
        fx_row0 = info["outputs"][mic]
        fx_row1 = info["outputs"][MIC_NUM + mic]
        fx_row0_c = complex(fx_row0[0] / scale_out, fx_row0[1] / scale_out)
        fx_row1_c = complex(fx_row1[0] / scale_out, fx_row1[1] / scale_out)
        err0 = abs(pinv_a[0, mic] - fx_row0_c)
        out.append((err0, {"stage": "output", "freq": freq, "mic": mic, "row": 0}))
        err1 = abs(pinv_a[1, mic] - fx_row1_c)
        out.append((err1, {"stage": "output", "freq": freq, "mic": mic, "row": 1}))

    return out


def compute_top_n_errors(max_freq: int, n: int = 10) -> list[tuple[float, dict]]:
    """
    One entry per frequency: max error for that freq (across all outputs). Return top n by error magnitude.
    """
    all_pairs: list[tuple[float, dict]] = []
    for freq in range(max_freq):
        err, info = compute_max_error_for_freq(freq)
        all_pairs.append((err, info))
    all_pairs.sort(key=lambda x: -x[0])
    return all_pairs[:n]


def dump_max_error_detail(max_info: dict) -> str:
    """
    Build a detailed report for the worst-case error location.
    """
    if not max_info:
        return ""
    freq = max_info["freq"]
    info = compute_outputs_for_freq(freq)
    float_info = compute_float_stages_for_freq(freq)

    g11_b, g12_r, g12_i, g22_b = info["g_before"]
    g11_a, g12_r_a, g12_i_a, g22_a = info["g_after"]
    det_info = info["det"]
    det = det_info["det"]
    inv_det_fp = det_info["inv_det_fp"]
    det_shift = det_info["det_shift"]
    inv_g11, inv_g12_r, inv_g12_i, inv_g22 = info["inv_g"]

    scale_g = 2 ** (2 * INPUT_FRAC_BITS)
    scale_det = 2 ** DET_FRAC_BITS
    scale_inv_det = 2 ** (DIVOUT_F_WIDTH + det_shift)
    scale_inv_g = 2 ** (DIVOUT_F_WIDTH + det_shift + 2 * INPUT_FRAC_BITS)
    scale_out = 2 ** (DIVOUT_F_WIDTH + det_shift + 3 * INPUT_FRAC_BITS)

    def fmt_c(z: complex) -> str:
        return f"{z.real:+.6e}{z.imag:+.6e}j"

    def fmt_m(m: np.ndarray) -> str:
        return (
            f"[[{fmt_c(m[0,0])}, {fmt_c(m[0,1])}], "
            f"[{fmt_c(m[1,0])}, {fmt_c(m[1,1])}]]"
        )

    lines = []
    lines.append("  [max_err_detail]")
    lines.append(f"    location: {max_info}")
    lines.append(f"    det_shift: {det_shift}")

    g_before_f = float_info["G_before"]
    g_after_f = float_info["G_after"]
    g_before_fx = np.array(
        [
            [complex(g11_b, 0), complex(g12_r, g12_i)],
            [complex(g12_r, -g12_i), complex(g22_b, 0)],
        ],
        dtype=np.complex128,
    ) / scale_g
    g_after_fx = np.array(
        [
            [complex(g11_a, 0), complex(g12_r_a, g12_i_a)],
            [complex(g12_r_a, -g12_i_a), complex(g22_a, 0)],
        ],
        dtype=np.complex128,
    ) / scale_g
    lines.append(f"    {'G_before float':<14} : {fmt_m(g_before_f)}")
    lines.append(f"    {'G_before fixed':<14} : {fmt_m(g_before_fx)}")
    lines.append(f"    {'G_after  float':<14} : {fmt_m(g_after_f)}")
    lines.append(f"    {'G_after  fixed':<14} : {fmt_m(g_after_fx)}")

    det_f = float_info["det"]
    det_fx = det / scale_det
    inv_det_f = float_info["inv_det"]
    inv_det_fx = inv_det_fp / scale_inv_det
    lines.append(f"    {'det      float':<14} : {fmt_c(det_f)}")
    lines.append(f"    {'det      fixed':<14} : {fmt_c(det_fx)}")
    lines.append(f"    {'inv_det  float':<14} : {fmt_c(inv_det_f)}")
    lines.append(f"    {'inv_det  fixed':<14} : {fmt_c(inv_det_fx)}")

    inv_g_f = float_info["inv_g"]
    inv_g_fx = np.array(
        [
            [complex(inv_g11, 0), complex(inv_g12_r, inv_g12_i)],
            [complex(inv_g12_r, -inv_g12_i), complex(inv_g22, 0)],
        ],
        dtype=np.complex128,
    ) / scale_inv_g
    lines.append(f"    {'inv_g    float':<14} : {fmt_m(inv_g_f)}")
    lines.append(f"    {'inv_g    fixed':<14} : {fmt_m(inv_g_fx)}")

    pinv_a = float_info["pinv_a"]
    if "mic" in max_info:
        mic = max_info["mic"]
        fx_row0 = info["outputs"][mic]
        fx_row1 = info["outputs"][MIC_NUM + mic]
        fx_row0_c = complex(fx_row0[0] / scale_out, fx_row0[1] / scale_out)
        fx_row1_c = complex(fx_row1[0] / scale_out, fx_row1[1] / scale_out)
        lines.append(f"    {'pinvA_row0 float':<14} : {fmt_c(pinv_a[0, mic])}")
        lines.append(f"    {'pinvA_row0 fixed':<14} : {fmt_c(fx_row0_c)}")
        lines.append(f"    {'pinvA_row1 float':<14} : {fmt_c(pinv_a[1, mic])}")
        lines.append(f"    {'pinvA_row1 fixed':<14} : {fmt_c(fx_row1_c)}")
    return "\n".join(lines)


def tune_divout_widths(max_freq: int) -> tuple[int, int, float, dict]:
    """
    Increase DIVOUT_F_WIDTH and DIVOUT_TDATA_WIDTH until max error is within ERR_TARGET.
    """
    global DIVOUT_F_WIDTH, DIVOUT_TDATA_WIDTH, DIVOUT_GUARD_BITS
    if not ENABLE_FLOAT_COMPARE:
        return DIVOUT_F_WIDTH, DIVOUT_TDATA_WIDTH, 0.0, {}
    max_err, max_info = compute_max_error_all(max_freq)

    if max_err <= ERR_TRIGGER:
        return DIVOUT_F_WIDTH, DIVOUT_TDATA_WIDTH, max_err, max_info

    while max_err > ERR_TARGET:
        grew = False
        if DIVOUT_F_WIDTH < min(DIVOUT_F_WIDTH_MAX, DIVOUT_TDATA_WIDTH - 1):
            DIVOUT_F_WIDTH += 1
            grew = True
            print(f"[tune] Adjusting DIVOUT_F_WIDTH -> {DIVOUT_F_WIDTH}")
        elif DIVOUT_TDATA_WIDTH < DIVOUT_TDATA_WIDTH_MAX:
            DIVOUT_TDATA_WIDTH += 1
            grew = True
            if DIVOUT_F_WIDTH < min(DIVOUT_F_WIDTH_MAX, DIVOUT_TDATA_WIDTH - 1):
                DIVOUT_F_WIDTH += 1
            print(
                "[tune] Adjusting DIVOUT_TDATA_WIDTH -> "
                f"{DIVOUT_TDATA_WIDTH}, DIVOUT_F_WIDTH -> {DIVOUT_F_WIDTH}"
            )
        elif DIVOUT_GUARD_BITS < DIVOUT_GUARD_BITS_MAX:
            DIVOUT_GUARD_BITS += 1
            grew = True
            print(f"[tune] Adjusting DIVOUT_GUARD_BITS -> {DIVOUT_GUARD_BITS}")
        if not grew:
            break
        max_err, max_info = compute_max_error_all(max_freq)

    if max_err > ERR_TARGET:
        print(
            "[tune][warn] Target not reached with divider widths "
            f"(max_err={max_err:.3e})"
        )
    return DIVOUT_F_WIDTH, DIVOUT_TDATA_WIDTH, max_err, max_info


def tune_internal_widths(max_freq: int) -> tuple[int, int, int, float, dict]:
    """
    Increase ACC/DET/INV_G widths until max error is within ERR_TARGET.
    """
    global ACC_WIDTH, DET_WIDTH, INV_G_WIDTH
    if not ENABLE_FLOAT_COMPARE:
        return ACC_WIDTH, DET_WIDTH, INV_G_WIDTH, 0.0, {}

    max_err, max_info = compute_max_error_all(max_freq)
    if max_err <= ERR_TRIGGER:
        return ACC_WIDTH, DET_WIDTH, INV_G_WIDTH, max_err, max_info

    while max_err > ERR_TARGET:
        grew = False
        if ACC_WIDTH < ACC_WIDTH_MAX:
            ACC_WIDTH += 1
            grew = True
        if DET_WIDTH < DET_WIDTH_MAX:
            DET_WIDTH += 1
            grew = True
        if INV_G_WIDTH < INV_G_WIDTH_MAX:
            INV_G_WIDTH += 1
            grew = True
        if not grew:
            break
        print(
            "[tune] Adjusting widths -> "
            f"ACC_WIDTH={ACC_WIDTH}, DET_WIDTH={DET_WIDTH}, INV_G_WIDTH={INV_G_WIDTH}"
        )
        max_err, max_info = compute_max_error_all(max_freq)

    if max_err > ERR_TARGET:
        print(
            "[tune][warn] Target not reached with internal widths "
            f"(max_err={max_err:.3e})"
        )
    return ACC_WIDTH, DET_WIDTH, INV_G_WIDTH, max_err, max_info


def shrink_widths(max_freq: int) -> tuple[dict, float, dict]:
    """
    Greedily reduce widths while keeping max_err <= ERR_TARGET.
    """
    global ACC_WIDTH
    global DET_WIDTH
    global INV_G_WIDTH
    global DIVOUT_F_WIDTH
    global DIVOUT_TDATA_WIDTH
    global DIVOUT_GUARD_BITS
    max_err, max_info = compute_max_error_all(max_freq)
    if max_err > ERR_TARGET:
        return {}, max_err, max_info

    changed = True
    while changed:
        changed = False
        # Try internal widths first (do not go below INV_G_WIDTH_FLOOR)
        if INV_G_WIDTH > max(1, INV_G_WIDTH_FLOOR):
            prev = INV_G_WIDTH
            INV_G_WIDTH -= 1
            err, info = compute_max_error_all(max_freq)
            if err <= ERR_TARGET:
                print(f"[shrink] INV_G_WIDTH -> {INV_G_WIDTH}")
                max_err, max_info = err, info
                changed = True
            else:
                INV_G_WIDTH = prev

        if DET_WIDTH > 1:
            prev = DET_WIDTH
            DET_WIDTH -= 1
            err, info = compute_max_error_all(max_freq)
            if err <= ERR_TARGET:
                print(f"[shrink] DET_WIDTH -> {DET_WIDTH}")
                max_err, max_info = err, info
                changed = True
            else:
                DET_WIDTH = prev

        if ACC_WIDTH > 1:
            prev = ACC_WIDTH
            ACC_WIDTH -= 1
            err, info = compute_max_error_all(max_freq)
            if err <= ERR_TARGET:
                print(f"[shrink] ACC_WIDTH -> {ACC_WIDTH}")
                max_err, max_info = err, info
                changed = True
            else:
                ACC_WIDTH = prev

        # Then divider widths
        if DIVOUT_TDATA_WIDTH > (DIVOUT_F_WIDTH + 1):
            prev = DIVOUT_TDATA_WIDTH
            DIVOUT_TDATA_WIDTH -= 1
            err, info = compute_max_error_all(max_freq)
            if err <= ERR_TARGET:
                print(f"[shrink] DIVOUT_TDATA_WIDTH -> {DIVOUT_TDATA_WIDTH}")
                max_err, max_info = err, info
                changed = True
            else:
                DIVOUT_TDATA_WIDTH = prev

        if DIVOUT_F_WIDTH > 1:
            prev = DIVOUT_F_WIDTH
            DIVOUT_F_WIDTH -= 1
            if DIVOUT_F_WIDTH >= DIVOUT_TDATA_WIDTH:
                DIVOUT_F_WIDTH = prev
            else:
                err, info = compute_max_error_all(max_freq)
                if err <= ERR_TARGET:
                    print(f"[shrink] DIVOUT_F_WIDTH -> {DIVOUT_F_WIDTH}")
                    max_err, max_info = err, info
                    changed = True
                else:
                    DIVOUT_F_WIDTH = prev

        if DIVOUT_GUARD_BITS > 0:
            prev = DIVOUT_GUARD_BITS
            DIVOUT_GUARD_BITS -= 1
            err, info = compute_max_error_all(max_freq)
            if err <= ERR_TARGET:
                print(f"[shrink] DIVOUT_GUARD_BITS -> {DIVOUT_GUARD_BITS}")
                max_err, max_info = err, info
                changed = True
            else:
                DIVOUT_GUARD_BITS = prev

    widths = {
        "ACC_WIDTH": ACC_WIDTH,
        "DET_WIDTH": DET_WIDTH,
        "INV_G_WIDTH": INV_G_WIDTH,
        "DIVOUT_TDATA_WIDTH": DIVOUT_TDATA_WIDTH,
        "DIVOUT_F_WIDTH": DIVOUT_F_WIDTH,
        "DIVOUT_GUARD_BITS": DIVOUT_GUARD_BITS,
    }
    return widths, max_err, max_info


def compute_required_divout_integer_bits(max_freq: int) -> tuple[int, int]:
    """
    Scan all freqs and return (max_integer_part_magnitude, recommended_integer_bits).
    Integer part = abs(inv_det_fp) >> DIVOUT_F_WIDTH; recommended bits = bit_length + 1 for sign.
    """
    max_int_mag = 0
    for freq in range(max_freq):
        info = compute_outputs_for_freq(freq)
        inv_det_fp = info["det"]["inv_det_fp"]
        int_mag = abs(inv_det_fp) >> DIVOUT_F_WIDTH
        max_int_mag = max(max_int_mag, int_mag)
    if max_int_mag == 0:
        return 0, 0
    recommended_bits = max_int_mag.bit_length() + 1  # +1 for sign
    return max_int_mag, recommended_bits


def main() -> None:
    """
    Compute and print software results for the first few freqs,
    so you can compare with RTL simulation at the corresponding addresses.
    Also writes the same output to RTF_software_output.txt.
    """
    global BRAM_REAL, BRAM_IMAG
    out_path = "./RTF_software_output.txt"
    max_freq = min(257, FREQ_NUM)

    def log(msg: str = "") -> None:
        print(msg)
        f.write(msg + "\n")

    def fmt_c(z: complex) -> str:
        return f"{z.real:+.6e}{z.imag:+.6e}j"

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(out_path, "w", encoding="utf-8") as f:
        try:
            BRAM_REAL, BRAM_IMAG = load_bram_inputs_from_hardware(
                HARDWARE_INPUT_PATH, TOTAL_NUM
            )
            log(f"[info] Loaded {TOTAL_NUM} samples from {HARDWARE_INPUT_PATH} (real/imag per line).")
        except FileNotFoundError:
            try:
                BRAM_REAL, BRAM_IMAG = load_bram_inputs(
                    REAL_INPUT_PATH, IMAG_INPUT_PATH, TOTAL_NUM
                )
                log(f"[info] Loaded from {REAL_INPUT_PATH} and {IMAG_INPUT_PATH}.")
            except FileNotFoundError:
                BRAM_REAL, BRAM_IMAG = None, None
                log("[warn] Input files not found, using testbench pattern.")
            except ValueError as exc:
                BRAM_REAL, BRAM_IMAG = None, None
                log(f"[warn] {exc}; using testbench pattern.")
        except ValueError as exc:
            BRAM_REAL, BRAM_IMAG = None, None
            log(f"[warn] {exc}; using testbench pattern.")

        f.write(f"Generated: {timestamp_str}\n")
        f.write(
            f"det_shift: {'FIXED=' + str(FIXED_DET_SHIFT) if USE_FIXED_DET_SHIFT else 'dynamic'}\n"
        )
        f.write(
            f"Auto-tune: {AUTO_TUNE_DIVOUT_F_WIDTH}, "
            f"ERR_TRIGGER={ERR_TRIGGER:.1e}, ERR_TARGET={ERR_TARGET:.1e}\n"
        )
        # Q-format summary for quick reference (approximate; some stages also include dynamic det_shift)
        log("Q-format summary (approx):")
        # Input a(f): s(INTEGER).FRAC where FRAC = INPUT_FRAC_BITS
        input_int_bits = max(1, DATA_WIDTH - INPUT_FRAC_BITS)
        log(f"  Input a(f):        s{input_int_bits}.{INPUT_FRAC_BITS}  (DATA_WIDTH={DATA_WIDTH})")
        # G elements use ACC_WIDTH and roughly 2*INPUT_FRAC_BITS fractional bits
        g_frac_bits = 2 * INPUT_FRAC_BITS
        g_int_bits = max(1, ACC_WIDTH - g_frac_bits)
        log(f"  G elements:        s{g_int_bits}.{g_frac_bits}  (ACC_WIDTH={ACC_WIDTH})")
        # det uses DET_WIDTH and DET_FRAC_BITS fractional bits
        det_int_bits = max(1, DET_WIDTH - DET_FRAC_BITS)
        log(f"  det:               s{det_int_bits}.{DET_FRAC_BITS}  (DET_WIDTH={DET_WIDTH})")
        # Divider output nominal Q format (before considering det_shift)
        div_int_bits = max(1, DIVOUT_TDATA_WIDTH - DIVOUT_F_WIDTH)
        log(
            f"  divider output:    s{div_int_bits}.{DIVOUT_F_WIDTH}  "
            f"(DIVOUT_TDATA_WIDTH={DIVOUT_TDATA_WIDTH})"
        )
        log(
            "  inv(G) / outputs:  "
            "INV_G_WIDTH bits, effective frac ≈ DIVOUT_F_WIDTH + det_shift + "
            f"{2 * INPUT_FRAC_BITS} (invG) / {3 * INPUT_FRAC_BITS} (outputs)"
        )
        log("")
        
        if AUTO_TUNE_DIVOUT_F_WIDTH:
            log("=" * 60)
            tuned_width, tuned_tdata, tuned_err, tuned_info = tune_divout_widths(max_freq)
            f.write(
                f"Tuned DIVOUT_F_WIDTH={tuned_width}, DIVOUT_TDATA_WIDTH={tuned_tdata} "
                f"(max_err={tuned_err:.3e})\n"
            )
            if tuned_info:
                f.write(f"Max error at {tuned_info}\n")
                if DUMP_MAX_ERROR_DETAIL:
                    f.write(dump_max_error_detail(tuned_info) + "\n")
        
        if AUTO_TUNE_INTERNAL_WIDTHS:
            log("=" * 60)
            acc_w, det_w, inv_w, tuned_err, tuned_info = tune_internal_widths(max_freq)
            f.write(
                f"Tuned ACC_WIDTH={acc_w}, DET_WIDTH={det_w}, INV_G_WIDTH={inv_w} "
                f"(max_err={tuned_err:.3e})\n"
            )
            if tuned_info:
                f.write(f"Max error at {tuned_info}\n")
                if DUMP_MAX_ERROR_DETAIL:
                    f.write(dump_max_error_detail(tuned_info) + "\n")
        if SHRINK_WIDTHS:
            log("=" * 60)
            widths, shrunk_err, shrunk_info = shrink_widths(max_freq)
            if widths:
                log("Shrunk widths:")
                max_key = max(len(k) for k in widths.keys())
                for k, v in widths.items():
                    log(f"  {k:<{max_key}} = {v}")
                log(f"  {'max_err':<{max_key}} = {shrunk_err:.3e}")
                if shrunk_info:
                    log(f"Max error at {shrunk_info}")
            else:
                log(f"Shrunk widths skipped: max_err={shrunk_err:.3e} > ERR_TARGET={ERR_TARGET:.1e}")
                top_errors = compute_top_n_errors(max_freq, TOP_N_ERRORS_TO_PRINT)
                for i, (err, loc) in enumerate(top_errors, 1):
                    log(f"  #{i:2d} err={err:.6e} at {loc}")
        # Always print max error vs float when float comparison is enabled
        if ENABLE_FLOAT_COMPARE:
            log("=" * 60)
            max_err, _ = compute_max_error_all(max_freq)
            log(f"Max error (vs float): {max_err:.6e}")
            top_errors = compute_top_n_errors(max_freq, TOP_N_ERRORS_TO_PRINT)
            for i, (err, loc) in enumerate(top_errors, 1):
                log(f"  #{i:2d} err={err:.6e} at {loc}")
        log("=" * 60)
        max_int_mag, req_int_bits = compute_required_divout_integer_bits(max_freq)
        log(
            f"Divider output integer part: max_magnitude={max_int_mag}, "
            f"recommended_integer_bits={req_int_bits} "
            f"(current={DIVOUT_TDATA_WIDTH - DIVOUT_F_WIDTH})"
        )
        log("=" * 60)
        f.write(
            f"Input Q format: {INPUT_Q_FORMAT} (float = val / 2^{INPUT_FRAC_BITS})\n"
        )
        for freq in range(max_freq):
            log("=" * 60)
            log(f"[freq {freq}] Inputs (sor0, sor1):")
            info = compute_outputs_for_freq(freq)

            # Scales and Q-format bit widths for float interpretation
            scale_in = 2 ** INPUT_FRAC_BITS
            scale_g = 2 ** (2 * INPUT_FRAC_BITS)
            scale_det = 2 ** DET_FRAC_BITS
            det_shift = info["det"]["det_shift"]
            scale_inv_det = 2 ** (DIVOUT_F_WIDTH + det_shift)
            scale_inv_g = 2 ** (DIVOUT_F_WIDTH + det_shift + 2 * INPUT_FRAC_BITS)
            scale_out = 2 ** (DIVOUT_F_WIDTH + det_shift + 3 * INPUT_FRAC_BITS)
            # Q format: signed int_bits.frac_bits (value = raw / 2^frac_bits)
            q_in_int = max(1, DATA_WIDTH - INPUT_FRAC_BITS)
            q_g_frac = 2 * INPUT_FRAC_BITS
            q_g_int = max(1, ACC_WIDTH - q_g_frac)
            q_det_int = max(1, DET_WIDTH - DET_FRAC_BITS)
            q_inv_det_frac = DIVOUT_F_WIDTH + det_shift
            q_inv_det_int = max(1, DIVOUT_TDATA_WIDTH - DIVOUT_F_WIDTH)
            q_inv_det_float_int = max(1, DIVOUT_TDATA_WIDTH - q_inv_det_frac)
            q_inv_g_frac = DIVOUT_F_WIDTH + det_shift + 2 * INPUT_FRAC_BITS
            q_inv_g_int = max(1, INV_G_WIDTH - q_inv_g_frac)
            q_out_frac = DIVOUT_F_WIDTH + det_shift + 3 * INPUT_FRAC_BITS
            q_out_int = max(1, INV_G_WIDTH - q_out_frac)

            # Print input vectors (raw + float in Q format)
            log(f"  [raw integer, float in s{q_in_int}.{INPUT_FRAC_BITS}]")
            for mic in range(MIC_NUM):
                a0 = info["sor0"][mic]
                a1 = info["sor1"][mic]
                log(
                    f"  mic={mic:2d}  sor0=({int(a0.real):6d}, {int(a0.imag):6d})  "
                    f"float=({int(a0.real)/scale_in:+.6e}, {int(a0.imag)/scale_in:+.6e})  "
                    f"sor1=({int(a1.real):6d}, {int(a1.imag):6d})  "
                    f"float=({int(a1.real)/scale_in:+.6e}, {int(a1.imag)/scale_in:+.6e})"
                )

            # Print G = A^H * A (before lambda) (raw + float in Q format)
            g11_b, g12_r, g12_i, g22_b = info["g_before"]
            log(f"\n  G = A^H * A (before lambda) [raw, float in s{q_g_int}.{q_g_frac}]:")
            log(f"    g11 = {g11_b:12d}   float = {g11_b/scale_g:+.6e}")
            log(f"    g12 = ({g12_r:12d}, {g12_i:12d})   float = ({g12_r/scale_g:+.6e}, {g12_i/scale_g:+.6e})")
            log(f"    g22 = {g22_b:12d}   float = {g22_b/scale_g:+.6e}")

            # Print G + lambda*I (raw + float in Q format)
            g11_a, g12_r_a, g12_i_a, g22_a = info["g_after"]
            log(f"\n  G + lambda*I [raw, float in s{q_g_int}.{q_g_frac}]:")
            log(f"    lambda = {LAMBDA:12d}   float = {LAMBDA/scale_g:+.6e}")
            log(f"    g11 = {g11_a:12d}   float = {g11_a/scale_g:+.6e}")
            log(f"    g12 = ({g12_r_a:12d}, {g12_i_a:12d})   float = ({g12_r_a/scale_g:+.6e}, {g12_i_a/scale_g:+.6e})")
            log(f"    g22 = {g22_a:12d}   float = {g22_a/scale_g:+.6e}")

            # Print det and divider output (inv_det) [raw + float by stage Q]
            det_info = info["det"]
            det_mul = det_info["det_mul"]
            det_sub = det_info["det_sub"]
            det = det_info["det"]
            inv_det_fp = det_info["inv_det_fp"]
            log(f"\n  det check [raw, float in s{q_det_int}.{DET_FRAC_BITS}]:")
            log(f"    det_mul = g11*g22           = {det_mul:12d}   float = {det_mul/scale_det:+.6e}")
            log(f"    det_sub = |g12|^2 (r^2+i^2) = {det_sub:12d}   float = {det_sub/scale_det:+.6e}")
            log(f"    det     = det_mul - det_sub = {det:12d}     float = {det/scale_det:+.6e}")

            # Divider inputs (match RTF_top.v for hardware comparison)
            dd = det_info
            log(f"\n  [Divider inputs - compare with RTL S_INVDET/S_SETDIV]")
            log(f"    det_mag             (|det|)              = {dd['det_mag']}")
            log(f"    det_bits            (bit_length)         = {dd['det_bits']}")
            log(f"    det_shift           (det_shift_use)      = {dd['det_shift']}")
            log(f"    det_scaled          (det >>> shift)      = {dd['det_scaled']}")
            log(f"    det_scaled_div      (divisor, 64b)       = {dd['det_scaled_div']}")
            log(f"    det_scaled_abs      (|det_scaled_div|)   = {dd['det_scaled_abs']}")
            log(f"    det_half            (det_scaled_abs>>>1) = {dd['det_half']}")
            log(f"    dividend_scaled     (1<<47)              = {dd['dividend_scaled']}")
            log(f"    det_scaled_div_zero                      = {dd['det_scaled_div_zero']}")
            log(f"    dividend_rounded    (dividend, 48b)      = {dd['dividend_rounded']}")

            # Hardware-style divider output (same as RTL: inv_det_q, inv_det_f, inv_det_raw, inv_det)
            log(f"\n  [Divider output - match RTL m_axis_dout, inv_det_raw, inv_det]")
            log(f"    inv_det_raw        (q<<23 + f, 72b)                 = {dd['inv_det_raw_hw']}")
            log(f"    inv_det_q          (quotient [71:24])               = {dd['inv_det_q_hw']}")
            log(f"    inv_det_f          (quotient [23:0])                = {dd['inv_det_f_hw']}")
            log(f"    inv_det_raw_zero                                    = {dd['inv_det_raw_zero_hw']}")
            # When det>0 intended inv_det is positive; negative here = 72-bit signed overflow (value > 2^71-1).
            log(f"    inv_det            (RTL-style, after negate<<, 72b) = {dd['inv_det_hw']}")

            # Same 1/det in software pipeline format (s48.24 + det_shift); raw integer differs from RTL format above.
            log(
                f"\n    inv_det (algorithm ref, s{q_inv_det_int}.{DIVOUT_F_WIDTH}, shift={det_shift}, float in s{q_inv_det_float_int}.{q_inv_det_frac}) = {inv_det_fp:16d}   "
                f"float = {inv_det_fp/scale_inv_det:+.6e}"
            )
            if det == 0:
                log("    [warn] det == 0 -> inv_det is forced to 0 by model")

            # Print inverse matrix elements (raw + float by inv_g scale)
            inv_g11, inv_g12_r, inv_g12_i, inv_g22 = info["inv_g"]
            log(f"\n  inv(G) [raw, float in s{q_inv_g_int}.{q_inv_g_frac}]:")
            log(f"    inv_g11_real = {inv_g11:16d}   float = {inv_g11/scale_inv_g:+.6e}")
            log(f"    inv_g12      = ({inv_g12_r:16d}, {inv_g12_i:16d})   float = ({inv_g12_r/scale_inv_g:+.6e}, {inv_g12_i/scale_inv_g:+.6e})")
            log(f"    inv_g22_real = {inv_g22:16d}   float = {inv_g22/scale_inv_g:+.6e}")

            # Print final outputs (raw + float = raw/scale_out)
            log(f"\n  Outputs (row0 then row1, 16 per freq) [raw, float in s{q_out_int}.{q_out_frac}]:")
            for idx, (r, i) in enumerate(info["outputs"]):
                log(f"    out_idx={idx:2d}  real={r:16d}  float={r/scale_out:+.6e}  imag={i:16d}  float={i/scale_out:+.6e}")

            # Float comparison against fixed outputs (scaled back to float domain)
            if ENABLE_FLOAT_COMPARE:
                float_info = compute_float_stages_for_freq(freq)
                pinv_a = float_info["pinv_a"]
                scale_g = 2 ** (2 * INPUT_FRAC_BITS)
                scale_det = 2 ** DET_FRAC_BITS
                det_shift = det_info["det_shift"]
                scale_inv_det = 2 ** (DIVOUT_F_WIDTH + det_shift)
                scale_inv_g = 2 ** (DIVOUT_F_WIDTH + det_shift + 2 * INPUT_FRAC_BITS)
                scale_out = 2 ** (DIVOUT_F_WIDTH + det_shift + 3 * INPUT_FRAC_BITS)

                log("\n  Float vs fixed stages (float / fixed / error):")
                g_before_f = float_info["G_before"]
                g_after_f = float_info["G_after"]
                g_before_fx = np.array(
                    [
                        [complex(g11_b, 0), complex(g12_r, g12_i)],
                        [complex(g12_r, -g12_i), complex(g22_b, 0)],
                    ],
                    dtype=np.complex128,
                ) / scale_g
                g_after_fx = np.array(
                    [
                        [complex(g11_a, 0), complex(g12_r_a, g12_i_a)],
                        [complex(g12_r_a, -g12_i_a), complex(g22_a, 0)],
                    ],
                    dtype=np.complex128,
                ) / scale_g

                for r in range(2):
                    for c in range(2):
                        err = abs(g_before_f[r, c] - g_before_fx[r, c])
                        log(
                            "    G_before[%d,%d] float=%s  fixed=%s  err=%.3e"
                            % (r, c, fmt_c(g_before_f[r, c]), fmt_c(g_before_fx[r, c]), err)
                        )
                for r in range(2):
                    for c in range(2):
                        err = abs(g_after_f[r, c] - g_after_fx[r, c])
                        log(
                            "    G_after [%d,%d] float=%s  fixed=%s  err=%.3e"
                            % (r, c, fmt_c(g_after_f[r, c]), fmt_c(g_after_fx[r, c]), err)
                        )

                det_f = float_info["det"]
                det_fx = det / scale_det
                det_err = abs(det_f - det_fx)
                log(
                    "    det           float=%s  fixed=%s  err=%.3e"
                    % (fmt_c(det_f), fmt_c(det_fx), det_err)
                )

                inv_det_f = float_info["inv_det"]
                inv_det_fx = inv_det_fp / scale_inv_det
                inv_det_err = abs(inv_det_f - inv_det_fx)
                log(
                    "    inv_det       float=%s  fixed=%s  err=%.3e"
                    % (fmt_c(inv_det_f), fmt_c(inv_det_fx), inv_det_err)
                )

                inv_g_f = float_info["inv_g"]
                inv_g_fx = np.array(
                    [
                        [complex(inv_g11, 0), complex(inv_g12_r, inv_g12_i)],
                        [complex(inv_g12_r, -inv_g12_i), complex(inv_g22, 0)],
                    ],
                    dtype=np.complex128,
                ) / scale_inv_g
                for r in range(2):
                    for c in range(2):
                        err = abs(inv_g_f[r, c] - inv_g_fx[r, c])
                        log(
                            "    inv(G)[%d,%d]   float=%s  fixed=%s  err=%.3e"
                            % (r, c, fmt_c(inv_g_f[r, c]), fmt_c(inv_g_fx[r, c]), err)
                        )

                log("\n  Float vs fixed comparison (pinvA):")
                for mic in range(MIC_NUM):
                    fp_row0 = pinv_a[0, mic]
                    fp_row1 = pinv_a[1, mic]
                    fx_row0 = info["outputs"][mic]
                    fx_row1 = info["outputs"][MIC_NUM + mic]
                    fx_row0_c = complex(fx_row0[0] / scale_out, fx_row0[1] / scale_out)
                    fx_row1_c = complex(fx_row1[0] / scale_out, fx_row1[1] / scale_out)
                    err0 = abs(fp_row0 - fx_row0_c)
                    err1 = abs(fp_row1 - fx_row1_c)
                    log(
                        "    mic=%2d  row0 float=%s  fixed=%s  err=%.3e"
                        "  row1 float=%s  fixed=%s  err=%.3e"
                        % (
                            mic,
                            fmt_c(fp_row0), fmt_c(fx_row0_c), err0,
                            fmt_c(fp_row1), fmt_c(fx_row1_c), err1,
                        )
                    )

    print(f"\nOutput also written to {out_path}")


if __name__ == "__main__":
    main()

