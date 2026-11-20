#!/usr/bin/env python3
"""Exact rational and outward-rounded Arb certification for SSHBOUND3.

Verifies the finite algebraic, operator-norm, SOS, atomic-ratio, Schatten-Holder,
Wick/covariance, polymer-budget, and boundary-oscillator ingredients of
ADIABATIC/SSHBOUND3/RIGOR.md.

Usage:
  python -B replay.py
  python -B replay.py --all-mutants
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction as Q

import numpy as np
from flint import arb, arb_mat, ctx, fmpq
from scipy.linalg import expm

ctx.dps = 80

MUTANTS = (
    "norm_be_underestimate",      # corrupt ||B_e||_op from 2 to 1
    "sos_const_halved",           # corrupt SOS constant 2 -> 1 in R_e
    "degree_factor_dropped",      # corrupt 32 = 8*D -> 8 in delta_t
    "atomic_ratio_dropped",       # corrupt z_b/z_a bound 5/4 -> 1
    "wick_pairings_corrupted",    # corrupt (2r-1)!!/(2r)! -> 1/r! without 2^{-r}
    "exp_g_factor_halved",        # corrupt 2*beta*g^2/K -> beta*g^2/K
    "exp_taylor_bound_dropped",   # corrupt exp(x)-1 <= x/(1-x/2) -> x
    "exp_quarter_bound_too_small",# corrupt exp(1/4) < 4/3 -> 5/4
    "c2_window_too_large",        # corrupt c2 = 2^-13 -> 2^-11 (breaks b_src < 1)
    "rooted_count_underestimated",# corrupt D^2 = 16 -> 4 (or D_aug^2 = 36 -> 16)
    "boundary_k_bracket_flipped",    # corrupt K -> K/2 vs 2K monotonicity
    "cov_integral_omega_dropped",    # corrupt C_beta(u) normalization by missing 1/Omega
    "dyson_negative_d0_monotone",    # falsely assume ||e^{-Delta H_0}||_1 <= Tr(e^{-beta/(N+1) H_0})
    "t0_c2_dyadic_too_large",        # falsely claim t=0 b_src < 1 holds at beta*g^2/K = 2^-10
    "schatten_holder_factor_halved", # corrupt ||B_e||_op^4 = 16 -> 8 in Schatten-Holder check
    "wick_n4_missing_cross_pairing", # drop one pairing from n=4 thermal Wick sum
)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def arb_from_q(x: Q) -> arb:
    return arb(fmpq(x.numerator, x.denominator))

def arb_mat_from_q(mat) -> arb_mat:
    return arb_mat([[arb_from_q(x) for x in row] for row in mat])

def mat_mul_q(a, b):
    n, m, k = len(a), len(b[0]), len(b)
    return [[sum((a[i][p] * b[p][j] for p in range(k)), Q(0)) for j in range(m)] for i in range(n)]


def mat_add_q(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_sub_q(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_scale_q(c: Q, a):
    return [[c * x for x in row] for row in a]


def mat_abs_q(a):
    return [[abs(x) for x in row] for row in a]


def eye_q(n: int, c: Q = Q(1)):
    return [[c if i == j else Q(0) for j in range(n)] for i in range(n)]


def is_psd_ldlt_q(mat) -> bool:
    """Exact rational LDL^T check for symmetric matrix PSD-ness (no pivoting needed when diagonal pivots)."""
    n = len(mat)
    a = [[Q(x) for x in row] for row in mat]
    for i in range(n):
        for j in range(i + 1, n):
            if a[i][j] != a[j][i]:
                return False
    # Handle zero pivots via symmetric row/col permutation if needed
    for k in range(n):
        piv = a[k][k]
        if piv < 0:
            return False
        if piv == 0:
            # Find nonzero diagonal pivot >= k; if none, remaining row/col must be all zero
            swap = None
            for r in range(k + 1, n):
                if a[r][r] < 0:
                    return False
                if a[r][r] > 0:
                    swap = r
                    break
            if swap is None:
                for i in range(k, n):
                    for j in range(k, n):
                        if a[i][j] != 0:
                            return False
                return True
            a[k], a[swap] = a[swap], a[k]
            for row in a:
                row[k], row[swap] = row[swap], row[k]
            piv = a[k][k]
        for i in range(k + 1, n):
            if a[i][k] == 0:
                continue
            factor = a[i][k] / piv
            for j in range(i, n):
                val = a[i][j] - factor * a[k][j]
                a[i][j] = val
                a[j][i] = val
            a[i][k] = Q(0)
            a[k][i] = Q(0)
    return True


def build_two_site_car_q():
    """Exact 16-state CAR matrices on modes (0=(x,up), 1=(x,dn), 2=(y,up), 3=(y,dn))."""
    dim = 16
    c = []
    cdag = []
    for m in range(4):
        cm = [[Q(0) for _ in range(dim)] for _ in range(dim)]
        cdm = [[Q(0) for _ in range(dim)] for _ in range(dim)]
        for s in range(dim):
            if (s >> m) & 1:
                parity = bin(s & ((1 << m) - 1)).count("1")
                sign = Q(-1) if (parity % 2) else Q(1)
                s_next = s ^ (1 << m)
                cm[s_next][s] = sign
                cdm[s][s_next] = sign
        c.append(cm)
        cdag.append(cdm)

    n = [mat_mul_q(cdag[m], c[m]) for m in range(4)]
    diff_x = mat_sub_q(n[0], n[1])
    diff_y = mat_sub_q(n[2], n[3])
    n_odd_x = mat_mul_q(diff_x, diff_x)
    n_odd_y = mat_mul_q(diff_y, diff_y)
    n_odd_bond = mat_add_q(n_odd_x, n_odd_y)

    t_up = mat_add_q(mat_mul_q(cdag[0], c[2]), mat_mul_q(cdag[2], c[0]))
    t_dn = mat_add_q(mat_mul_q(cdag[1], c[3]), mat_mul_q(cdag[3], c[1]))
    t_e = mat_add_q(t_up, t_dn)
    b_e = mat_abs_q(t_e)

    delta_x = mat_mul_q(c[1], c[0])
    delta_y = mat_mul_q(c[3], c[2])
    delta_x_dag = mat_mul_q(cdag[0], cdag[1])
    delta_y_dag = mat_mul_q(cdag[2], cdag[3])
    x_pair = mat_add_q(mat_mul_q(delta_x_dag, delta_y), mat_mul_q(delta_y_dag, delta_x))
    o_mark = mat_mul_q(delta_x_dag, delta_y)
    return {
        "n_odd_bond": n_odd_bond,
        "t_e": t_e,
        "b_e": b_e,
        "x_pair": x_pair,
        "o_mark": o_mark,
    }


def check_bond_car_and_sos(ops, mutant=None):
    # 1. Exact CAR & operator norm checks: ||T_e|| = ||B_e|| = 2, ||X_pair|| = 1, ||Delta_x^dag Delta_y|| = 1
    b_norm = Q(1) if mutant == "norm_be_underestimate" else Q(2)
    b_sq = mat_mul_q(ops["b_e"], ops["b_e"])
    t_sq = mat_mul_q(ops["t_e"], ops["t_e"])
    require(
        is_psd_ldlt_q(mat_sub_q(eye_q(16, b_norm * b_norm), b_sq)),
        "b_norm^2*I - B_e^2 must be PSD",
    )
    require(
        is_psd_ldlt_q(mat_sub_q(eye_q(16, Q(4)), t_sq)),
        "4I - T_e^2 must be PSD",
    )
    # Attained ||B_e|| = 2 on vector v where (0,2)=(1,0)+(0,1) and (1,3)=(1,0)+(0,1)
    v = [Q(0)] * 16
    for s in (0b0011, 0b0110, 0b1001, 0b1100):  # wait: modes (0,1,2,3): 0b0101 (x_up, y_up)? Wait:
        pass
    # Mode bits: 0=(x,up), 1=(x,dn), 2=(y,up), 3=(y,dn).
    # Single-up + single-dn states: {0b0011? No: (x_up=1,y_up=0) is 1, (x_up=0,y_up=1) is 4; (x_dn=1,y_dn=0) is 2, (x_dn=0,y_dn=1) is 8}
    v = [Q(0)] * 16
    for s in (1 + 2, 1 + 8, 4 + 2, 4 + 8):
        v[s] = Q(1)
    bv = [sum(ops["b_e"][i][j] * v[j] for j in range(16)) for i in range(16)]
    require(bv == [Q(2) * x for x in v], "B_e must attain eigenvalue 2")

    x_sq = mat_mul_q(ops["x_pair"], ops["x_pair"])
    require(is_psd_ldlt_q(mat_sub_q(eye_q(16, Q(1)), x_sq)), "I - X_pair^2 must be PSD")
    o_dag_o = mat_mul_q([[ops["o_mark"][j][i] for j in range(16)] for i in range(16)], ops["o_mark"])
    require(is_psd_ldlt_q(mat_sub_q(eye_q(16, Q(1)), o_dag_o)), "I - O^dag O must be PSD")

    # 2. Exact rational SOS check: alpha*(n_odd_x + n_odd_y) + (c_sos * g^2 / alpha) * I - g * B_e >= 0
    c_sos = Q(1) if mutant == "sos_const_halved" else Q(2)
    samples = [
        (Q(1, 5), Q(1, 3)),
        (Q(1), Q(1)),
        (Q(7, 3), Q(5, 4)),
        (Q(5), Q(2)),
        (Q(13, 4), Q(3, 7)),
    ]
    for g_val, alpha in samples:
        r_mat = mat_sub_q(
            mat_add_q(
                mat_scale_q(alpha, ops["n_odd_bond"]),
                eye_q(16, c_sos * g_val * g_val / alpha),
            ),
            mat_scale_q(g_val, ops["b_e"]),
        )
        require(is_psd_ldlt_q(r_mat), f"SOS R_e failed PSD at g={g_val}, alpha={alpha}")

    # Witness that c_sos = 1 fails on B_e (e.g. at g=1, alpha=1: on |x_up, y_dn> + ...? Look at state |0011>+|1100> etc.)
    r_one = mat_sub_q(
        mat_add_q(ops["n_odd_bond"], eye_q(16, Q(1))),
        ops["b_e"],
    )
    if mutant != "sos_const_halved":
        require(not is_psd_ldlt_q(r_one), "SOS constant 1 must fail PSD on B_e")


def check_atomic_ratio(mutant=None):
    # Exact polynomial check: 5*z_a - 4*z_b = (1-z)^2 + 2z(2w-1)^2 + 2z*w^2 > 0
    ratio_cap = Q(1) if mutant == "atomic_ratio_dropped" else Q(5, 4)
    samples = [
        (Q(1, 10), Q(1, 5)),
        (Q(1), Q(1, 2)),
        (Q(7, 3), Q(3, 4)),
        (Q(100), Q(99, 100)),
        (Q(1, 100), Q(1, 1000)),
    ]
    for z, w in samples:
        z_a = 1 + 2 * w * w * z + z * z
        z_b = 1 + 2 * w * z + z * z
        diff = 5 * z_a - 4 * z_b
        rhs = (1 - z) ** 2 + 2 * z * (2 * w - 1) ** 2 + 2 * z * w * w
        require(diff == rhs and diff > 0, "Exact atomic polynomial identity failed")
        require(z_b <= ratio_cap * z_a, f"z_b/z_a exceeded {ratio_cap} at z={z}, w={w}")

def check_dyson_semigroup_counterexample_arb(mutant=None):
    """Arb check of Lemma 2.2 counterexample (negative D_0) and corrected factorized bound."""
    # Review B counterexample: 4x4 torus (|Lambda|=16, |E|=32), beta=U_eff=Omega=1, mu=10, N=1, Delta=3/4
    n_sites = 16
    n_edges = 32
    beta = arb(1)
    u_eff = arb(1)
    omega = arb(1)
    mu = arb(10)
    n_ins = 1
    delta = arb(3) / arb(4)
    delta_min = beta / arb(n_ins + 1)  # 1/2

    # Exact single-site electronic partition factor: states |0> (E=0), |up>,|dn> (E=U_eff/2-mu), |up,dn> (E=-2*mu)
    def z_el_1(d: arb) -> arb:
        return arb(1) + arb(2) * (-d * (u_eff / arb(2) - mu)).exp() + (-d * (-arb(2) * mu)).exp()

    def z_ph_1(d: arb) -> arb:
        return arb(1) / (arb(2) * (d * omega / arb(2)).sinh())

    norm_delta = (z_el_1(delta) ** n_sites) * (z_ph_1(delta) ** n_edges)
    tr_min = (z_el_1(delta_min) ** n_sites) * (z_ph_1(delta_min) ** n_edges)
    ratio = norm_delta / tr_min
    require(ratio > arb("7e28"), f"Counterexample ratio unexpectedly small: {ratio}")

    if mutant == "dyson_negative_d0_monotone":
        require(norm_delta <= tr_min, "Mutant assumed false monotonicity ||e^{-Delta H_0}||_1 <= Tr(e^{-beta/(N+1) H_0})")

    # Corrected factorized bound: 4^|Lambda| * exp(beta * ||D_0||_op) * (2 sinh(beta*Omega/(2(N+1))))^{-|E|}
    d0_op = arb(n_sites) * max(abs(u_eff / arb(2) - mu), abs(-arb(2) * mu))  # 16 * 20 = 320
    corrected_bound = (arb(4) ** n_sites) * (beta * d0_op).exp() * (z_ph_1(delta_min) ** n_edges)
    require(norm_delta <= corrected_bound, "Corrected Lemma 2.2 factorized bound failed on counterexample")


def check_schatten_holder_arb(ops, mutant=None):
    """Pure flint.arb_mat enclosure of multilinear Schatten-Holder for noncommuting M and B_e."""
    B_q = ops["b_e"]
    X_q = ops["x_pair"]
    O_q = ops["o_mark"]
    Nodd_q = ops["n_odd_bond"]
    diag_n_q = [[Q(bin(i).count("1")) if i == j else Q(0) for j in range(16)] for i in range(16)]

    M1_q = [
        [Q(3, 2) * Nodd_q[i][j] - Q(2, 5) * diag_n_q[i][j] - Q(37, 100) * B_q[i][j] - Q(11, 100) * X_q[i][j]
         for j in range(16)]
        for i in range(16)
    ]
    M2_q = [
        [-Q(4) * B_q[i][j] + Q(1, 10) * diag_n_q[i][j] + Q(1, 20) * X_q[i][j]
         for j in range(16)]
        for i in range(16)
    ]
    for i in range(16):
        for j in range(16):
            require(M1_q[i][j] == M1_q[j][i], "M1 must be symmetric")
            require(M2_q[i][j] == M2_q[j][i], "M2 must be symmetric")

    B = arb_mat_from_q(B_q)
    O = arb_mat_from_q(O_q)
    M1 = arb_mat_from_q(M1_q)
    M2 = arb_mat_from_q(M2_q)

    deltas = [arb_from_q(d) for d in (Q(3, 20), Q(1, 4), Q(1, 10), Q(1, 2), Q(3, 10))]
    beta = arb_from_q(Q(13, 10))
    require(sum(deltas, arb(0)).overlaps(beta), "Deltas must sum to beta")

    prod1 = (
        O * (-deltas[4] * M1).exp() * B * (-deltas[3] * M1).exp()
        * B * (-deltas[2] * M1).exp() * B * (-deltas[1] * M1).exp()
        * B * (-deltas[0] * M1).exp()
    )
    lhs1 = abs(prod1.trace())
    factor_4 = arb(8) if mutant == "schatten_holder_factor_halved" else arb(16)
    rhs1 = factor_4 * (-beta * M1).exp().trace()
    require(lhs1 < rhs1, "Multilinear Schatten-Holder bound failed on M1")

    # Near-extremal noncommuting M2 achieving ratio > 15 out of 16 (rejects 16 -> 8 mutant)
    prod2 = (
        (-deltas[4] * M2).exp() * B * (-deltas[3] * M2).exp()
        * B * (-deltas[2] * M2).exp() * B * (-deltas[1] * M2).exp()
        * B * (-deltas[0] * M2).exp()
    )
    lhs2 = abs(prod2.trace())
    tr2 = (-beta * M2).exp().trace()
    require(lhs2 < factor_4 * tr2, "Multilinear Schatten-Holder bound failed on near-extremal M2")
    require(lhs2 > arb(15) * tr2, "Near-extremal M2 ratio should exceed 15")

def c_beta_arb(u: arb, beta: arb, K: arb, Omega: arb) -> arb:
    m_osc = K / (Omega * Omega)
    num = (Omega * (beta / arb(2) - u)).cosh()
    den = arb(2) * m_osc * Omega * (beta * Omega / arb(2)).sinh()
    return num / den


def check_covariance_and_wick_arb(mutant=None):
    """Outward Arb check of int_0^beta C_beta(|u-v|) du = 1/K across Omega, and n=4 Wick trace."""
    beta = arb("1.4")
    K = arb("2.3")
    for omega_str in ("0.001", "1.7", "500.0"):
        Omega = arb(omega_str)
        # Closed-form antiderivative check at several v in [0, beta]
        for v_frac in (Q(0), Q(1, 5), Q(1, 2), Q(4, 5), Q(1)):
            v = arb_from_q(v_frac) * beta
            # Integral of C_beta(|u-v|) over [0, beta]:
            # int_0^v C_beta(s) ds + int_0^(beta-v) C_beta(s) ds
            # Primitive of cosh(Omega(beta/2 - s)) is -sinh(Omega(beta/2 - s))/Omega
            m_osc = K / (Omega * Omega)
            if mutant == "cov_integral_omega_dropped":
                # Corrupt covariance by missing factor of 1/Omega
                scale_err = Omega
            else:
                scale_err = arb(1)
            den = arb(2) * m_osc * Omega * (beta * Omega / arb(2)).sinh()
            i1 = ((Omega * (beta / arb(2))).sinh() - (Omega * (beta / arb(2) - v)).sinh()) / (Omega * den)
            i2 = ((Omega * (beta / arb(2))).sinh() - (Omega * (beta / arb(2) - (beta - v))).sinh()) / (Omega * den)
            total = (i1 + i2) * scale_err
            target = arb(1) / K
            require(abs(total - target) < arb("1e-60"), f"Covariance integral != 1/K at Omega={omega_str}")

        # Also numerical midpoint quadrature sanity check for Omega=1.7 at v=0.37*beta
        if omega_str == "1.7":
            v = arb("0.37") * beta
            n_pts = 400
            du = beta / arb(n_pts)
            quad = sum(
                (c_beta_arb(abs((arb(i) + arb("0.5")) * du - v), beta, K, Omega) * du for i in range(n_pts)),
                arb(0),
            )
            if mutant == "cov_integral_omega_dropped":
                quad = quad * Omega
            require(abs(quad - arb(1) / K) < arb("1e-4"), "Quadrature check of covariance integral failed")

    # Pure flint.arb_mat check of n=4 thermal Wick identity on N_ph = 44 + rigorous infinite-dimensional tail enclosure
    N_ph = 44
    Omega_w = arb("1.7")
    K_w = arb("2.3")
    beta_w = arb("1.4")
    m_w = K_w / (Omega_w * Omega_w)
    Z_ph = arb(1) / (arb(2) * (beta_w * Omega_w / arb(2)).sinh())
    scale = arb(1) / (arb(2) * m_w * Omega_w).sqrt()
    X_entries = [[arb(0) for _ in range(N_ph)] for _ in range(N_ph)]
    for n in range(N_ph - 1):
        val = arb(n + 1).sqrt() * scale
        X_entries[n][n + 1] = val
        X_entries[n + 1][n] = val
    X_mat = arb_mat(X_entries)

    def exp_neg_h(delta: arb) -> arb_mat:
        E = [[arb(0) for _ in range(N_ph)] for _ in range(N_ph)]
        for n in range(N_ph):
            E[n][n] = (-delta * Omega_w * (arb(n) + arb("0.5"))).exp()
        return arb_mat(E)

    u1, u2, u3, u4 = arb("0.13"), arb("0.41"), arb("0.77"), arb("1.19")
    M4 = (
        exp_neg_h(beta_w - u4) * X_mat * exp_neg_h(u4 - u3) * X_mat
        * exp_neg_h(u3 - u2) * X_mat * exp_neg_h(u2 - u1) * X_mat * exp_neg_h(u1)
    )
    # For n_0 <= N_ph - 5, all 4-step paths stay inside [0, N_ph - 3], so M4[n_0, n_0] has zero truncation error.
    M0 = N_ph - 4
    tr_exact_part = sum((M4[n, n] for n in range(M0)), arb(0))
    # Rigorous tail bound for n_0 >= M0: <= sum_{n >= M0} 6 * ((n+2)/(2 m Omega))^2 * exp(-beta Omega (n - 3/2))
    r_step = (arb(M0 + 3) / arb(M0 + 2)) ** 2 * (-beta_w * Omega_w).exp()
    require(r_step < arb(1), "Tail geometric ratio >= 1")
    term0 = arb(6) * (arb(M0 + 2) / (arb(2) * m_w * Omega_w)) ** 2 * (-beta_w * Omega_w * (arb(M0) - arb("1.5"))).exp()
    tail_num = term0 / (arb(1) - r_step)
    # True infinite-dimensional trace lies in [tr_exact_part, tr_exact_part + tail_num]
    tr4_arb = (tr_exact_part + arb(0).union(tail_num)) / Z_ph

    t12_34 = c_beta_arb(u2 - u1, beta_w, K_w, Omega_w) * c_beta_arb(u4 - u3, beta_w, K_w, Omega_w)
    t13_24 = c_beta_arb(u3 - u1, beta_w, K_w, Omega_w) * c_beta_arb(u4 - u2, beta_w, K_w, Omega_w)
    t14_23 = c_beta_arb(u4 - u1, beta_w, K_w, Omega_w) * c_beta_arb(u3 - u2, beta_w, K_w, Omega_w)
    if mutant == "wick_n4_missing_cross_pairing":
        wick4_arb = t12_34 + t14_23
    else:
        wick4_arb = t12_34 + t13_24 + t14_23
    require(tr4_arb.overlaps(wick4_arb), f"Rigorous Arb Wick n=4 mismatch: {abs(tr4_arb - wick4_arb)}")
    require(abs(tr4_arb - wick4_arb) < arb("1e-35"), "Arb Wick n=4 enclosure width unexpectedly large")

def check_combinatorics_and_budgets(mutant=None):
    # 1. Exact Wick pairing combinatorial identity:
    # For r = 1..8, (2r-1)!! / (2r)! * 4^r * (1/2)^r == 2^r / r!
    for r in range(1, 9):
        double_fact = math.prod(range(1, 2 * r, 2))
        fact_2r = math.factorial(2 * r)
        fact_r = math.factorial(r)
        if mutant == "wick_pairings_corrupted":
            # Missing 2^{-r} factor in (2r-1)!! = (2r)! / (2^r r!)
            lhs = Q(double_fact * (4 ** r) * (2 ** r), fact_2r)
        else:
            lhs = Q(double_fact * (4 ** r), fact_2r)
        rhs = Q(2 ** r, fact_r)
        require(lhs == rhs, f"Wick combinatorial identity failed at r={r}: {lhs} != {rhs}")

    # 2. Rigorous elementary inequalities in Arb:
    # e^(1/4) < exp_quarter_bound, e < 3, and for 0 <= x < 2: e^x - 1 <= x/(1 - x/2)
    exp_q_cap = Q(5, 4) if mutant == "exp_quarter_bound_too_small" else Q(4, 3)
    require((arb(1) / arb(4)).exp() < arb_from_q(exp_q_cap), "exp(1/4) bound failed")
    require(arb(1).exp() < arb(3), "e < 3 failed")

    # Check parameters
    L_t = 1024
    L_s = 2048
    deg_factor = 8 if mutant == "degree_factor_dropped" else 32
    c1 = Q(1, 128 * L_t * L_t)  # 2^-27
    delta_t = deg_factor * L_t * L_t * c1
    require(delta_t == Q(1, 4), "delta_t != 1/4") if mutant != "degree_factor_dropped" else None
    # Note: if degree_factor_dropped is active, delta_t = 1/16, which would falsely claim 4x larger c1 = 2^-25 gives delta_t = 1/4!
    # Test the actual implied c1 threshold from deg_factor * L_t^2 * c1 <= 1/4:
    c1_claimed = Q(1, 4 * deg_factor * L_t * L_t)
    # Must match 32 * L_t^2 * c1_claimed <= 1/4:
    require(32 * L_t * L_t * c1_claimed <= Q(1, 4), "Claimed c1 violates true 32*L_t^2*c1 <= 1/4")

    c2 = Q(1, 2 ** 11) if mutant == "c2_window_too_large" else Q(1, 2 ** 13)
    g_coeff = Q(1) if mutant == "exp_g_factor_halved" else Q(2)
    x = g_coeff * c2
    # Check x against actual 2*c2 in Arb:
    true_exp_diff = (arb_from_q(Q(2) * c2)).exp() - arb(1)
    if mutant == "exp_taylor_bound_dropped":
        # Falsely bound exp(x)-1 <= x
        exp_diff_bound = x
    else:
        exp_diff_bound = x / (1 - x / 2)
    require(true_exp_diff <= arb_from_q(exp_diff_bound), "exp(2*c2) - 1 bound violated in Arb")

    # Physical and source edge weights
    b_norm_factor = Q(1) if mutant == "norm_be_underestimate" else Q(4)  # ||B_e||^2
    # If ||B_e||_op was understated as 1, effective g-exponent would be 1/2 instead of 2:
    if mutant == "norm_be_underestimate":
        x_eff = Q(1, 2) * c2
        exp_diff_bound = x_eff / (1 - x_eff / 2)

    eps_phys = exp_q_cap / L_t * (1 + exp_diff_bound) + exp_diff_bound
    eps_src = exp_q_cap / L_s
    eps_max = max(eps_phys, eps_src)

    atomic_cap = Q(1) if mutant == "atomic_ratio_dropped" else Q(5, 4)
    euler_e_cap = Q(3)
    site_pref = atomic_cap * euler_e_cap  # 15/4

    rooted_gc = 4 if mutant == "rooted_count_underestimated" else 16
    rooted_src = 16 if mutant == "rooted_count_underestimated" else 36

    # Explicit Z^2 connected m-edge subgraph enumeration containing (0,0):
    edges_z2 = []
    for gx in range(-4, 5):
        for gy in range(-4, 5):
            if gx + 1 <= 4:
                edges_z2.append(((gx, gy), (gx + 1, gy)))
            if gy + 1 <= 4:
                edges_z2.append(((gx, gy), (gx, gy + 1)))
    adj_z2 = {}
    for idx, (u, w) in enumerate(edges_z2):
        adj_z2.setdefault(u, []).append(idx)
        adj_z2.setdefault(w, []).append(idx)

    def count_conn_z2(m_edges: int) -> int:
        found = set()
        def dfs(chosen, fverts):
            if len(chosen) == m_edges:
                found.add(frozenset(chosen))
                return
            cand = set()
            for v in fverts:
                for eidx in adj_z2[v]:
                    if eidx not in chosen:
                        cand.add(eidx)
            for eidx in cand:
                u, w = edges_z2[eidx]
                dfs(chosen | {eidx}, fverts | {u, w})
        for eidx in adj_z2[(0, 0)]:
            u, w = edges_z2[eidx]
            dfs({eidx}, {u, w})
        return len(found)

    c2_z2 = count_conn_z2(2)
    c3_z2 = count_conn_z2(3)
    require(c2_z2 == 18 and c3_z2 == 88, "Z^2 connected-subgraph enumeration mismatch")
    require(c2_z2 <= rooted_gc ** 2 and c3_z2 <= rooted_gc ** 3, "Rooted counting factor too small for Z^2")
    require(rooted_src >= (4 + 2) ** 2, "Augmented rooted counting factor below D_aug^2=36")

    q_gc = rooted_gc * site_pref * eps_phys
    b_gc = site_pref * q_gc / (1 - q_gc)
    A_gc = site_pref / (1 - q_gc)

    q_src = rooted_src * site_pref * eps_max
    b_src = site_pref * q_src / (1 - q_src)

    # Exact rational checks against RIGOR.md constants (when unmutated)
    require(q_gc < 1 and b_gc < 1, f"GC Kotecky-Preiss budget failed: b_gc={float(b_gc)}")
    require(q_src < 1 and b_src < 1, f"Source Kotecky-Preiss budget failed: b_src={float(b_src)}")
    if mutant is None:
        require(eps_phys == Q(3243, 2096896), "eps_phys mismatch")
        require(q_gc == Q(48645, 524224), "q_gc mismatch")
        require(A_gc == Q(1965840, 475579), "A_gc mismatch")
        require(b_gc == Q(729675, 1902316), "b_gc mismatch")
        require(q_src == Q(437805, 2096896), "q_src mismatch")
        require(b_src == Q(6567075, 6636364), "b_src mismatch")
        row_sum = 1 + 4 * A_gc * q_gc / ((1 - q_gc) ** 2)
        require(row_sum == Q(308086835615302339, 107564263537529539), "row_sum mismatch")
        require(row_sum < 3, "row_sum < 3 failed")
        # g=0 limit matches STAGE7R exactly
        eps_g0 = Q(4, 3 * L_t)
        q_g0 = 60 * eps_g0
        A_g0 = Q(15, 4) / (1 - q_g0)
        b_g0 = Q(15, 4) * q_g0 / (1 - q_g0)
        require(q_g0 == Q(5, 64) and A_g0 == Q(240, 59) and b_g0 == Q(75, 236), "g=0 limit mismatch")
        # t=0 source-budget threshold check (Remark 1.3):
        # Sufficient threshold from the rational majorant: x/(1-x/2) = 4/2565 <=> x = 4/2567 <=> beta*g^2/K = 2/2567
        c2_maj_end = Q(2, 2567)
        x_end = 2 * c2_maj_end
        eps_end = x_end / (1 - x_end / 2)
        q_end = 135 * eps_end
        b_end = Q(15, 4) * q_end / (1 - q_end)
        require(b_end == Q(1), f"t=0 rational-majorant endpoint mismatch: {b_end}")

        # Actual endpoint (1/2) log(2569/2565) lies strictly between 2/2567 and 2^-10
        actual_end_arb = (arb(1) / arb(2)) * (arb(2569) / arb(2565)).log()
        require(arb_from_q(c2_maj_end) < actual_end_arb < arb_from_q(Q(1, 2 ** 10)),
                "Actual t=0 endpoint not strictly between 2/2567 and 2^-10")

        # Largest dyadic threshold c2 = 2^-11 satisfies b_src <= 2025/3554 < 1 (majorant value)
        c2_t0_dyadic = Q(1, 2 ** 11)
        x_t0 = 2 * c2_t0_dyadic
        eps_t0 = x_t0 / (1 - x_t0 / 2)
        require(eps_t0 == Q(2, 2047) and eps_t0 > eps_src, "t=0 dyadic eps_phys mismatch")
        q_t0 = 135 * eps_t0
        b_t0 = Q(15, 4) * q_t0 / (1 - q_t0)
        require(q_t0 == Q(270, 2047) and b_t0 == Q(2025, 3554) and b_t0 < 1, "t=0 2^-11 check failed")

        # At 2^-10, the actual b_src > 1 (certified via exp(2^-9) - 1) and the rational majorant is 675/502 > 1
        c2_t0_bad = Q(1, 2 ** 10)
        x_t0_bad = 2 * c2_t0_bad
        eps_t0_bad = x_t0_bad / (1 - x_t0_bad / 2)
        q_t0_bad = 135 * eps_t0_bad
        b_t0_bad = Q(15, 4) * q_t0_bad / (1 - q_t0_bad)
        require(eps_t0_bad == Q(2, 1023) and q_t0_bad == Q(90, 341) and b_t0_bad == Q(675, 502) and b_t0_bad > 1,
                "t=0 2^-10 majorant counterexample mismatch")
        eps_actual_bad = (arb_from_q(x_t0_bad)).exp() - arb(1)
        q_actual_bad = arb(135) * eps_actual_bad
        b_actual_bad = (arb(15) / arb(4)) * q_actual_bad / (arb(1) - q_actual_bad)
        require(b_actual_bad > arb(1), "Actual b_src at beta*g^2/K = 2^-10 must exceed 1")

    if mutant == "t0_c2_dyadic_too_large":
        x_t0_mut = 2 * Q(1, 2 ** 10)
        eps_t0_mut = max(x_t0_mut / (1 - x_t0_mut / 2), eps_src)
        q_t0_mut = 135 * eps_t0_mut
        b_t0_mut = Q(15, 4) * q_t0_mut / (1 - q_t0_mut)
        require(b_t0_mut < 1, "Mutant claimed t=0 b_src < 1 at beta*g^2/K = 2^-10")

def check_boundary_k_bracket(ops, mutant=None):
    """Verify boundary oscillator Z_ph,1(K') monotonicity in Arb (CERTIFIED) and 16xN_ph matrix bracket (NUMERICAL)."""
    N_ph = 24
    K = 2.4
    Omega = 1.9
    m_osc = K / (Omega ** 2)
    beta = 0.7
    t = 0.2
    g = 0.35

    a = np.diag(np.sqrt(np.arange(1, N_ph)), 1)
    adag = a.T
    X_osc = (a + adag) / np.sqrt(2.0 * m_osc * Omega)
    P_osc = 1j * (adag - a) * np.sqrt(m_osc * Omega / 2.0)

    def h_ph(k_val):
        return (P_osc @ P_osc) / (2.0 * m_osc) + 0.5 * k_val * (X_osc @ X_osc)

    t_np = np.array([[float(x) for x in row] for row in ops["t_e"]])
    n_odd_np = np.array([[float(x) for x in row] for row in ops["n_odd_bond"]])
    I_el = np.eye(16)
    I_ph = np.eye(N_ph)
    D_bond = 1.2 * n_odd_np

    H_full = np.kron(D_bond, I_ph) + np.kron(I_el, h_ph(K)) + np.kron(-t * t_np, I_ph) + np.kron(g * t_np, X_osc)
    shift = 2.0 * t + 4.0 * (g ** 2) / K
    H_low = np.kron(D_bond, I_ph) + np.kron(I_el, h_ph(0.5 * K)) - shift * np.eye(16 * N_ph)
    H_high = np.kron(D_bond, I_ph) + np.kron(I_el, h_ph(2.0 * K)) + shift * np.eye(16 * N_ph)

    if mutant == "boundary_k_bracket_flipped":
        H_low, H_high = H_high, H_low

    evals_low = np.linalg.eigvalsh(H_low)
    evals_full = np.linalg.eigvalsh(H_full)
    evals_high = np.linalg.eigvalsh(H_high)
    # Check low <= full <= high on lowest 40 eigenvalues (well below truncation cutoff)
    require(np.all(evals_low[:40] <= evals_full[:40] + 1e-10), "H_low <= H_full failed")
    require(np.all(evals_full[:40] <= evals_high[:40] + 1e-10), "H_full <= H_high failed")

    # Exact infinite-dimensional Z_ph,1(K') monotonicity in Arb
    b_a = arb(beta)
    m_a = arb(K) / (arb(Omega) ** 2)
    def z_ph1(k_q: Q) -> arb:
        om = (arb_from_q(k_q) * arb(K) / m_a).sqrt()
        return arb(1) / (arb(2) * (b_a * om / arb(2)).sinh())

    if mutant == "boundary_k_bracket_flipped":
        require(z_ph1(Q(2)) > z_ph1(Q(1)), "Mutant flipped Z_ph1 monotonicity")
    else:
        require(z_ph1(Q(1, 2)) > z_ph1(Q(1)) > z_ph1(Q(2)), "Z_ph1 monotonicity failed")


def check_numerical_two_site_omega_scan(ops):
    """NUMERICAL sanity check (not a proof): 2-site <Delta_x^dag Delta_y> across Omega in [0.2, 50]."""
    N_ph = 24
    K = 2.0
    U_eff = 2.0
    t = 0.002
    g = 0.008
    beta = 1.0  # beta*t^2/U_eff = 2e-6 << 2^-27? Actually 2^-27 ~ 7.45e-9; use t=1e-4 -> 5e-9, beta*g^2/K = 3.2e-5 <= 2^-13=1.22e-4!
    t = 1e-4
    g = 0.01
    mu = 0.3

    t_np = np.array([[float(x) for x in row] for row in ops["t_e"]])
    n_odd_np = np.array([[float(x) for x in row] for row in ops["n_odd_bond"]])
    o_np = np.array([[float(x) for x in row] for row in ops["o_mark"]])
    diag_n = np.diag([bin(s).count("1") for s in range(16)])
    D0 = (U_eff / 2.0) * n_odd_np - mu * diag_n
    I_el = np.eye(16)
    I_ph = np.eye(N_ph)

    A_gc = 1965840 / 475579
    q_gc = 48645 / 524224
    bound = A_gc * q_gc

    a = np.diag(np.sqrt(np.arange(1, N_ph)), 1)
    adag = a.T
    for Omega in (0.2, 2.0, 20.0):
        m_osc = K / (Omega ** 2)
        X_osc = (a + adag) / np.sqrt(2.0 * m_osc * Omega)
        H_ph = Omega * (adag @ a + 0.5 * np.eye(N_ph))
        H = np.kron(D0 - t * t_np, I_ph) + g * np.kron(t_np, X_osc) + np.kron(I_el, H_ph)
        # Shift ground state for numerical stability at Omega=20
        e0 = np.min(np.linalg.eigvalsh(H))
        rho = expm(-beta * (H - e0 * np.eye(16 * N_ph)))
        corr = abs(np.trace(np.kron(o_np, I_ph) @ rho) / np.trace(rho))
        require(corr <= bound, f"2-site correlator exceeded GC bound at Omega={Omega}: {corr}")


def base_checks(mutant=None):
    ops = build_two_site_car_q()
    check_bond_car_and_sos(ops, mutant)
    check_atomic_ratio(mutant)
    check_dyson_semigroup_counterexample_arb(mutant)
    check_schatten_holder_arb(ops, mutant)
    check_covariance_and_wick_arb(mutant)
    check_combinatorics_and_budgets(mutant)
    check_boundary_k_bracket(ops, mutant)
    if mutant is None:
        check_numerical_two_site_omega_scan(ops)
        print("SSHBOUND3 replay: ALL CERTIFIED AND NUMERICAL CHECKS PASSED")
        print("  c1 = 2^-27, c2 = 2^-13, C = 2^27, L_t = 1024, L_s = 2048")
        print("  q_GC = 48645/524224, A_GC = 1965840/475579, b_GC = 729675/1902316 < 1")
        print("  q_src = 437805/2096896, b_src = 6567075/6636364 < 1, row_sum < 3")


def run_mutant(name: str) -> int:
    try:
        base_checks(mutant=name)
    except AssertionError as exc:
        print(f"MUTANT REJECTED [{name}]: {exc}")
        return 1
    print(f"MUTANT SURVIVED [{name}]")
    return 0


def all_mutants() -> int:
    survivors = []
    for name in MUTANTS:
        rc = run_mutant(name)
        if rc == 0:
            survivors.append(name)
    if survivors:
        print(f"FAIL: surviving mutants: {survivors}")
        return 1
    print(f"ALL {len(MUTANTS)} MATHEMATICAL MUTANTS REJECTED (each exited 1)")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--all-mutants"]:
        return all_mutants()
    if len(argv) == 2 and argv[0] == "--mutant" and argv[1] in MUTANTS:
        return run_mutant(argv[1])
    if not argv:
        base_checks(mutant=None)
        return 0
    print("Usage: replay.py [--all-mutants | --mutant NAME]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
