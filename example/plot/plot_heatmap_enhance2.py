import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"]
plt.rcParams["text.usetex"] = True


def generalized_harmonic(n: int, s: float) -> float:
    """
    Calculate generalized harmonic number H_{n,s} = sum_{j=1}^{n} j^{-s}
    """
    return np.sum([j**(-s) for j in range(1, n + 1)])


def find_r(A: int, s: float, rho: float) -> int:
    """
    Find r = argmin_i such that H_{i,s} / H_{|A|,s} >= rho
    """
    H_A_s = generalized_harmonic(A, s)
    for i in range(1, A + 1):
        H_i_s = generalized_harmonic(i, s)
        if H_i_s / H_A_s >= rho:
            return i
    return A


def lower_bound_enhanced_s1(A: int, s: float, m_bar: float, epsilon: float, rho: float) -> float:
    """
    Revised lower bound (Phi-form, valid for s >= 0 under theorem conditions):
    P_c >= exp(-mu/H * Phi_s(B) - mu^2/H^2 * Phi_{2s}(B)) * (1 - mu*rho*(1-epsilon))
    where B = 2*mu*(1-epsilon)/(1+epsilon) and r = argmin_i H_{i,s}/H_{A,s} >= rho.

    Phi_q(x) = ((r+x-1)^{1-q} - (r-1)^{1-q})/(1-q), q != 1
             = ln((r+x-1)/(r-1)), q = 1
    """
    if s < 0:
        return np.nan

    H_A_s = generalized_harmonic(A, s)
    if m_bar <= 0 or m_bar > H_A_s:
        return np.nan

    if epsilon <= -1:
        return np.nan

    r_pivot = find_r(A, s, rho)
    # Approach A: use an effective pivot r_eff = max(2, r_pivot) so that all
    # (r-1)-based terms are well-defined (e.g., Phi_q), while preserving
    # H_{r_eff,s}/H_{A,s} >= rho because r_eff >= r_pivot.
    r = max(2, int(r_pivot))

    mult_term = 1 - m_bar * rho * (1 - epsilon)
    mult_term = max(0.0, float(mult_term))
    if mult_term == 0.0:
        return 0.0

    # Expected symmetric-difference size under J >= epsilon.
    B = (2 * m_bar * (1 - epsilon)) / (1 + epsilon)
    B = max(0.0, B)

    def _phi_q(q: float, x: float, r0: int) -> float:
        if x <= 0:
            return 0.0
        r_float = float(r0)
        x_float = float(x)
        if np.isclose(q, 1.0):
            return np.log((r_float + x_float - 1.0) / (r_float - 1.0))
        return (((r_float + x_float - 1.0) ** (1.0 - q)) - ((r_float - 1.0) ** (1.0 - q))) / (1.0 - q)

    phi_s = _phi_q(s, B, r)
    phi_2s = _phi_q(2 * s, B, r)

    term1 = -(m_bar / H_A_s) * phi_s
    term2 = -((m_bar**2) / (H_A_s**2)) * phi_2s
    exp_term = np.exp(term1 + term2)

    return float(exp_term * mult_term)


def lower_bound_with_validity(A: int, s: float, m_bar: float, epsilon: float, rho: float) -> tuple[float, bool]:
    """
    Return (bound, is_valid) where is_valid indicates theorem conditions are met.
    """
    if s < 0:
        return np.nan, False

    H_A_s = generalized_harmonic(A, s)
    if m_bar <= 0 or m_bar > H_A_s:
        return np.nan, False

    if not (0 <= epsilon <= 1):
        return np.nan, False

    r_pivot = find_r(A, s, rho)
    r = max(2, int(r_pivot))

    mult_term = 1 - m_bar * rho * (1 - epsilon)
    mult_term = max(0.0, float(mult_term))
    if mult_term == 0.0:
        return 0.0, True
    p_r = (m_bar * (r ** (-s))) / H_A_s
    # The proof assumes low-probability tail labels; monotonicity implies checking p_r is sufficient.
    if p_r > 0.6:
        return np.nan, False

    B = (2 * m_bar * (1 - epsilon)) / (1 + epsilon)
    B = max(0.0, B)

    def _phi_q(q: float, x: float, r0: int) -> float:
        if x <= 0:
            return 0.0
        r_float = float(r0)
        x_float = float(x)
        if np.isclose(q, 1.0):
            return np.log((r_float + x_float - 1.0) / (r_float - 1.0))
        return (((r_float + x_float - 1.0) ** (1.0 - q)) - ((r_float - 1.0) ** (1.0 - q))) / (1.0 - q)

    phi_s = _phi_q(s, B, r)
    phi_2s = _phi_q(2 * s, B, r)
    if not (np.isfinite(phi_s) and np.isfinite(phi_2s)):
        return np.nan, False
    if phi_s < 0 or phi_2s < 0:
        return np.nan, False

    bound = lower_bound_enhanced_s1(A, s, m_bar, epsilon, rho)
    if not np.isfinite(bound) or bound < 0 or bound > 1:
        return np.nan, False
    return bound, True


def plot_four_heatmaps(A_fixed: int = 50, s_fixed: float = 1.0,
                       epsilon_fixed: float = 0.8, rho_fixed: float = 0.8,
                       m_bar_ratio_fixed: float = 0.5,
                       output_file: str = 'heatmaps_enhance2.pdf'):
    """
    Create a 2x2 figure with four heatmaps showing hyperparameter analysis.
    Uses the revised Phi-form lower bound and validity masking.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.44, wspace=0.22)

    def _nearest_index(values: np.ndarray, target: float) -> int:
        values = np.asarray(values, dtype=float)
        return int(np.argmin(np.abs(values - target)))

    def _safe_minmax(arr: np.ndarray) -> tuple[float, float]:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return np.nan, np.nan
        return float(np.min(valid)), float(np.max(valid))

    def _diagnose_invalid(A: int, s: float, m_bar: float, epsilon: float, rho: float) -> list[str]:
        reasons: list[str] = []
        if s < 0:
            reasons.append(f"s={s:.4g}<0")
            return reasons

        if not (0 <= epsilon <= 1):
            reasons.append(f"epsilon={epsilon:.4g} not in [0,1]")

        if not (0 < rho < 1):
            reasons.append(f"rho={rho:.4g} not in (0,1)")

        H_A_s = generalized_harmonic(A, s)
        if not (m_bar > 0):
            reasons.append(f"mu={m_bar:.4g} <= 0")
        if m_bar > H_A_s + 1e-12:
            reasons.append(f"mu={m_bar:.4g} > H_A_s={H_A_s:.4g}")

        r_pivot = find_r(A, s, rho)
        if r_pivot < 1 or r_pivot > A:
            reasons.append(f"r_pivot={r_pivot} not in [1,A]")
            return reasons

        r = max(2, int(r_pivot))
        if r_pivot < 2:
            reasons.append(f"r_pivot={r_pivot} < 2; using r_eff={r} (Approach A)")

        if H_A_s <= 0 or not np.isfinite(H_A_s):
            reasons.append(f"H_A_s={H_A_s} not finite/positive")
            return reasons

        # Tail low-probability condition: for i>=r, p_i is decreasing, so check p_r.
        p_r = (m_bar * (r ** (-s))) / H_A_s
        if not np.isfinite(p_r):
            reasons.append(f"p_r={p_r} not finite")
        elif p_r > 0.6:
            reasons.append(f"tail condition violated: p_r={p_r:.4g} > 0.6")

        if epsilon > -1:
            B = (2 * m_bar * (1 - epsilon)) / (1 + epsilon)
        else:
            B = np.nan
        if not np.isfinite(B):
            reasons.append(f"B={B} not finite")
            return reasons
        if B < 0:
            reasons.append(f"B={B:.4g} < 0")

        def _phi_q(q: float, x: float, r0: int) -> float:
            if x <= 0:
                return 0.0
            r_float = float(r0)
            x_float = float(x)
            if np.isclose(q, 1.0):
                return np.log((r_float + x_float - 1.0) / (r_float - 1.0))
            return (((r_float + x_float - 1.0) ** (1.0 - q)) - ((r_float - 1.0) ** (1.0 - q))) / (1.0 - q)

        phi_s = _phi_q(s, max(0.0, B), r)
        phi_2s = _phi_q(2 * s, max(0.0, B), r)
        if not np.isfinite(phi_s):
            reasons.append(f"Phi_s(B)={phi_s} not finite")
        elif phi_s < 0:
            reasons.append(f"Phi_s(B)={phi_s:.4g} < 0")
        if not np.isfinite(phi_2s):
            reasons.append(f"Phi_2s(B)={phi_2s} not finite")
        elif phi_2s < 0:
            reasons.append(f"Phi_2s(B)={phi_2s:.4g} < 0")

        mult_term = 1 - m_bar * rho * (1 - epsilon)
        if mult_term <= 0:
            reasons.append(f"probability factor clipped: 1-mu*rho*(1-eps)={mult_term:.4g} <= 0 -> max(0,.)")

        bound = lower_bound_enhanced_s1(A, s, m_bar, epsilon, rho)
        if not np.isfinite(bound):
            reasons.append(f"bound={bound} not finite")
        elif bound < 0 or bound > 1:
            reasons.append(f"bound={bound:.4g} not in [0,1]")

        if len(reasons) == 0:
            reasons.append("masked as invalid (unspecified)")
        return reasons

    # Top-left: Heatmap s vs |A|
    ax1 = axes[0, 0]
    A_values = np.arange(10, 1000, 10)
    s_values = np.linspace(1.0, 3.0, 41)

    A_grid, s_grid = np.meshgrid(A_values, s_values)
    bounds1 = np.zeros_like(A_grid, dtype=float)

    for i in range(len(s_values)):
        for j in range(len(A_values)):
            A_val = int(A_grid[i, j])
            s_val = s_grid[i, j]
            H_A_s_val = generalized_harmonic(A_val, s_val)
            m_bar = m_bar_ratio_fixed * H_A_s_val
            bound_val, is_valid = lower_bound_with_validity(A_val, s_val, m_bar, epsilon_fixed, rho_fixed)
            bounds1[i, j] = bound_val if is_valid else np.nan

    contour1 = ax1.contourf(A_grid, s_grid, bounds1, levels=20, cmap='viridis')
    cbar1 = fig.colorbar(contour1, ax=ax1, pad=0.02)
    cbar1.ax.tick_params(labelsize=18)
    cbar1.set_label('Lower Bound of $P_c$', fontsize=24)
    ax1.set_xlabel('$|A|$ (Number of Labels)', fontsize=28)
    ax1.set_ylabel('$s$ (Skewness Factor)', fontsize=28)
    s_ticks = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    ax1.set_yticks(s_ticks)
    ax1.set_title(f'(a) $s$ vs $|A|$ ($\\mu$={m_bar_ratio_fixed}$H_{{|A|,s}}$, $\\epsilon$={epsilon_fixed})', fontsize=30)
    ax1.tick_params(axis='both', labelsize=26)

    # Print gridded (1D sweep) sample values for plot (a)
    print("\nPlot (a) - s vs |A| sample values (1D sweeps on grid):")
    A_fix = 100
    A_fix_idx = _nearest_index(A_values, A_fix)
    print(f"  Fix |A|={A_values[A_fix_idx]}: sweep s")
    for s_rep in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        s_idx = _nearest_index(s_values, s_rep)
        s_val = float(s_values[s_idx])
        A_val = int(A_values[A_fix_idx])
        bound_val = bounds1[s_idx, A_fix_idx]
        if not np.isfinite(bound_val):
            reasons = _diagnose_invalid(A_val, s_val, m_bar_ratio_fixed * generalized_harmonic(A_val, s_val), epsilon_fixed, rho_fixed)
            print(f"    s={s_val:.3g}: invalid -> " + "; ".join(reasons))
            continue
        H_A_s = generalized_harmonic(A_val, s_val)
        m_bar = m_bar_ratio_fixed * H_A_s
        r_pivot = find_r(A_val, s_val, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_val))) / H_A_s
        print(f"    s={s_val:.3g}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    s_fix = 1.0
    s_fix_idx = _nearest_index(s_values, s_fix)
    print(f"  Fix s={float(s_values[s_fix_idx]):.3g}: sweep |A|")
    for A_rep in [50, 100, 200, 500, 900]:
        A_idx = _nearest_index(A_values, A_rep)
        A_val = int(A_values[A_idx])
        s_val = float(s_values[s_fix_idx])
        bound_val = bounds1[s_fix_idx, A_idx]
        if not np.isfinite(bound_val):
            reasons = _diagnose_invalid(A_val, s_val, m_bar_ratio_fixed * generalized_harmonic(A_val, s_val), epsilon_fixed, rho_fixed)
            print(f"    |A|={A_val}: invalid -> " + "; ".join(reasons))
            continue
        H_A_s = generalized_harmonic(A_val, s_val)
        m_bar = m_bar_ratio_fixed * H_A_s
        r_pivot = find_r(A_val, s_val, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_val))) / H_A_s
        print(f"    |A|={A_val}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    min1, max1 = _safe_minmax(bounds1)
    print(f"  Min bound: {min1:.4f}, Max bound: {max1:.4f}")

    # Top-right: Heatmap s vs epsilon
    ax2 = axes[0, 1]
    epsilon_grid2, s_grid2 = np.meshgrid(
        np.linspace(0.6, 1.0, 30),
        np.linspace(1.0, 3.0, 41)
    )
    bounds2 = np.zeros_like(epsilon_grid2)

    for i in range(s_grid2.shape[0]):
        for j in range(epsilon_grid2.shape[1]):
            s_val2 = s_grid2[i, j]
            H_A_s2 = generalized_harmonic(A_fixed, s_val2)
            m_bar2 = m_bar_ratio_fixed * H_A_s2
            bound_val, is_valid = lower_bound_with_validity(A_fixed, s_val2, m_bar2, epsilon_grid2[i, j], rho_fixed)
            bounds2[i, j] = bound_val if is_valid else np.nan

    contour2 = ax2.contourf(epsilon_grid2, s_grid2, bounds2, levels=20, cmap='viridis')
    cbar2 = fig.colorbar(contour2, ax=ax2, pad=0.02)
    cbar2.ax.tick_params(labelsize=18)
    cbar2.set_label('Lower Bound of $P_c$', fontsize=24)
    ax2.set_xlabel('$\\epsilon$ (Jaccard Similarity)', fontsize=28)
    ax2.set_ylabel('$s$ (Skewness Factor)', fontsize=28)
    ax2.set_yticks(s_ticks)
    ax2.set_title(f'(b) $s$ vs $\\epsilon$ ($|A|$={A_fixed}, $\\mu$={m_bar_ratio_fixed}$H_{{|A|,s}}$)', fontsize=30)
    ax2.tick_params(axis='both', labelsize=26)

    # Print gridded (1D sweep) sample values for plot (b)
    print("\nPlot (b) - s vs \\epsilon sample values (1D sweeps on grid):")
    epsilon_values_b = np.linspace(0.6, 1.0, 30)
    s_values_b = np.linspace(1.0, 3.0, 41)

    eps_fix = 0.8
    eps_fix_idx = _nearest_index(epsilon_values_b, eps_fix)
    print(f"  Fix \\epsilon={float(epsilon_values_b[eps_fix_idx]):.3g}: sweep s")
    for s_rep in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        s_idx = _nearest_index(s_values_b, s_rep)
        s_val = float(s_values_b[s_idx])
        eps_val = float(epsilon_values_b[eps_fix_idx])
        bound_val = bounds2[s_idx, eps_fix_idx]
        if not np.isfinite(bound_val):
            reasons = _diagnose_invalid(A_fixed, s_val, m_bar_ratio_fixed * generalized_harmonic(A_fixed, s_val), eps_val, rho_fixed)
            print(f"    s={s_val:.3g}: invalid -> " + "; ".join(reasons))
            continue
        H_A_s = generalized_harmonic(A_fixed, s_val)
        m_bar = m_bar_ratio_fixed * H_A_s
        r_pivot = find_r(A_fixed, s_val, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_val))) / H_A_s
        print(f"    s={s_val:.3g}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    s_fix = 1.0
    s_fix_idx = _nearest_index(s_values_b, s_fix)
    print(f"  Fix s={float(s_values_b[s_fix_idx]):.3g}: sweep \\epsilon")
    for eps_rep in [0.6, 0.7, 0.8, 0.9, 1.0]:
        eps_idx = _nearest_index(epsilon_values_b, eps_rep)
        eps_val = float(epsilon_values_b[eps_idx])
        s_val = float(s_values_b[s_fix_idx])
        bound_val = bounds2[s_fix_idx, eps_idx]
        if not np.isfinite(bound_val):
            reasons = _diagnose_invalid(A_fixed, s_val, m_bar_ratio_fixed * generalized_harmonic(A_fixed, s_val), eps_val, rho_fixed)
            print(f"    \\epsilon={eps_val:.3g}: invalid -> " + "; ".join(reasons))
            continue
        H_A_s = generalized_harmonic(A_fixed, s_val)
        m_bar = m_bar_ratio_fixed * H_A_s
        r_pivot = find_r(A_fixed, s_val, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_val))) / H_A_s
        print(f"    \\epsilon={eps_val:.3g}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    min2, max2 = _safe_minmax(bounds2)
    print(f"  Min bound: {min2:.4f}, Max bound: {max2:.4f}")

    # Bottom-left: Heatmap mu vs s
    ax4 = axes[1, 0]
    s_grid4, m_ratio_grid4 = np.meshgrid(
        np.linspace(1.0, 3.0, 41),
        np.linspace(0.1, 1.0, 30)
    )
    bounds4 = np.zeros_like(s_grid4)

    for i in range(s_grid4.shape[0]):
        for j in range(s_grid4.shape[1]):
            s_val4 = s_grid4[i, j]
            H_A_s_val4 = generalized_harmonic(A_fixed, s_val4)
            m_bar4 = m_ratio_grid4[i, j] * H_A_s_val4
            bound_val, is_valid = lower_bound_with_validity(A_fixed, s_val4, m_bar4, epsilon_fixed, rho_fixed)
            bounds4[i, j] = bound_val if is_valid else np.nan

    contour4 = ax4.contourf(s_grid4, m_ratio_grid4, bounds4, levels=20, cmap='plasma')
    cbar4 = fig.colorbar(contour4, ax=ax4, pad=0.02)
    cbar4.ax.tick_params(labelsize=18)
    cbar4.set_label('Lower Bound of $P_c$', fontsize=24)
    ax4.set_xlabel('$s$ (Skewness Factor)', fontsize=28)
    ax4.set_ylabel('$\\frac{\\mu}{H_{|A|,s}}$ (Scaling Factor)', fontsize=28)
    ax4.set_xticks(s_ticks)
    ax4.set_title(f'(c) $\\mu$ vs $s$ ($|A|$={A_fixed}, $\\epsilon$={epsilon_fixed})', fontsize=30)
    ax4.tick_params(axis='both', labelsize=26)

    # Print gridded (1D sweep) sample values for plot (c)
    print("\nPlot (c) - (\\mu/H) vs s sample values (1D sweeps on grid):")
    s_values_c = np.linspace(1.0, 3.0, 41)
    m_ratio_values_c = np.linspace(0.1, 1.0, 30)

    m_ratio_fix = 0.5
    m_fix_idx = _nearest_index(m_ratio_values_c, m_ratio_fix)
    print(f"  Fix (\\mu/H)={float(m_ratio_values_c[m_fix_idx]):.3g}: sweep s")
    for s_rep in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        s_idx = _nearest_index(s_values_c, s_rep)
        s_val = float(s_values_c[s_idx])
        m_ratio = float(m_ratio_values_c[m_fix_idx])
        bound_val = bounds4[m_fix_idx, s_idx]
        if not np.isfinite(bound_val):
            H_A_s = generalized_harmonic(A_fixed, s_val)
            reasons = _diagnose_invalid(A_fixed, s_val, float(m_ratio_values_c[m_fix_idx]) * H_A_s, epsilon_fixed, rho_fixed)
            print(f"    s={s_val:.3g}: invalid -> " + "; ".join(reasons))
            continue
        H_A_s = generalized_harmonic(A_fixed, s_val)
        m_bar = m_ratio * H_A_s
        r_pivot = find_r(A_fixed, s_val, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_val))) / H_A_s
        print(f"    s={s_val:.3g}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    s_fix = 1.0
    s_fix_idx = _nearest_index(s_values_c, s_fix)
    print(f"  Fix s={float(s_values_c[s_fix_idx]):.3g}: sweep (\\mu/H)")
    for m_ratio_rep in [0.1, 0.3, 0.5, 0.7, 1.0]:
        m_idx = _nearest_index(m_ratio_values_c, m_ratio_rep)
        s_val = float(s_values_c[s_fix_idx])
        m_ratio = float(m_ratio_values_c[m_idx])
        bound_val = bounds4[m_idx, s_fix_idx]
        if not np.isfinite(bound_val):
            H_A_s = generalized_harmonic(A_fixed, s_val)
            reasons = _diagnose_invalid(A_fixed, s_val, m_ratio * H_A_s, epsilon_fixed, rho_fixed)
            print(f"    (\\mu/H)={m_ratio:.3g}: invalid -> " + "; ".join(reasons))
            continue
        H_A_s = generalized_harmonic(A_fixed, s_val)
        m_bar = m_ratio * H_A_s
        r_pivot = find_r(A_fixed, s_val, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_val))) / H_A_s
        print(f"    (\\mu/H)={m_ratio:.3g}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    min4, max4 = _safe_minmax(bounds4)
    print(f"  Min bound: {min4:.4f}, Max bound: {max4:.4f}")

    # Bottom-right: Heatmap mu vs epsilon
    ax3 = axes[1, 1]
    epsilon_grid3, m_ratio_grid3 = np.meshgrid(
        np.linspace(0.6, 1.0, 30),
        np.linspace(0.1, 1.0, 30)
    )
    bounds3 = np.zeros_like(epsilon_grid3)
    H_A_s3 = generalized_harmonic(A_fixed, s_fixed)

    for i in range(epsilon_grid3.shape[0]):
        for j in range(epsilon_grid3.shape[1]):
            m_bar3 = m_ratio_grid3[i, j] * H_A_s3
            bound_val, is_valid = lower_bound_with_validity(A_fixed, s_fixed, m_bar3, epsilon_grid3[i, j], rho_fixed)
            bounds3[i, j] = bound_val if is_valid else np.nan

    contour3 = ax3.contourf(epsilon_grid3, m_ratio_grid3, bounds3, levels=20, cmap='plasma')
    cbar3 = fig.colorbar(contour3, ax=ax3, pad=0.02)
    cbar3.ax.tick_params(labelsize=18)
    cbar3.set_label('Lower Bound of $P_c$', fontsize=24)
    ax3.set_xlabel('$\\epsilon$ (Jaccard Similarity)', fontsize=28)
    ax3.set_ylabel('$\\frac{\\mu}{H_{|A|,s}}$ (Scaling Factor)', fontsize=28)
    ax3.set_title(f'(d) $\\mu$ vs $\\epsilon$ ($|A|$={A_fixed}, $s$={s_fixed})', fontsize=30)
    ax3.tick_params(axis='both', labelsize=26)

    # Print gridded (1D sweep) sample values for plot (d)
    print("\nPlot (d) - (\\mu/H) vs \\epsilon sample values (1D sweeps on grid):")
    epsilon_values_d = np.linspace(0.6, 1.0, 30)
    m_ratio_values_d = np.linspace(0.1, 1.0, 30)

    m_ratio_fix = 0.5
    m_fix_idx = _nearest_index(m_ratio_values_d, m_ratio_fix)
    print(f"  Fix (\\mu/H)={float(m_ratio_values_d[m_fix_idx]):.3g}: sweep \\epsilon")
    for eps_rep in [0.6, 0.7, 0.8, 0.9, 1.0]:
        eps_idx = _nearest_index(epsilon_values_d, eps_rep)
        eps_val = float(epsilon_values_d[eps_idx])
        m_ratio = float(m_ratio_values_d[m_fix_idx])
        bound_val = bounds3[m_fix_idx, eps_idx]
        if not np.isfinite(bound_val):
            reasons = _diagnose_invalid(A_fixed, s_fixed, float(m_ratio_values_d[m_fix_idx]) * H_A_s3, eps_val, rho_fixed)
            print(f"    \\epsilon={eps_val:.3g}: invalid -> " + "; ".join(reasons))
            continue
        m_bar = m_ratio * H_A_s3
        r_pivot = find_r(A_fixed, s_fixed, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_fixed))) / H_A_s3
        print(f"    \\epsilon={eps_val:.3g}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    eps_fix = 0.8
    eps_fix_idx = _nearest_index(epsilon_values_d, eps_fix)
    print(f"  Fix \\epsilon={float(epsilon_values_d[eps_fix_idx]):.3g}: sweep (\\mu/H)")
    for m_ratio_rep in [0.1, 0.3, 0.5, 0.7, 1.0]:
        m_idx = _nearest_index(m_ratio_values_d, m_ratio_rep)
        m_ratio = float(m_ratio_values_d[m_idx])
        eps_val = float(epsilon_values_d[eps_fix_idx])
        bound_val = bounds3[m_idx, eps_fix_idx]
        if not np.isfinite(bound_val):
            reasons = _diagnose_invalid(A_fixed, s_fixed, m_ratio * H_A_s3, eps_val, rho_fixed)
            print(f"    (\\mu/H)={m_ratio:.3g}: invalid -> " + "; ".join(reasons))
            continue
        m_bar = m_ratio * H_A_s3
        r_pivot = find_r(A_fixed, s_fixed, rho_fixed)
        r_eff = max(2, int(r_pivot))
        p_r_eff = (m_bar * (r_eff ** (-s_fixed))) / H_A_s3
        print(f"    (\\mu/H)={m_ratio:.3g}: P_c >= {bound_val:.4f} (r_pivot={r_pivot}, r_eff={r_eff}, p_r_eff={p_r_eff:.4f})")

    min3, max3 = _safe_minmax(bounds3)
    print(f"  Min bound: {min3:.4f}, Max bound: {max3:.4f}")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Figure saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("Generating Four Enhanced Heatmaps (s=1 included)")
    print("=" * 70)

    A_default = 100
    s_default = 1.5
    epsilon_default = 0.8
    rho_default = 0.8
    m_bar_ratio_default = 0.5

    plot_four_heatmaps(
        A_fixed=A_default,
        s_fixed=s_default,
        epsilon_fixed=epsilon_default,
        rho_fixed=rho_default,
        m_bar_ratio_fixed=m_bar_ratio_default,
        output_file='heatmaps_enhance2.pdf'
    )
