"""
two_fluid_utils.py
==================
Physics engine for the Two-Fluid Barotropic CO2 ejector model.

Key design decisions (locked):
  - EQUILIBRIUM isentrope only — no NEB / metastable physics.
  - One isentrope per fluid inlet: s_mot = f(P_mot_in, T_mot_in)
                                   s_suc = f(P_suc_in, T_suc_in)
  - Stretched pressure grid: dense band of ±5 bar around P_crit.
  - CoolProp HEOS backend used throughout.

Speed of Sound — HEM fully-relaxed formulation (two-phase region):

    1 / (rho * c_eq^2) =
        alpha_l / (rho_l * c_l^2)  +  alpha_v / (rho_v * c_v^2)      [mechanical]
      + T * [ alpha_l * rho_l * (ds_l/dP)^2 / Cp_l
             + alpha_v * rho_v * (ds_v/dP)^2 / Cp_v ]                 [thermal]

  ds_l/dP and ds_v/dP are computed by central difference *strictly along
  the saturation curve* (Q=0 and Q=1 respectively), NOT along the isentrope.

  Single-phase: formula reduces to standard CoolProp acoustic speed.
  Critical-point guard: HEM is bypassed within ±0.3 bar of P_crit to
  prevent Cp/density singularity overflow; CoolProp SoS is substituted.

Author: auto-generated for Ahmed's cfx_writer / NEB pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from colorama import Fore, Style

# ── CO2 constants ──────────────────────────────────────────────────────────── #
P_CRIT_CO2   = CP.PropsSI("Pcrit",   "CO2")   # 73.773e5 Pa
T_CRIT_CO2   = CP.PropsSI("Tcrit",   "CO2")   # 304.13 K
P_TRIPLE_CO2 = CP.PropsSI("ptriple", "CO2")   # 5.185e5 Pa

# Half-band around P_crit within which HEM is bypassed [Pa]
_P_CRIT_GUARD = 0.3e5   # 0.3 bar


# ================================================================================
# 1.  STRETCHED PRESSURE GRID
# ================================================================================

def make_pressure_grid(
    P_max:   float,
    P_min:   float = P_TRIPLE_CO2,
    P_crit:  float = P_CRIT_CO2,
    n_total: int   = 600,
    n_dense: int   = 150,
    band:    float = 5.0e5,     # [Pa]  +/-5 bar dense band around P_crit
) -> np.ndarray:
    """
    Build a non-uniform pressure grid from P_max to P_min (descending).

    Dense sub-grid  : P_crit +/- band   (supercritical <-> two-phase transition)
    Coarse sub-grids: [P_min, P_crit-band] and [P_crit+band, P_max]

    Returns
    -------
    P_grid : np.ndarray  descending [Pa]
    """
    P_lo_edge = max(P_crit - band, P_min)
    P_hi_edge = min(P_crit + band, P_max)

    n_coarse = n_total - n_dense
    n_low    = n_coarse // 2
    n_high   = n_coarse - n_low

    P_low   = np.linspace(P_min,     P_lo_edge, n_low,   endpoint=False)
    P_dense = np.linspace(P_lo_edge, P_hi_edge, n_dense, endpoint=False)
    P_high  = np.linspace(P_hi_edge, P_max,     n_high,  endpoint=True)

    return np.unique(np.concatenate([P_low, P_dense, P_high]))[::-1]


# ================================================================================
# 2.  HEM SPEED OF SOUND — HELPER FUNCTIONS
# ================================================================================

def _sat_entropy_derivatives(
    P:     float,
    fluid: str,
    dP:    float = 10.0,
) -> tuple[float, float]:
    """
    Compute ds_l/dP and ds_v/dP strictly along the saturation curve.

    Method: central finite difference at fixed quality.
      ds_l/dP ~ [s_l(P+dP) - s_l(P-dP)] / (2*dP)   with Q=0 fixed
      ds_v/dP ~ [s_v(P+dP) - s_v(P-dP)] / (2*dP)   with Q=1 fixed

    This is categorically different from differentiating along the isentrope.
    The isentrope determines WHICH pressures we visit; these derivatives
    measure how the saturation boundary moves as pressure changes.

    Parameters
    ----------
    P     : float  Saturation pressure [Pa]
    fluid : str    CoolProp fluid string
    dP    : float  Half-step for central difference [Pa]  (default 10 Pa)

    Returns
    -------
    ds_l_dP, ds_v_dP : [J kg^-1 K^-1 Pa^-1]
    """
    # Liquid saturation line  (Q=0 throughout)
    s_l_p = float(CP.PropsSI("S", "P", P + dP, "Q", 0, fluid))
    s_l_m = float(CP.PropsSI("S", "P", P - dP, "Q", 0, fluid))
    ds_l_dP = (s_l_p - s_l_m) / (2.0 * dP)

    # Vapour saturation line  (Q=1 throughout)
    s_v_p = float(CP.PropsSI("S", "P", P + dP, "Q", 1, fluid))
    s_v_m = float(CP.PropsSI("S", "P", P - dP, "Q", 1, fluid))
    ds_v_dP = (s_v_p - s_v_m) / (2.0 * dP)

    return ds_l_dP, ds_v_dP


def _hem_sos_two_phase(
    P:       float,
    x_v:     float,
    rho_mix: float,
    T_sat:   float,
    fluid:   str,
    dP:      float = 10.0,
) -> float | None:
    """
    Fully-relaxed HEM speed of sound for a two-phase state.

    Formula
    -------
    1 / (rho_mix * c_eq^2)
        = alpha_l / (rho_l * c_l^2)  +  alpha_v / (rho_v * c_v^2)     [mechanical]
        + T * [ alpha_l * rho_l * (ds_l/dP)^2 / Cp_l
               + alpha_v * rho_v * (ds_v/dP)^2 / Cp_v ]               [thermal]

    Volume fraction derivation (from mass+volume balance simultaneously):
        alpha_v = x_v * rho_mix / rho_v
        alpha_l = 1 - alpha_v

    Single-phase limits:
        alpha_v -> 0  => c_eq -> c_l  (pure liquid, thermal term -> 0 because
                                        alpha_v=0 and alpha_l=1 but ds_l/dP is
                                        evaluated ON the sat. curve so this is
                                        only exact at exactly Q=0)
        alpha_v -> 1  => c_eq -> c_v  (pure vapour, similarly)

    Returns
    -------
    c_hem : float [m/s], or None if any guard condition triggers.

    None is returned (rather than raising) so the caller can substitute
    the CoolProp value cleanly without branching on exception type.
    """
    # ── Saturation boundary properties ────────────────────────────────────────
    rho_l = float(CP.PropsSI("D",  "P", P, "Q", 0, fluid))
    rho_v = float(CP.PropsSI("D",  "P", P, "Q", 1, fluid))
    c_l   = float(CP.PropsSI("d(P)/d(D)|S",  "P", P, "Q", 0, fluid))**0.5   # frozen phase SoS
    c_v   = float(CP.PropsSI("d(P)/d(D)|S",  "P", P, "Q", 1, fluid))**0.5
    Cp_l  = float(CP.PropsSI("C",  "P", P, "Q", 0, fluid))
    Cp_v  = float(CP.PropsSI("C",  "P", P, "Q", 1, fluid))

    # ── Guard: unphysical saturation properties signal critical singularity ────
    if (c_l <= 0.0 or c_v <= 0.0 or Cp_l <= 0.0 or Cp_v <= 0.0
            or not np.isfinite(Cp_l) or not np.isfinite(Cp_v)
            or not np.isfinite(c_l)  or not np.isfinite(c_v)):
        return None

    # ── Volume fractions (mass+volume balance) ────────────────────────────────
    alpha_v = float(np.clip(x_v * rho_mix / rho_v, 0.0, 1.0))
    alpha_l = 1.0 - alpha_v

    # ── Saturation entropy derivatives (strictly along dome) ──────────────────
    ds_l_dP, ds_v_dP = _sat_entropy_derivatives(P, fluid, dP)

    # ── Mechanical compressibility ─────────────────────────────────────────────
    mech = (alpha_l / (rho_l * c_l ** 2)
            + alpha_v / (rho_v * c_v ** 2))

    # ── Thermal relaxation ─────────────────────────────────────────────────────
    therm = T_sat * (
        alpha_l * rho_l * ds_l_dP ** 2 / Cp_l
        + alpha_v * rho_v * ds_v_dP ** 2 / Cp_v
    )

    rho_c2_inv = mech + therm     # = 1 / (rho_mix * c_eq^2)

    # ── Positive-definite check ────────────────────────────────────────────────
    if rho_c2_inv <= 0.0 or not np.isfinite(rho_c2_inv):
        return None

    c_hem = float(np.sqrt(1.0 / (rho_mix * rho_c2_inv)))

    if not np.isfinite(c_hem) or c_hem <= 0.0:
        return None

    return c_hem


# ================================================================================
# 3.  PHASE FRACTION HELPER
# ================================================================================

def _phase_fractions(
    P:       float,
    s_in:    float,
    rho_mix: float,
    fluid:   str,
) -> tuple[float, float]:
    """
    Return (alpha_v, x_v) for a given equilibrium (P, s_in) state.

    alpha_v : vapour volume fraction  in [0, 1]
    x_v     : vapour mass fraction (quality Q)  in [0, 1]
    """
    phase_key = CP.PhaseSI("P", P, "S", s_in, fluid)
    is_two    = "two" in phase_key.lower()

    if is_two:
        Q       = float(CP.PropsSI("Q", "P", P, "S", s_in, fluid))
        rho_v   = float(CP.PropsSI("D", "P", P, "Q", 1.0,  fluid))
        alpha_v = Q * rho_mix / rho_v
        x_v     = Q
    elif phase_key in ("gas", "supercritical_gas"):
        alpha_v, x_v = 1.0, 1.0
    else:
        alpha_v, x_v = 0.0, 0.0

    return (float(np.clip(alpha_v, 0.0, 1.0)),
            float(np.clip(x_v,     0.0, 1.0)))


# ================================================================================
# 4.  EQUILIBRIUM ISENTROPE TABLE  (main entry point)
# ================================================================================

def generate_equilibrium_table(
    P_in:     float,
    T_in:     float,
    fluid:    str,
    P_min:    float = P_TRIPLE_CO2,
    label:    str   = "fluid",
    dP_deriv: float = 10.0,
) -> pd.DataFrame:
    """
    Generate a barotropic property table along the equilibrium isentrope
    anchored at (P_in, T_in).

    Speed of Sound routing
    ----------------------
    two-phase  AND  |P - P_crit| > _P_CRIT_GUARD
        => HEM fully-relaxed formula (_hem_sos_two_phase)
    two-phase  AND  |P - P_crit| <= _P_CRIT_GUARD  (critical-point guard)
        => CoolProp PropsSI('A') as fallback  [labelled 'CoolProp_guard']
    _hem_sos_two_phase returns None (numerical failure)
        => CoolProp PropsSI('A') as fallback  [labelled 'CoolProp_fallback']
    single-phase (liquid, vapour, supercritical)
        => CoolProp PropsSI('A') directly     [labelled 'CoolProp_single']

    Output columns
    --------------
    Pressure, Rho, Mu,
    SoS            -- value sent to CFX interpolation table
    SoS_CoolProp   -- raw CoolProp value, kept for verification / plotting
    SoS_source     -- 'HEM' | 'CoolProp_guard' | 'CoolProp_fallback' | 'CoolProp_single'
    AlphaV, X_v, X_l_sat, X_l_meta, Phase
    """
    s_in = float(CP.PropsSI("S", "P", P_in, "T", T_in, fluid))

    print(f"\n{Fore.CYAN}[EqTable:{label}]{Style.RESET_ALL} "
          f"P_in = {P_in/1e5:.3f} bar | T_in = {T_in - 273.15:.2f} C | "
          f"s_in = {s_in:.4f} J/kg/K")

    P_grid   = make_pressure_grid(P_max=P_in, P_min=P_min)
    records  = []
    n_fail   = 0
    n_hem    = 0
    n_guard  = 0
    n_fallbk = 0

    for P in P_grid:
        try:
            rho    = float(CP.PropsSI("D", "P", P, "S", s_in, fluid))
            T_loc  = float(CP.PropsSI("T", "P", P, "S", s_in, fluid))
            mu     = float(CP.PropsSI("V", "P", P, "S", s_in, fluid))
            sos_cp = float(CP.PropsSI("d(P)/d(D)|S", "P", P, "S", s_in, fluid))**0.5

            phase_key = CP.PhaseSI("P", P, "S", s_in, fluid)
            is_two    = "two" in phase_key.lower()

            alpha_v, x_v = _phase_fractions(P, s_in, rho, fluid)

            near_crit = abs(P - P_CRIT_CO2) <= _P_CRIT_GUARD

            if is_two and not near_crit:
                # HEM path
                c_hem = None
                try:
                    c_hem = _hem_sos_two_phase(
                        P=P, x_v=x_v, rho_mix=rho,
                        T_sat=T_loc, fluid=fluid, dP=dP_deriv,
                    )
                except Exception:
                    c_hem = None

                if c_hem is not None:
                    sos    = c_hem
                    source = "HEM"
                    n_hem += 1
                else:
                    sos    = sos_cp
                    source = "CoolProp_fallback"
                    n_fallbk += 1

            elif is_two and near_crit:
                # Critical-point guard: bypass HEM
                sos    = sos_cp
                source = "CoolProp_guard"
                n_guard += 1

            else:
                # Single-phase: formula collapses to standard SoS
                sos    = sos_cp
                source = "CoolProp_single"

            records.append({
                "Pressure":     P,
                "Rho":          rho,
                "Mu":           mu,
                "SoS":          sos,
                "SoS_CoolProp": sos_cp,
                "SoS_source":   source,
                "AlphaV":       alpha_v,
                "X_v":          x_v,
                "X_l_sat":      1.0 - x_v,
                "X_l_meta":     0.0,
                "Phase":        phase_key,
            })

        except Exception:
            n_fail += 1
            continue

    if n_fail:
        print(f"  {Fore.YELLOW}[Warning] {n_fail} points skipped "
              f"(CoolProp errors){Style.RESET_ALL}")
    if n_fallbk:
        print(f"  {Fore.YELLOW}[Warning] {n_fallbk} two-phase points fell back "
              f"to CoolProp SoS (HEM numerical failure){Style.RESET_ALL}")

    df = (pd.DataFrame(records)
            .sort_values("Pressure")
            .reset_index(drop=True))

    n_2ph = n_hem + n_guard + n_fallbk
    print(
        f"  {Fore.GREEN}[OK] {len(df)} points  "
        f"[{df['Pressure'].min()/1e5:.2f} - {df['Pressure'].max()/1e5:.2f} bar]"
        f"{Style.RESET_ALL}\n"
        f"  SoS: HEM={n_hem}  crit-guard={n_guard}  "
        f"HEM-fallback={n_fallbk}  single-phase={len(df)-n_2ph}"
    )
    return df


# ================================================================================
# 5.  VERIFICATION PLOT
# ================================================================================

def plot_sos_comparison(
    df:        pd.DataFrame,
    label:     str           = "",
    figsize:   tuple         = (13, 7),
    save_path: str | None    = None,
) -> plt.Figure:
    """
    Four-panel verification plot: HEM SoS vs CoolProp SoS along the isentrope.

    Panels
    ------
    [0,0]  SoS [m/s] vs P — both curves, colour-coded by SoS_source
    [0,1]  Relative deviation (SoS_HEM - SoS_CP) / SoS_CP [%] — two-phase only
    [1,0]  AlphaV and X_v vs P — two-phase region reference
    [1,1]  Rho vs P — density sanity check

    Vertical dashed lines mark saturation dome entry/exit.
    Red dotted line marks P_crit.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor("#ffffff")

    title = f"HEM vs CoolProp Speed of Sound  |  {label}" if label else \
            "HEM vs CoolProp Speed of Sound"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    P_bar = df["Pressure"] / 1e5

    is_hem      = df["SoS_source"] == "HEM"
    is_guard    = df["SoS_source"] == "CoolProp_guard"
    is_fallback = df["SoS_source"] == "CoolProp_fallback"
    is_two      = is_hem | is_guard | is_fallback
    is_single   = ~is_two

    two_P     = df.loc[is_two, "Pressure"]
    P_dome_lo = two_P.min() / 1e5 if len(two_P) else None
    P_dome_hi = two_P.max() / 1e5 if len(two_P) else None

    C_HEM  = "#2ca02c"
    C_CP   = "#1f77b4"
    C_SING = "#9467bd"
    C_FALL = "#ff7f0e"
    C_GARD = "#d62728"

    def _dome_lines(ax):
        if P_dome_lo is not None:
            ax.axvline(P_dome_lo, color="#aaaaaa", ls="--", lw=0.8)
        if P_dome_hi is not None:
            ax.axvline(P_dome_hi, color="#aaaaaa", ls="--", lw=0.8)
        ax.axvline(P_CRIT_CO2 / 1e5, color=C_GARD,
                   ls=":", lw=0.9, label="$P_{crit}$")

    # ── Panel [0,0]: SoS curves ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.set_facecolor("#ffffff")
    ax.plot(P_bar, df["SoS_CoolProp"],
            color=C_CP, lw=1.4, ls="--", alpha=0.7, label="CoolProp (ref.)")
    if is_hem.any():
        ax.plot(P_bar[is_hem], df.loc[is_hem, "SoS"],
                color=C_HEM, lw=2.2, label="HEM (fully relaxed)")
    if is_single.any():
        ax.plot(P_bar[is_single], df.loc[is_single, "SoS"],
                color=C_SING, lw=1.5, alpha=0.55, label="Single-phase")
    if is_fallback.any():
        ax.scatter(P_bar[is_fallback], df.loc[is_fallback, "SoS"],
                   color=C_FALL, s=20, zorder=5, label="HEM fallback")
    if is_guard.any():
        ax.scatter(P_bar[is_guard], df.loc[is_guard, "SoS"],
                   color=C_GARD, s=20, marker="x", zorder=5, label="Crit. guard")
    _dome_lines(ax)
    ax.set_xlabel("Pressure [bar]")
    ax.set_ylabel("Speed of Sound [m/s]")
    ax.set_title("SoS: HEM vs CoolProp")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Panel [0,1]: Relative deviation ──────────────────────────────────────
    ax = axes[0, 1]
    ax.set_facecolor("#ffffff")
    hem_df = df[is_hem].copy()
    if len(hem_df) > 0:
        rel = (hem_df["SoS"] - hem_df["SoS_CoolProp"]) / hem_df["SoS_CoolProp"] * 100.0
        ax.fill_between(hem_df["Pressure"] / 1e5, rel, alpha=0.2, color=C_HEM)
        ax.plot(hem_df["Pressure"] / 1e5, rel, color=C_HEM, lw=1.8)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        _dome_lines(ax)

        idx_max = rel.abs().idxmax()
        ax.annotate(
            f"{rel[idx_max]:.1f}%\n@ {hem_df.loc[idx_max,'Pressure']/1e5:.1f} bar",
            xy=(hem_df.loc[idx_max, "Pressure"] / 1e5, rel[idx_max]),
            xytext=(15, 5), textcoords="offset points", fontsize=8, color=C_HEM,
            arrowprops=dict(arrowstyle="->", color=C_HEM, lw=0.8),
        )
    else:
        ax.text(0.5, 0.5, "No HEM two-phase points",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Pressure [bar]")
    ax.set_ylabel("(SoS_HEM - SoS_CP) / SoS_CP  [%]")
    ax.set_title("Relative Deviation (two-phase HEM only)")
    ax.grid(True, alpha=0.25)

    # ── Panel [1,0]: Phase fractions ──────────────────────────────────────────
    ax = axes[1, 0]
    ax.set_facecolor("#ffffff")
    ax.plot(P_bar, df["AlphaV"], color="#e377c2", lw=1.8, label="alpha_v (vol.)")
    ax.plot(P_bar, df["X_v"],    color="#8c564b", lw=1.8, ls="--", label="x_v (quality)")
    _dome_lines(ax)
    ax.set_xlabel("Pressure [bar]")
    ax.set_ylabel("Fraction [—]")
    ax.set_title("Phase Fractions along Isentrope")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Panel [1,1]: Density ──────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.set_facecolor("#ffffff")
    ax.plot(P_bar, df["Rho"], color="#7f7f7f", lw=1.8)
    _dome_lines(ax)
    ax.set_xlabel("Pressure [bar]")
    ax.set_ylabel("Density [kg/m^3]")
    ax.set_title("Mixture Density along Isentrope")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
        print(f"[Plot] Saved: {save_path}")
    return fig


# ================================================================================
# 6.  QUICK SANITY PRINT
# ================================================================================

def print_table_summary(df: pd.DataFrame, label: str = "") -> None:
    """Print a thinned-row summary of key table quantities."""
    tag = f"[{label}] " if label else ""
    print(f"\n{Fore.MAGENTA}{tag}Table Summary{Style.RESET_ALL}")
    cols      = ["Pressure", "Rho", "Mu", "SoS", "SoS_CoolProp", "AlphaV", "X_v"]
    available = [c for c in cols if c in df.columns]
    subset    = df.iloc[:: max(1, len(df) // 8)][available]
    with pd.option_context("display.float_format", "{:.4e}".format,
                           "display.width", 140):
        print(subset.to_string(index=False))
