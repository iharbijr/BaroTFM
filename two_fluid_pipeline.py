"""
two_fluid_pipeline.py
=====================
CCL writer, CSE script generator, and pipeline orchestrator for the
Two-Fluid Barotropic CO2 ejector model.

CCL structure verified against a working CFX-Pre 24.x multiphase .def.
Every block, option token, and nesting level mirrors the proven schema.

KEY STRUCTURAL FACTS (parsed from working .def)
------------------------------------------------
Option = Material Library          NOT "Material", NOT "Mixture Component"
MORPHOLOGY: sub-block              NOT inline "Morphology Option ="
FLUID: <n> under BOUNDARY:        NOT under BOUNDARY CONDITIONS:
VOLUME FRACTION inside FLUID BC    same pattern as working .def
REFERENCE PRESSURE: in DOMAIN MODELS (0 [bar])
MULTIPHASE MODELS: Homogeneous Model = On  (flag, not Option = Homogeneous)
FLUID PAIR: is a top-level domain block, NOT inside FLUID MODELS
Outlet  -> OPENING + Opening Pressure and Direction
Suc. inlet -> Static Pressure  (motive -> Total Pressure)
&replace DOMAIN: <n>           replaces entire domain block in .def

CEL NAMING (CFX legacy parser — underscores are internal delimiters)
---------------------------------------------------------------------
All FUNCTION, ADDITIONAL VARIABLE, MATERIAL, FLUID DEFINITION, and
EXPRESSION names are strictly alphanumeric.  BOUNDARY / DOMAIN names
and Python config keys are exempt — they are never CEL identifiers.

ENTHALPY AV CHAIN (solver-level mixture reconstruction)
-------------------------------------------------------
Algebraic AVs evaluated inside FLUID: MotiveCO2 block (CFX requires
a fluid host even for mixture-level quantities):

  RhoMix   = VF_mot * MotiveRho(P)  + VF_suc * SuctionRho(P)
  Ymot     = (VF_mot * MotiveRho(P)) / RhoMix
  Hmix     = Ymot*H0mot + (1-Ymot)*H0suc - 0.5*(u^2+v^2+w^2)
  Xmix     = max(0, min(1, (Hmix - fnHlsat(P))/(fnHvsat(P)-fnHlsat(P))))
  VfVapMix = (Xmix * RhoMix) / fnRhov(P)

Xmix is clipped to [0, 1] to maintain stability in subcooled / superheated
single-phase regions where the enthalpy falls outside the saturation dome.

Velocity kinetic energy uses component form valid in solver CEL:
  0.5 * (Velocity u^2 + Velocity v^2 + Velocity w^2)

EXTENDED SUCTION RANGE
-----------------------
The SuctionRho and SuctionMu tables are built on a grid from P_triple
to P_mot_in (not P_suc_in).  This ensures the barotropic functions
remain valid during compression of the suction fluid in the mixing
section where static pressure may exceed the suction inlet pressure.
The isentrope itself is still anchored at (P_suc_in, T_suc_in).

Author: auto-generated for Ahmed's cfx_writer / NEB pipeline
"""

from __future__ import annotations

import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from two_fluid_utils import (
    generate_equilibrium_table,
    generate_saturation_table,
    print_table_summary,
    plot_sos_comparison,
)

_CO2_MOLAR_MASS = 44.01   # kg kmol^-1

# Wall face list — latest geometry configuration
_DEFAULT_WALL_LOCATION = (
    "Primitive 2D AI,Primitive 2D AJ,Primitive 2D AK,Primitive 2D AL,"
    "Primitive 2D AM,Primitive 2D AN,Primitive 2D AO,Primitive 2D AP,"
    "walls zone2d diff,walls zone2d exit,walls zone2d mc_diff,walls zone2d mc_div"
)


# ================================================================================
# 0.  CEL RAMP BUILDER
# ================================================================================

def build_ramp_cel(
    steps:        list[tuple[int, float]],
    start_expr:   str = "Pstart",
    target_expr:  str = "Psucin",
) -> str:
    """
    Build a nested CEL if-expression that ramps a pressure from start to target
    using CFX's built-in iteration counter variable ``aitern``.

    Parameters
    ----------
    steps       : list of (iter_threshold, fraction) tuples.
                  fraction is the proportion of the full range to apply at that
                  stage.  Entries with fraction == 1.0 are stripped — the full
                  target is the implicit else-branch of the last real step.
                  Steps are sorted ascending by threshold before processing.
    start_expr  : CEL expression for the ramp start (default "Pstart").
    target_expr : CEL expression for the final target (default "Psucin").

    Returns
    -------
    cel : str   A valid CEL expression string, e.g.:
                  if(aitern <= 1500, Pstart + 0.3300 * (Psucin - Pstart),
                     if(aitern <= 3000, Pstart + 0.6600 * (Psucin - Pstart),
                        Psucin))

    Special cases
    -------------
    Empty or None steps  ->  returns target_expr directly (no ramp, immediate full BC).
    Single step (f=1.0)  ->  returns target_expr (same as empty).
    """
    if not steps:
        return target_expr

    # Sort and strip trailing full-target steps
    sorted_steps = sorted(steps, key=lambda x: x[0])
    while sorted_steps and sorted_steps[-1][1] >= 1.0:
        sorted_steps.pop()

    if not sorted_steps:
        return target_expr   # all steps were full-target

    def _ramp_val(frac: float) -> str:
        return f"{start_expr} + {frac:.4f} * ({target_expr} - {start_expr})"

    def _build(remaining: list[tuple[int, float]]) -> str:
        if not remaining:
            return target_expr
        itern, frac = remaining[0]
        return (
            f"if(aitern <= {itern}, {_ramp_val(frac)}, "
            f"{_build(remaining[1:])})"
        )

    return _build(sorted_steps)


# ================================================================================
# 1.  LOW-LEVEL HELPERS
# ================================================================================

def _format_pairs(x_data, y_data, indent: int = 10) -> str:
    """Format (x, y) pairs into a wrapped CCL Data Pairs string (~75 chars/line)."""
    pairs    = [f"{x:.6E}, {y:.6E}" for x, y in zip(x_data, y_data)]
    full_str = ", ".join(pairs)
    chunk    = 75
    pad      = " " * indent
    lines: list[str] = []
    cur = 0
    while cur < len(full_str):
        end = min(cur + chunk, len(full_str))
        if end < len(full_str):
            bp = full_str.rfind(",", cur, end) + 1
            if bp <= cur:
                bp = end
        else:
            bp = end
        suffix = "\\\n" if bp < len(full_str) else "\n"
        lines.append(f"{pad}{full_str[cur:bp]}{suffix}")
        cur = bp
    return "".join(lines)


def _clean_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].interpolate(method="linear").fillna(0.0)


# ================================================================================
# 2.  CATALOGUES
# ================================================================================

# Material-property functions  ->  MotiveRho, MotiveMu, SuctionRho, SuctionMu
_MAT_FUNC_CATALOGUE: dict[str, tuple[str, str]] = {
    "Rho": ("Rho", "kg m^-3"),
    "Mu":  ("Mu",  "kg m^-1 s^-1"),
}

# Per-phase isentrope AV functions  ->  fnMotiveAlphaV, fnSuctionSoS, etc.
_AV_FUNC_CATALOGUE: dict[str, tuple[str, str]] = {
    "AlphaV": ("AlphaV",  "[]"),
    "SoS":    ("SoS",     "m s^-1"),
    "Xv":     ("X_v",     "[]"),
    "Xlsat":  ("X_l_sat", "[]"),
}

# Saturation-dome functions (path-independent, from dedicated df_sat)
_SAT_FUNC_CATALOGUE: dict[str, tuple[str, str]] = {
    "fnTsat":  ("Tsat",  "K"),
    "fnHlsat": ("Hlsat", "J kg^-1"),
    "fnHvsat": ("Hvsat", "J kg^-1"),
    "fnRhol":  ("Rhol",  "kg m^-3"),
    "fnRhov":  ("Rhov",  "kg m^-3"),
    "fnCl":    ("Cl",    "m s^-1"),
    "fnCv":    ("Cv",    "m s^-1"),
    "fnCpl":   ("Cpl",   "J kg^-1 K^-1"),
    "fnCpv":   ("Cpv",   "J kg^-1 K^-1"),
    "fndSldP": ("dSldP", "J kg^-1 K^-1 Pa^-1"),
    "fndSvdP": ("dSvdP", "J kg^-1 K^-1 Pa^-1"),
}

# VCM Algebraic AV chain (phase topology & HEM acoustic reconstruction)
_VCM_AV_CATALOGUE: dict[str, str] = {
    "Hmix":       "m^2 s^-2",
    "XmixRaw":    "[]",
    "Xmix":       "[]",
    "VfVapMix":   "[]",
    "VfLiqMix":   "[]",
    "HEMMech":    "kg^-1 m s^2",
    "HEMTherm":   "kg^-1 m s^2",
    "HEMSoS":     "m s^-1",
}


# ================================================================================
# 3.  CCL WRITER  (structure verified against working CFX-Pre 24.x .def)
# ================================================================================

def wrap_ccl_locations(loc_string: str, max_len: int = 70) -> str:
    """
    Wrap long comma-separated location strings using CFX continuation '\\'.

    Parameters
    ----------
    loc_string : str
        Comma-separated location string (e.g., "DIFF,EXIT,MC,...")
    max_len : int
        Maximum line length before wrapping (default 70)

    Returns
    -------
    str
        Wrapped string with CFX line continuation characters
    """
    if not loc_string:
        return ""

    items = [item.strip() for item in loc_string.split(',')]
    lines = []
    current_line = ""

    for item in items:
        test_line = current_line + ("," + item if current_line else item)
        if len(test_line) > max_len and current_line:
            lines.append(current_line + ", \\")
            current_line = "        " + item  # Indent continuation
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


def write_two_fluid_ccl(
    filename: str,
    df_mot:   pd.DataFrame,
    df_suc:   pd.DataFrame,
    df_sat:   pd.DataFrame,
    config:   dict,
    h0_mot:   float = 0.0,
    h0_suc:   float = 0.0,
) -> None:
    """
    Write a complete &replace CCL patch for the Two-Fluid Barotropic ejector.

    Solver control parameters are read from config["SOLVER_SETTINGS"] (optional):
      MAX_ITER       : int   Maximum solver iterations          (default 2000)
      BACKUP_INTERVAL: int   Backup results every N iterations  (default 500)
      RAMPING_STEPS  : list  [(iter, fraction), ...]  suction pressure ramp
                             fraction = proportion of range Pout -> Psucin
                             default: [] = immediate full target (no ramp)

    Pressure ramp logic:
      Pstart     = Pout   (discharge pressure — ramp anchor)
      PsucinRamp = build_ramp_cel(steps)
                 = Psucin  when no steps (full target immediately)
                 = if(aitern <= N1, Pstart + f1*(Psucin-Pstart), ...)  with ramp

    --------------------
    P_mot_in, T_mot_in       Motive  inlet stagnation [Pa], [K]
    P_suc_in, T_suc_in       Suction inlet stagnation [Pa], [K]
    P_out                    Outlet static pressure   [Pa]
    DOMAIN_NAME              e.g. "Ejector"
    DOMAIN_LOCATIONS         mesh region string from .def
    MOT_BC_NAME/_LOCATION    motive  inlet boundary name + mesh face tag
    SUC_BC_NAME/_LOCATION    suction inlet boundary name + mesh face tag
    OUT_BC_NAME/_LOCATION    outlet boundary name + mesh face tag
    WALL_BC_NAME/_LOCATIONS  wall boundary name + face list
    SYMM1_NAME/_LOCATION     symmetry plane 1 (optional)
    SYMM2_NAME/_LOCATION     symmetry plane 2 (optional)
    FLUID_TEMPERATURE        isothermal fluid temperature [K]  (default 330)
    """
    domain      = config.get("DOMAIN_NAME",      "Ejector")
    dom_loc_raw = config.get("DOMAIN_LOCATIONS",
                             "DIFF,EXIT,MC,MC_DIFF,MC_DIV,MN,MN_in,SN")
    dom_loc     = wrap_ccl_locations(dom_loc_raw)

    bc_mot      = config.get("MOT_BC_NAME",      "Inlet MN")
    bc_mot_loc  = config.get("MOT_BC_LOCATION",  "Inlet_MN")
    bc_suc      = config.get("SUC_BC_NAME",      "Inlet SN")
    bc_suc_loc  = config.get("SUC_BC_LOCATION",  "Inlet_SN")
    bc_out      = config.get("OUT_BC_NAME",      "Outlet")
    bc_out_loc  = config.get("OUT_BC_LOCATION",  "outlet")
    bc_wall     = config.get("WALL_BC_NAME",     "Ejector Default")
    bc_wall_loc_raw = config.get("WALL_LOCATIONS", _DEFAULT_WALL_LOCATION)
    bc_wall_loc = wrap_ccl_locations(bc_wall_loc_raw)
    symm1       = config.get("SYMM1_NAME",       "Symm 1")
    symm1_loc   = config.get("SYMM1_LOCATION",   "Sym_1")
    symm2       = config.get("SYMM2_NAME",       "Symm 2")
    symm2_loc   = config.get("SYMM2_LOCATION",   "Sym_2")
    T_fluid     = config.get("FLUID_TEMPERATURE", 330.0)

    P_mot = config["P_mot_in"]
    T_mot = config["T_mot_in"]
    P_suc = config["P_suc_in"]
    P_out = config["P_out"]

    # ── Solver / output settings (from optional SOLVER_SETTINGS sub-dict) ──────
    _ss              = config.get("SOLVER_SETTINGS", {})
    max_iter         = int(_ss.get("MAX_ITER",        2000))
    backup_interval  = int(_ss.get("BACKUP_INTERVAL", 500))
    ramping_steps    = _ss.get("RAMPING_STEPS",       [])

    # Build the suction ramp CEL expression
    ramp_cel         = build_ramp_cel(ramping_steps)
    has_ramp         = (ramp_cel != "Psucin")

    _phases = [("Motive", df_mot), ("Suction", df_suc)]

    # ── inner helpers ──────────────────────────────────────────────────────────
    def _write_func(f, name, df, col, units):
        f.write(f"    FUNCTION: {name}\n")
        f.write( "      Argument Units = Pa\n")
        f.write( "      Option = Interpolation\n")
        f.write(f"      Result Units = {units}\n")
        f.write( "      INTERPOLATION DATA:\n")
        f.write( "        Extend Max = On\n")
        f.write( "        Extend Min = On\n")
        f.write( "        Option = One Dimensional\n")
        f.write( "        Data Pairs = \\\n")
        f.write(_format_pairs(df["Pressure"], _clean_series(df, col)))
        f.write( "      END\n")
        f.write( "    END\n")

    def _write_av_def(f, av_name, units):
        f.write(f"  ADDITIONAL VARIABLE: {av_name}\n")
        f.write( "    Option = Definition\n")
        f.write(f"    Units = {units}\n")
        f.write( "    Tensor Type = SCALAR\n")
        f.write( "    Variable Type = Unspecified\n")
        f.write( "  END\n")


    with open(filename, "w", encoding="utf-8") as f:

        # ── File header ────────────────────────────────────────────────────────
        f.write("# ============================================================\n")
        f.write("# VCM Barotropic CCL  --  auto-generated\n")
        f.write(f"# Motive : {P_mot/1e5:.3f} bar / {T_mot-273.15:.2f} C (Total Pressure inlet)\n")
        f.write(f"# Suction: {P_suc/1e5:.3f} bar (Stagnation / Total Pressure inlet)\n")
        f.write(f"# Outlet : {P_out/1e5:.3f} bar (Static Pressure Outlet)\n")
        f.write(f"# Domain : {domain}\n")
        f.write("# Model  : Variable Composition Mixture (VCM) + Isothermal\n")
        f.write("# Phase topology: Hmix->XmixRaw->Xmix[0,1]->VfVapMix,VfLiqMix\n")
        f.write("# HEM Speed of Sound: HEMMech + HEMTherm -> HEMSoS\n")
        f.write("# ============================================================\n\n")

        # ======================================================================
        # LIBRARY
        # ======================================================================
        f.write("LIBRARY:\n\n")

        # -- AV definitions (VCM algebraic chain) -----------------------------
        f.write("  # -- VCM Algebraic AV Chain ---------------------------------\n")
        for av_name, units in _VCM_AV_CATALOGUE.items():
            _write_av_def(f, av_name, units)

        # -- MATERIAL: Pure Components for VCM --------------------------------
        f.write("\n  # -- Pure Components (MotiveCO2, SuctionCO2) ---------------\n")
        for phase, _ in _phases:
            mat = f"{phase}CO2"
            f.write(f"\n  &replace MATERIAL: {mat}\n")
            f.write(f"    Material Description = CO2 component for {phase} stream\n")
            f.write( "    Material Group = User\n")
            f.write( "    Object Origin = User\n")
            f.write( "    Option = Pure Substance\n")
            f.write( "    PROPERTIES:\n")
            f.write( "      Option = General Material\n")
            f.write( "      DYNAMIC VISCOSITY:\n")
            f.write( "        Option = Value\n")
            f.write( "        Dynamic Viscosity = 1.0 [kg m^-1 s^-1]\n")
            f.write( "      END\n")
            f.write( "      EQUATION OF STATE:\n")
            f.write( "        Density = 1.0 [kg m^-3]\n")
            f.write(f"        Molar Mass = {_CO2_MOLAR_MASS} [kg kmol^-1]\n")
            f.write( "        Option = Value\n")
            f.write( "      END\n")
            f.write( "      SPECIFIC HEAT CAPACITY:\n")
            f.write( "        Option = Value\n")
            f.write( "        Specific Heat Capacity = 1000.0 [J kg^-1 K^-1]\n")
            f.write( "        Specific Heat Type = Constant Pressure\n")
            f.write( "      END\n")
            f.write( "    END\n")
            f.write( "  END\n")

        # -- MATERIAL: Variable Composition Mixture ----------------------------
        f.write("\n  &replace MATERIAL: MixtureCO2\n")
        f.write( "    Material Description = Variable Composition Mixture of CO2 streams\n")
        f.write( "    Material Group = User\n")
        f.write( "    Object Origin = User\n")
        f.write( "    Option = Variable Composition Mixture\n")
        f.write( "    Thermodynamic State = Gas\n")
        f.write( "    Materials List = MotiveCO2,SuctionCO2\n")
        f.write( "    MIXTURE PROPERTIES:\n")
        f.write( "      Option = Ideal Mixture\n")
        f.write( "      EQUATION OF STATE:\n")
        f.write( "        Option = Ideal Mixture\n")
        f.write( "      END\n")
        f.write( "      SPECIFIC HEAT CAPACITY:\n")
        f.write( "        Option = Ideal Mixture\n")
        f.write( "      END\n")
        f.write( "      DYNAMIC VISCOSITY:\n")
        f.write( "        Option = Ideal Mixture\n")
        f.write( "      END\n")
        f.write( "    END\n")
        f.write( "  END\n")

        # -- CEL block --------------------------------------------------------
        f.write("\n  CEL:\n")
        f.write("    EXPRESSIONS:\n")
        f.write(f"      Pmotin = {P_mot/1e5:.6f} [bar]\n")
        f.write(f"      Tmotin = {T_mot:.4f} [K]\n")
        f.write(f"      Psucin = {P_suc/1e5:.6f} [bar]\n")
        f.write(f"      PoutTarget = {P_out/1e5:.6f} [bar]\n")
        # Stagnation enthalpies as fixed CEL constants
        f.write(f"      H0mot  = {h0_mot:.4f} [J kg^-1]\n")
        f.write(f"      H0suc  = {h0_suc:.4f} [J kg^-1]\n")
        # Pressure ramp: Pstart is the ramp anchor (discharge pressure).
        # PsucinRamp evaluates to Psucin immediately when no ramp is configured.
        f.write(f"      Pstart = {P_out/1e5:.6f} [bar]\n")
        f.write(f"      PsucinRamp = {ramp_cel}\n")
        f.write("    END\n")

        # Saturation dome functions (path-independent, from dedicated df_sat)
        f.write("\n    # -- Saturation Functions (VCM Phase Topology & HEM) -----\n")
        f.write("    # Source: dedicated saturation grid [P_triple, P_crit-eps]\n")
        f.write("    # Used by VCM AV chain for phase reconstruction & HEM SoS.\n")
        for func_name, (col, units) in _SAT_FUNC_CATALOGUE.items():
            _write_func(f, func_name, df_sat, col, units)

        f.write("  END\n")    # CEL
        f.write("END\n\n")   # LIBRARY

        # ======================================================================
        # FLOW
        # ======================================================================
        f.write("# ============================================================\n")
        f.write(f"# FLOW -- &replace DOMAIN: {domain}\n")
        f.write("# ============================================================\n")
        f.write("FLOW: Flow Analysis 1\n")
        f.write(f"  &replace DOMAIN: {domain}\n")
        f.write( "    Coord Frame = Coord 0\n")
        f.write( "    Domain Type = Fluid\n")
        f.write(f"    Location = {dom_loc}\n")

        # Wall
        f.write(f"\n    BOUNDARY: {bc_wall}\n")
        f.write( "      Boundary Type = WALL\n")
        f.write( "      Create Other Side = Off\n")
        f.write( "      Interface Boundary = Off\n")
        f.write(f"      Location = {bc_wall_loc}\n")
        f.write( "      BOUNDARY CONDITIONS:\n")
        # f.write( "        HEAT TRANSFER:\n")
        # f.write( "          Option = Adiabatic\n")
        # f.write( "        END\n")
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = No Slip Wall\n")
        f.write( "        END\n")
        f.write( "        WALL ROUGHNESS:\n")
        f.write( "          Option = Smooth Wall\n")
        f.write( "        END\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # Motive inlet (Total Pressure)
        f.write(f"\n    BOUNDARY: {bc_mot}\n")
        f.write( "      Boundary Type = INLET\n")
        f.write( "      Interface Boundary = Off\n")
        f.write(f"      Location = {bc_mot_loc}\n")
        f.write( "      BOUNDARY CONDITIONS:\n")
        f.write( "        FLOW DIRECTION:\n")
        f.write( "          Option = Normal to Boundary Condition\n")
        f.write( "        END\n")
        f.write( "        FLOW REGIME:\n")
        f.write( "          Option = Subsonic\n")
        f.write( "        END\n")
        f.write( "        COMPONENT: MotiveCO2\n")
        f.write( "          Mass Fraction = 1.0\n")
        f.write( "          Option = Mass Fraction\n")
        f.write( "        END\n")
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = Total Pressure\n")
        f.write( "          Relative Pressure = Pmotin\n")
        f.write( "        END\n")
        f.write( "        TURBULENCE:\n")
        f.write( "          Option = Medium Intensity and Eddy Viscosity Ratio\n")
        f.write( "        END\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # Suction inlet (Stagnation / Total Pressure)
        f.write(f"\n    BOUNDARY: {bc_suc}\n")
        f.write( "      Boundary Type = INLET\n")
        f.write( "      Interface Boundary = Off\n")
        f.write(f"      Location = {bc_suc_loc}\n")
        f.write( "      BOUNDARY CONDITIONS:\n")
        f.write( "        FLOW DIRECTION:\n")
        f.write( "          Option = Normal to Boundary Condition\n")
        f.write( "        END\n")
        f.write( "        FLOW REGIME:\n")
        f.write( "          Option = Subsonic\n")
        f.write( "        END\n")
        f.write( "        COMPONENT: MotiveCO2\n")
        f.write( "          Mass Fraction = 0.0\n")
        f.write( "          Option = Mass Fraction\n")
        f.write( "        END\n")
        # f.write( "        HEAT TRANSFER:\n")
        # f.write( "          Option = Total Enthalpy\n")
        # f.write( "          Total Enthalpy = H0suc\n")
        # f.write( "        END\n")
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = Total Pressure\n")
        f.write( "          Relative Pressure = PsucinRamp\n")
        f.write( "        END\n")
        f.write( "        TURBULENCE:\n")
        f.write( "          Option = Medium Intensity and Eddy Viscosity Ratio\n")
        f.write( "        END\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # Outlet (Static Pressure Outlet)
        f.write(f"\n    BOUNDARY: {bc_out}\n")
        f.write( "      Boundary Type = OUTLET\n")
        f.write( "      Interface Boundary = Off\n")
        f.write(f"      Location = {bc_out_loc}\n")
        f.write( "      BOUNDARY CONDITIONS:\n")
        f.write( "        FLOW REGIME:\n")
        f.write( "          Option = Subsonic\n")
        f.write( "        END\n")
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = Static Pressure\n")
        f.write( "          Relative Pressure = PoutTarget\n")
        f.write( "        END\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # Symmetry planes
        if symm1 and symm1_loc:
            f.write(f"\n    BOUNDARY: {symm1}\n")
            f.write( "      Boundary Type = SYMMETRY\n")
            f.write(f"      Location = {symm1_loc}\n")
            f.write( "    END\n")
        if symm2 and symm2_loc:
            f.write(f"\n    BOUNDARY: {symm2}\n")
            f.write( "      Boundary Type = SYMMETRY\n")
            f.write(f"      Location = {symm2_loc}\n")
            f.write( "    END\n")

        # Domain models
        f.write("\n    DOMAIN MODELS:\n")
        f.write( "      BUOYANCY MODEL:\n")
        f.write( "        Option = Non Buoyant\n")
        f.write( "      END\n")
        f.write( "      DOMAIN MOTION:\n")
        f.write( "        Option = Stationary\n")
        f.write( "      END\n")
        f.write( "      MESH DEFORMATION:\n")
        f.write( "        Option = None\n")
        f.write( "      END\n")
        f.write( "      REFERENCE PRESSURE:\n")
        f.write( "        Reference Pressure = 0 [bar]\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # Fluid definition (VCM)
        f.write("\n    # -- Fluid definition (VCM) -------------------------------\n")
        f.write( "    FLUID DEFINITION: MixtureCO2\n")
        f.write( "      Material = MixtureCO2\n")
        f.write( "      Option = Material Library\n")
        f.write( "      MORPHOLOGY:\n")
        f.write( "        Option = Continuous Fluid\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # ── FLUID MODELS ───────────────────────────────────────────────────────
        f.write("\n    FLUID MODELS:\n")

        # Component transport options
        f.write( "      COMPONENT: MotiveCO2\n")
        f.write( "        Option = Transport Equation\n")
        f.write( "      END\n")
        f.write( "      COMPONENT: SuctionCO2\n")
        f.write( "        Option = Constraint\n")
        f.write( "      END\n")

        # VCM Algebraic AV chain equations with explicit unit stripping
        _vcm_av_eqs = [
            ("Hmix",
             "(MotiveCO2.Mass Fraction * (H0mot / 1.0 [J kg^-1]) "
             "+ SuctionCO2.Mass Fraction * (H0suc / 1.0 [J kg^-1]) "
             "- 0.5 * ((Velocity u / 1.0 [m s^-1])^2 + (Velocity v / 1.0 [m s^-1])^2 "
             "+ (Velocity w / 1.0 [m s^-1])^2)) * 1.0 [J kg^-1]"),
            ("XmixRaw",
             "((Hmix / 1.0 [J kg^-1]) - (fnHlsat(Absolute Pressure) / 1.0 [J kg^-1])) "
             "/ max((fnHvsat(Absolute Pressure) / 1.0 [J kg^-1]) "
             "- (fnHlsat(Absolute Pressure) / 1.0 [J kg^-1]), 1e-10)"),
            ("Xmix",
             "min(max(XmixRaw, 0.0), 1.0)"),
            ("VfVapMix",
             "Xmix * (Density / 1.0 [kg m^-3]) "
             "/ max(fnRhov(Absolute Pressure) / 1.0 [kg m^-3], 1e-10)"),
            ("VfLiqMix",
             "1.0 - VfVapMix"),
            ("HEMMech",
             "(VfLiqMix / max((fnRhol(Absolute Pressure) / 1.0 [kg m^-3]) "
             "* (fnCl(Absolute Pressure) / 1.0 [m s^-1])^2, 1e-10)) * 1.0 [kg^-1 m s^2] "
             "+ (VfVapMix / max((fnRhov(Absolute Pressure) / 1.0 [kg m^-3]) "
             "* (fnCv(Absolute Pressure) / 1.0 [m s^-1])^2, 1e-10)) * 1.0 [kg^-1 m s^2]"),
            ("HEMTherm",
             "((fnTsat(Absolute Pressure) / 1.0 [K]) * ("
             "(VfLiqMix * (fnRhol(Absolute Pressure) / 1.0 [kg m^-3]) "
             "* (fndSldP(Absolute Pressure) / 1.0 [J kg^-1 K^-1 Pa^-1])^2) "
             "/ max(fnCpl(Absolute Pressure) / 1.0 [J kg^-1 K^-1], 1e-10) "
             "+ (VfVapMix * (fnRhov(Absolute Pressure) / 1.0 [kg m^-3]) "
             "* (fndSvdP(Absolute Pressure) / 1.0 [J kg^-1 K^-1 Pa^-1])^2) "
             "/ max(fnCpv(Absolute Pressure) / 1.0 [J kg^-1 K^-1], 1e-10))) "
             "* 1.0 [kg^-1 m s^2]"),
            ("HEMSoS",
             "sqrt(1.0 / max((Density / 1.0 [kg m^-3]) "
             "* ((HEMMech + HEMTherm) / 1.0 [kg^-1 m s^2]), 1e-16)) * 1.0 [m s^-1]"),
        ]

        def _write_av_eq(f, av_name, expr, indent=6):
            pad = " " * indent
            f.write(f"{pad}ADDITIONAL VARIABLE: {av_name}\n")
            f.write(f"{pad}  Additional Variable Value = {expr}\n")
            f.write(f"{pad}  Option = Algebraic Equation\n")
            f.write(f"{pad}END\n")

        f.write( "      COMBUSTION MODEL:\n")
        f.write( "        Option = None\n")
        f.write( "      END\n")

        # VCM algebraic AVs - flattened as direct children of FLUID MODELS
        f.write("\n      # -- VCM Algebraic AV chain (flattened) -----------------\n")
        for av_name, eq in _vcm_av_eqs:
            _write_av_eq(f, av_name, eq, indent=6)

        f.write( "      HEAT TRANSFER MODEL:\n")
        f.write( "        Option = Isothermal\n")
        f.write(f"        Fluid Temperature = {T_fluid:.1f} [K]\n")
        f.write( "      END\n")
        f.write( "      THERMAL RADIATION MODEL:\n")
        f.write( "        Option = None\n")
        f.write( "      END\n")
        f.write( "      TURBULENCE MODEL:\n")
        f.write( "        Option = SST\n")
        f.write( "      END\n")
        f.write( "      TURBULENT WALL FUNCTIONS:\n")
        f.write( "        Option = Automatic\n")
        f.write( "      END\n")
        f.write( "    END\n")   # FLUID MODELS

        # EXPERT PARAMETERS inside DOMAIN
        f.write( "\n    EXPERT PARAMETERS:\n")
        f.write( "      solve energy = f\n")
        f.write( "    END\n")

        f.write( "  END\n")   # DOMAIN
        f.write( "END\n\n")  # FLOW (domain)

        # ── Each top-level control block gets its OWN FLOW: wrapper ──────────
        # Rule: sub-blocks NEVER carry &replace — only the immediate child
        #       of FLOW: does.  Nested &replace causes "empty context rule" errors
        #       because CFX cannot resolve the parameter schema without the full
        #       parent block context being present.

        # ── SOLUTION UNITS ─────────────────────────────────────────────────────
        f.write("FLOW: Flow Analysis 1\n")
        f.write("  &replace SOLUTION UNITS:\n")
        f.write("    Angle Units = [rad]\n")
        f.write("    Length Units = [m]\n")
        f.write("    Mass Units = [kg]\n")
        f.write("    Solid Angle Units = [sr]\n")
        f.write("    Temperature Units = [K]\n")
        f.write("    Time Units = [s]\n")
        f.write("  END\n")
        f.write("END\n\n")

        # ── SOLVER CONTROL ─────────────────────────────────────────────────────
        # Complete sub-block set required — CFX schema validates ALL siblings
        # before resolving any individual parameter (hence the context error).
        f.write("FLOW: Flow Analysis 1\n")
        f.write("  &replace SOLVER CONTROL:\n")
        f.write("    Turbulence Numerics = High Resolution\n")
        f.write("    ADVECTION SCHEME:\n")
        f.write("      Option = High Resolution\n")
        f.write("    END\n")
        f.write("    COMPRESSIBILITY CONTROL:\n")
        f.write("      Clip Pressure for Properties = On\n")
        f.write("      High Speed Numerics = On\n")
        f.write("      Minimum Pressure for Properties = 0 [Pa]\n")
        f.write("    END\n")
        f.write("    CONVERGENCE CONTROL:\n")
        f.write("      Length Scale Option = Conservative\n")
        f.write(f"      Maximum Number of Iterations = {max_iter}\n")
        f.write("      Minimum Number of Iterations = 1\n")
        f.write("      Timescale Control = Auto Timescale\n")
        f.write("      Timescale Factor = 1.0\n")
        f.write("    END\n")
        f.write("    CONVERGENCE CRITERIA:\n")
        f.write("      Conservation Target = 0.001\n")
        f.write("      Residual Target = 0.0001\n")
        f.write("      Residual Type = MAX\n")
        f.write("    END\n")
        f.write("    DYNAMIC MODEL CONTROL:\n")
        f.write("      Global Dynamic Model Control = On\n")
        f.write("    END\n")
        f.write("    INTERRUPT CONTROL:\n")
        f.write("      Option = All Interrupts\n")
        f.write("      CONVERGENCE CONDITIONS:\n")
        f.write("        Option = Default Conditions\n")
        f.write("      END\n")
        f.write("    END\n")
        f.write("    MULTIPHASE CONTROL:\n")
        f.write("      Initial Volume Fraction Smoothing = Volume-Weighted\n")
        f.write("      Volume Fraction Coupling = Coupled\n")
        f.write("    END\n")
        f.write("    VELOCITY PRESSURE COUPLING:\n")
        f.write("      Rhie Chow Option = Fourth Order\n")
        f.write("    END\n")
        f.write("  END\n")
        f.write("END\n\n")

        # ── OUTPUT CONTROL ─────────────────────────────────────────────────────
        f.write("FLOW: Flow Analysis 1\n")
        f.write("  &replace OUTPUT CONTROL:\n")
        f.write("    BACKUP DATA RETENTION:\n")
        f.write("      Option = Delete Old Files\n")
        f.write("    END\n")
        # Name must be "Backup Results" — verified from working GUI CCL
        f.write("    BACKUP RESULTS: Backup Results\n")
        f.write("      File Compression Level = Default\n")
        f.write("      Option = Standard\n")
        f.write("      OUTPUT FREQUENCY:\n")
        f.write(f"        Iteration Interval = {backup_interval}\n")
        f.write("        Option = Iteration Interval\n")
        f.write("      END\n")
        f.write("    END\n")   # BACKUP RESULTS
        f.write("    MONITOR OBJECTS:\n")
        f.write("      MONITOR BALANCES:\n")
        f.write("        Option = Full\n")
        f.write("      END\n")
        f.write("      MONITOR FORCES:\n")
        f.write("        Option = Full\n")
        f.write("      END\n")
        f.write("      MONITOR PARTICLES:\n")
        f.write("        Option = Full\n")
        f.write("      END\n")
        f.write("      MONITOR RESIDUALS:\n")
        f.write("        Option = Full\n")
        f.write("      END\n")
        f.write("      MONITOR TOTALS:\n")
        f.write("        Option = Full\n")
        f.write("      END\n")
        f.write("    END\n")   # MONITOR OBJECTS
        f.write("    RESULTS:\n")
        f.write("      File Compression Level = Default\n")
        f.write("      Option = Standard\n")
        f.write("    END\n")
        f.write("  END\n")    # OUTPUT CONTROL
        f.write("END\n\n")

    n_sat = len(_SAT_FUNC_CATALOGUE)
    n_vcm = len(_VCM_AV_CATALOGUE)
    print(f"[Pipeline] CCL written: {filename}  "
          f"(VCM + {n_sat} sat funcs + {n_vcm} algebraic AVs, UTF-8, &replace)")


# ================================================================================
# 4.  CSE POST-PROCESSING SCRIPT  (20 columns)
# ================================================================================

def write_two_fluid_cse(
    filename: str,
    config:   dict,
    x_min:    float = 0.0,
    x_max:    float = 0.0835,
    n_points: int   = 100,
) -> None:
    """
    Generate a CFX-Post CSE script for VCM centerline extraction.

    Extraction method: Polyline along geometric centerline (1D).
    Variables: Velocity, Absolute Pressure, Density, MotiveCO2.Mass Fraction,
               Hmix, Xmix, VfVapMix, HEMSoS

    AV names in evaluate() match the ADDITIONAL VARIABLE names in the CCL exactly.
    Mass Fraction field name: MotiveCO2.Mass Fraction
    """
    csv_name  = os.path.splitext(os.path.basename(filename))[0] + ".csv"
    p_mot_bar = config["P_mot_in"] / 1e5
    t_mot_c   = config["T_mot_in"] - 273.15
    p_suc_bar = config["P_suc_in"] / 1e5
    p_out_bar = config["P_out"]    / 1e5

    cse = f"""# COMMAND FILE:
#   CFX Post Version = 24.2
#   VCM Barotropic CO2 Ejector -- centerline extraction
#   Motive : {p_mot_bar:.3f} bar / {t_mot_c:.2f} C  (Total Pressure inlet)
#   Suction: {p_suc_bar:.3f} bar  (Static Pressure inlet)
#   Outlet : {p_out_bar:.3f} bar  (Opening)
#   VCM: MotiveCO2 (transport), SuctionCO2 (constraint)
#   Algebraic AVs: Hmix, Xmix, VfVapMix, HEMSoS
# END

POLYLINE: Centerline
  Boundary List = /BOUNDARY:*
  Method = Boundary Intersection
  Number of Samples = {n_points}
  POLYLINE DEFINITION:
    Point 1 = {x_min}, 0.0, 0.0
    Point 2 = {x_max}, 0.0, 0.0
  END
END

EXPORT:
  File Format = CSV
  File = {csv_name}
  Append = Off
  Export Polyline:
    Polyline = Centerline
    Boundary Values = Conservative
  EXPORT VARIABLES:
    Absolute Pressure = On
    Density = On
    Velocity = On
    MotiveCO2.Mass Fraction = On
    Hmix = On
    Xmix = On
    VfVapMix = On
    HEMSoS = On
  END
END
"""
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(cse)
    print(f"[Pipeline] CSE written: {filename}  ({n_points} points, centerline, UTF-8)")


# ================================================================================
# 5.  PIPELINE ORCHESTRATORS
# ================================================================================

def _build_case_name(config: dict) -> str:
    p_mot = config["P_mot_in"] / 1e5
    p_suc = config["P_suc_in"] / 1e5
    p_out = config["P_out"]    / 1e5
    return (
        f"TwoFluid_Pmot{p_mot:.1f}_Psuc{p_suc:.1f}_Pout{p_out:.1f}"
        .replace(".", "_")
    )


def run_two_fluid_pipeline(config: dict) -> tuple:
    """
    Full pre-processing pipeline for one operating point.

    Suction table P_max is set to P_mot_in (extended range) so that
    SuctionRho / SuctionMu are valid across the full mixing-section
    pressure range.  The isentrope entropy is still s(P_suc_in, T_suc_in).

    Returns
    -------
    (df_mot, df_suc, df_sat,
     excel_mot, excel_suc, excel_sat,
     ccl, cse, plot_mot, plot_suc)
    """
    case  = _build_case_name(config)
    fluid = config["Fluid"]

    excel_mot = f"{case}_Motive_Table.xlsx"
    excel_suc = f"{case}_Suction_Table.xlsx"
    excel_sat = f"{case}_Saturation_Table.xlsx"
    ccl_name  = f"{case}_Physics.ccl"
    cse_name  = f"{case}_PostProcess.cse"

    # 1. Motive isentrope table (standard: P_max = P_mot_in)
    df_mot, h0_mot = generate_equilibrium_table(
        P_in=config["P_mot_in"],
        T_in=config["T_mot_in"],
        fluid=fluid,
        label="Motive",
    )

    # 2. Suction isentrope table (EXTENDED: grid P_max = P_mot_in)
    #    Isentrope anchor remains at (P_suc_in, T_suc_in).
    #    The higher P_max makes SuctionRho/Mu valid in the mixing section
    #    where static pressure may exceed P_suc_in during compression.
    df_suc, h0_suc = generate_equilibrium_table(
        P_in  = config["P_suc_in"],
        T_in  = config["T_suc_in"],
        fluid = fluid,
        P_max = config["P_mot_in"],   # ← extended range
        label = "Suction",
    )

    if df_mot is None or df_suc is None:
        raise RuntimeError("Isentrope table generation failed.")

    # 3. Saturation table — dedicated dome-property grid for CCL functions
    df_sat = generate_saturation_table(
        P_max=config["P_mot_in"], fluid=fluid, label="sat",
    )

    print_table_summary(df_mot, "Motive")
    print_table_summary(df_suc, "Suction")
    print(f"\n  [h0] Motive  = {h0_mot/1e3:.3f} kJ/kg\n"
          f"  [h0] Suction = {h0_suc/1e3:.3f} kJ/kg")

    # 4. Excel audit files
    df_mot.to_excel(excel_mot, index=False, sheet_name="Motive_Isentrope")
    df_suc.to_excel(excel_suc, index=False, sheet_name="Suction_Isentrope")
    df_sat.to_excel(excel_sat, index=False, sheet_name="Saturation_Dome")
    print(f"[Pipeline] Excel: {excel_mot}, {excel_suc}, {excel_sat}")

    # 5. Verification plots
    plot_mot = f"{case}_SoS_Motive.png"
    plot_suc = f"{case}_SoS_Suction.png"
    try:
        fig = plot_sos_comparison(df_mot, label="Motive",  save_path=plot_mot)
        plt.close(fig)
        fig = plot_sos_comparison(df_suc, label="Suction (extended)",
                                  save_path=plot_suc)
        plt.close(fig)
    except Exception as exc:
        print(f"  [Warning] Verification plot failed: {exc}")
        plot_mot = plot_suc = None

    # 6. CCL
    write_two_fluid_ccl(
        ccl_name, df_mot, df_suc, df_sat, config,
        h0_mot=h0_mot, h0_suc=h0_suc,
    )

    # 7. CSE
    write_two_fluid_cse(cse_name, config)

    return (df_mot, df_suc, df_sat,
            excel_mot, excel_suc, excel_sat,
            ccl_name, cse_name, plot_mot, plot_suc)


def run_two_fluid_pipeline_in_folder(config: dict, base_dir: str = ".") -> str:
    """
    Run pipeline, move all outputs to a dedicated case folder, write run_case.sh.
    Returns absolute path to the bash script.
    """
    (df_mot, df_suc, df_sat,
     excel_mot, excel_suc, excel_sat,
     ccl, cse, plot_mot, plot_suc) = run_two_fluid_pipeline(config)

    case     = _build_case_name(config)
    case_dir = os.path.join(base_dir, case)
    os.makedirs(case_dir, exist_ok=True)

    all_files = [excel_mot, excel_suc, excel_sat, ccl, cse, plot_mot, plot_suc]
    for fname in all_files:
        if fname and os.path.exists(fname):
            shutil.move(fname, os.path.join(case_dir, os.path.basename(fname)))

    cfx_bin   = config.get("CFX_BIN",   "")
    node_str  = config.get("NODES",     "node-4-16*40")
    mem_flags = config.get("MEM_FLAGS", "-S 1.9 -sizepar 1.5")
    solver    = os.path.join(cfx_bin, "cfx5solve") if cfx_bin else "cfx5solve"
    post_bin  = os.path.join(cfx_bin, "cfx5post")  if cfx_bin else "cfx5post"
    res_base  = f"Result_{case}"

    bash = f"""#!/bin/bash
# ============================================================
# Auto-generated Two-Fluid run script
# Case  : {case}
# CCL   : {ccl}
#   &replace DOMAIN -- fully substitutes domain in .def
#   Reference Pressure = 0 [bar]
#   Suction table extends to P_mot_in for mixing-section validity
#   Enthalpy AVs: RhoMix -> Ymot -> Hmix -> Xmix[0,1] -> VfVapMix
# ============================================================

cd "{os.path.abspath(case_dir)}"

echo "[INFO] Starting CFX solver..."
{solver} -def ../base_setup.def \\
         -ccl {ccl} \\
         -double \\
         -start-method 'Intel MPI Local Parallel' \\
         -par-dist '{node_str}' \\
         {mem_flags} \\
         -name {res_base}

[ $? -ne 0 ] && {{ echo "[ERROR] Solver failed."; exit 1; }}

echo "[INFO] Locating result file..."
RES_FILE=$(ls -t {res_base}*.res 2>/dev/null | head -n 1)

[ -f "$RES_FILE" ] || {{ echo "[ERROR] No .res file found."; exit 1; }}

echo "[INFO] Post-processing: $RES_FILE"
{post_bin} -batch {cse} -res "$RES_FILE"
echo "[SUCCESS] Done."
"""

    bash_path = os.path.join(case_dir, "run_case.sh")
    with open(bash_path, "w", encoding="utf-8") as fh:
        fh.write(bash)
    os.chmod(bash_path, 0o755)

    print(f"\n[Pipeline] Case folder : {os.path.abspath(case_dir)}")
    print(f"[Pipeline] Run script  : {bash_path}")
    return bash_path