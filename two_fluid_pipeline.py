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
FLUID: <name> under BOUNDARY:     NOT under BOUNDARY CONDITIONS:
VOLUME FRACTION inside FLUID BC   same pattern as working .def
REFERENCE PRESSURE: in DOMAIN MODELS (bar, not Pa)
MULTIPHASE MODELS: Homogeneous Model = On  (flag, not Option = Homogeneous)
FLUID PAIR: is a top-level domain block, NOT inside FLUID MODELS
Outlet  → OPENING + Opening Pressure and Direction
Suc. inlet → Static Pressure  (motive → Total Pressure)
&replace DOMAIN: <name>           replaces entire domain block in .def

CEL NAMING (CFX legacy parser — underscores are internal delimiters)
---------------------------------------------------------------------
All FUNCTION, ADDITIONAL VARIABLE, MATERIAL, FLUID DEFINITION, and
EXPRESSION names are strictly alphanumeric.  BOUNDARY / DOMAIN names
and Python config keys are exempt — they are never CEL identifiers.

Author: auto-generated for Ahmed's cfx_writer / NEB pipeline
"""

from __future__ import annotations

import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from two_fluid_utils import (
    generate_equilibrium_table,
    print_table_summary,
    plot_sos_comparison,
)

_CO2_MOLAR_MASS = 44.01   # kg kmol^-1

# Wall face list extracted from base_setup.def — geometry-specific
_DEFAULT_WALL_LOCATION = (
    "F156.152,F171.37,F172.182,F183.37,F184.37,F185.54,F186.54,"
    "F187.43,F188.43,F189.43,F19.18,F20.18,F21.18,F318.216,"
    "F319.216,F320.216,F329.216,F330.216,F331.216,F332.216,"
    "F333.216,F352.43,F353.43,F354.43,F36.43,F368.18,F373.148,"
    "F374.148,F375.148,F376.148,F390.148,F391.148,F392.148,"
    "F393.148,F409.18,F410.18,F46.54,F55.43,F83.148"
)


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

# Material-property interpolation functions  →  MotiveRho, MotiveMu, etc.
_MAT_FUNC_CATALOGUE: dict[str, tuple[str, str]] = {
    "Rho": ("Rho", "kg m^-3"),
    "Mu":  ("Mu",  "kg m^-1 s^-1"),
}

# AV interpolation functions  →  fnMotiveAlphaV, fnSuctionSoS, etc.
# Keys are alphanumeric (compose CEL names); df column names may use underscores.
_AV_FUNC_CATALOGUE: dict[str, tuple[str, str]] = {
    "AlphaV": ("AlphaV",  "[]"),
    "SoS":    ("SoS",     "m s^-1"),
    "Xv":     ("X_v",     "[]"),
    "Xlsat":  ("X_l_sat", "[]"),
}


# ================================================================================
# 3.  CCL WRITER  (structure verified against working CFX-Pre 24.x .def)
# ================================================================================

def write_two_fluid_ccl(
    filename: str,
    df_mot:   pd.DataFrame,
    df_suc:   pd.DataFrame,
    config:   dict,
) -> None:
    """
    Write a complete &replace CCL patch for the Two-Fluid Barotropic ejector.

    The file contains two top-level blocks:

    LIBRARY
      ├── ADDITIONAL VARIABLE definitions  (MotiveAlphaV, SuctionSoS, ...)
      ├── MATERIAL: MotiveCO2              (ρ, μ = CEL function calls)
      ├── MATERIAL: SuctionCO2
      └── CEL
            ├── EXPRESSIONS  (Pmotin, Psucin, Pout — alphanumeric only)
            ├── FUNCTION: MotiveRho, MotiveMu, ...       (material tables)
            └── FUNCTION: fnMotiveAlphaV, fnSuctionSoS,... (AV tables)

    FLOW: Flow Analysis 1
      └── &replace DOMAIN: <DOMAIN_NAME>
            Domain Type = Fluid
            Location = <DOMAIN_LOCATION>
            BOUNDARY: <WALL_BC_NAME>        (wall — preserved from .def)
            BOUNDARY: <MOT_BC_NAME>         (motive inlet, Total Pressure)
            BOUNDARY: <SUC_BC_NAME>         (suction inlet, Static Pressure)
            BOUNDARY: <OUT_BC_NAME>         (opening outlet)
            BOUNDARY: <SYMM1_NAME>          (symmetry 1, if present)
            BOUNDARY: <SYMM2_NAME>          (symmetry 2, if present)
            DOMAIN MODELS
              └── REFERENCE PRESSURE: 0 [bar]
            FLUID DEFINITION: MotiveCO2
              Option = Material Library     ← verified token
              MORPHOLOGY: Continuous Fluid
            FLUID DEFINITION: SuctionCO2   (same)
            FLUID MODELS
              └── SST turbulence, isothermal heat transfer
            FLUID PAIR: MotiveCO2 | SuctionCO2
              └── no interphase transfer
            MULTIPHASE MODELS
              └── Homogeneous Model = On    ← flag form, verified

    Volume fraction BC syntax (verified from working .def):
      FLUID: MotiveCO2                      ← directly under BOUNDARY:
        BOUNDARY CONDITIONS:
          VOLUME FRACTION:
            Option = Value
            Volume Fraction = 1.0
          END
        END
      END

    Required config keys
    --------------------
    P_mot_in, T_mot_in         Motive  inlet stagnation [Pa], [K]
    P_suc_in, T_suc_in         Suction inlet static     [Pa], [K]
    P_out                      Outlet opening pressure  [Pa]
    DOMAIN_NAME                e.g. "Ejector"
    DOMAIN_LOCATION            mesh region string from .def
    MOT_BC_NAME / _LOCATION    motive inlet boundary name + mesh face tag
    SUC_BC_NAME / _LOCATION    suction inlet boundary name + mesh face tag
    OUT_BC_NAME / _LOCATION    outlet boundary name + mesh face tag
    WALL_BC_NAME / _LOCATION   wall boundary name + face list string
    SYMM1_NAME / _LOCATION     symmetry plane 1 (optional)
    SYMM2_NAME / _LOCATION     symmetry plane 2 (optional)
    FLUID_TEMPERATURE          isothermal temperature [K]  (default 330)
    """
    # ── Unpack config ──────────────────────────────────────────────────────────
    domain      = config.get("DOMAIN_NAME",      "Ejector")
    dom_loc     = config.get("DOMAIN_LOCATION",
                             "DIFF,EXIT,MC,MC_DIFF,MC_DIV,MN,MN_in,SN")

    bc_mot      = config.get("MOT_BC_NAME",      "Inlet MN")
    bc_mot_loc  = config.get("MOT_BC_LOCATION",  "Inlet_MN")
    bc_suc      = config.get("SUC_BC_NAME",      "Inlet SN")
    bc_suc_loc  = config.get("SUC_BC_LOCATION",  "Inlet_SN")
    bc_out      = config.get("OUT_BC_NAME",      "Outlet")
    bc_out_loc  = config.get("OUT_BC_LOCATION",  "Outlet")
    bc_wall     = config.get("WALL_BC_NAME",     "Ejector Default")
    bc_wall_loc = config.get("WALL_LOCATION",    _DEFAULT_WALL_LOCATION)
    symm1       = config.get("SYMM1_NAME",       "Symm 1")
    symm1_loc   = config.get("SYMM1_LOCATION",   "Sym_1")
    symm2       = config.get("SYMM2_NAME",       "Symm 2")
    symm2_loc   = config.get("SYMM2_LOCATION",   "Sym_2")
    T_fluid     = config.get("FLUID_TEMPERATURE", 330.0)

    P_mot = config["P_mot_in"]
    T_mot = config["T_mot_in"]
    P_suc = config["P_suc_in"]
    P_out = config["P_out"]

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

    def _write_vf_fluid(f, fluid_name, vf_value, indent="    "):
        """Write a FLUID: volume-fraction block directly under BOUNDARY: (verified syntax)."""
        pad = indent
        if vf_value == "Zero Gradient":
            f.write(f"{pad}FLUID: {fluid_name}\n")
            f.write(f"{pad}  BOUNDARY CONDITIONS:\n")
            f.write(f"{pad}    VOLUME FRACTION:\n")
            f.write(f"{pad}      Option = Zero Gradient\n")
            f.write(f"{pad}    END\n")
            f.write(f"{pad}  END\n")
            f.write(f"{pad}END\n")
        else:
            f.write(f"{pad}FLUID: {fluid_name}\n")
            f.write(f"{pad}  BOUNDARY CONDITIONS:\n")
            f.write(f"{pad}    VOLUME FRACTION:\n")
            f.write(f"{pad}      Option = Value\n")
            f.write(f"{pad}      Volume Fraction = {vf_value}\n")
            f.write(f"{pad}    END\n")
            f.write(f"{pad}  END\n")
            f.write(f"{pad}END\n")

    # ── open UTF-8 ─────────────────────────────────────────────────────────────
    with open(filename, "w", encoding="utf-8") as f:

        # ======================================================================
        # FILE HEADER
        # ======================================================================
        f.write("# ============================================================\n")
        f.write("# Two-Fluid Barotropic CCL  --  auto-generated\n")
        f.write(f"# Motive : {P_mot/1e5:.3f} bar / {T_mot-273.15:.2f} C\n")
        f.write(f"# Suction: {P_suc/1e5:.3f} bar (static)\n")
        f.write(f"# Outlet : {P_out/1e5:.3f} bar (opening)\n")
        f.write(f"# Domain : {domain}\n")
        f.write("# Model  : Eulerian Homogeneous Multiphase, HEM SoS\n")
        f.write("# Schema : verified against CFX-Pre 24.x working .def\n")
        f.write("# ============================================================\n\n")

        # ======================================================================
        # LIBRARY
        # ======================================================================
        f.write("LIBRARY:\n\n")

        # -- AV definitions (alphanumeric names) --------------------------------
        f.write("  # -- Per-phase Algebraic Additional Variables ---------------\n")
        for phase, _ in _phases:
            for key, (_, units) in _AV_FUNC_CATALOGUE.items():
                _write_av_def(f, f"{phase}{key}", units)  # e.g. MotiveAlphaV

        # -- MATERIAL blocks (&replace overwrites existing library entry) ---------
        for phase, _ in _phases:
            mat = f"{phase}CO2"
            f.write(f"\n  # -- Material: {mat} ------------------------------------\n")
            f.write(f"  &replace MATERIAL: {mat}\n")
            f.write(f"    Material Description = Barotropic CO2 {phase} equilibrium isentrope\n")
            f.write( "    Material Group = User\n")
            f.write( "    Object Origin = User\n")
            f.write( "    Option = Pure Substance\n")
            f.write( "    PROPERTIES:\n")
            f.write( "      Option = General Material\n")
            # Order: DYNAMIC VISCOSITY → EQUATION OF STATE → SPECIFIC HEAT
            # (matches verified working CCL field order)
            f.write( "      DYNAMIC VISCOSITY:\n")
            f.write(f"        Dynamic Viscosity = {phase}Mu(Absolute Pressure)\n")
            f.write( "        Option = Value\n")
            f.write( "      END\n")
            f.write( "      EQUATION OF STATE:\n")
            f.write(f"        Density = {phase}Rho(Absolute Pressure)\n")
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

        # -- CEL block ---------------------------------------------------------
        f.write("\n  CEL:\n")
        f.write("    EXPRESSIONS:\n")
        # All expression names strictly alphanumeric
        f.write(f"      Pmotin = {P_mot/1e5:.6f} [bar]\n")
        f.write(f"      Tmotin = {T_mot:.4f} [K]\n")
        f.write(f"      Psucin = {P_suc/1e5:.6f} [bar]\n")
        f.write(f"      Pout   = {P_out/1e5:.6f} [bar]\n")
        f.write("    END\n")

        # Material-property interpolation functions
        f.write("\n    # -- Material property functions (rho, mu per phase) ----\n")
        for phase, df_phase in _phases:
            for key, (col, units) in _MAT_FUNC_CATALOGUE.items():
                _write_func(f, f"{phase}{key}", df_phase, col, units)

        # AV interpolation functions (fn prefix keeps them distinct from AV names)
        f.write("\n    # -- AV functions (per-phase thermodynamic state) --------\n")
        for phase, df_phase in _phases:
            for key, (col, units) in _AV_FUNC_CATALOGUE.items():
                _write_func(f, f"fn{phase}{key}", df_phase, col, units)

        f.write("  END\n")   # CEL
        f.write("END\n\n")  # LIBRARY

        # ======================================================================
        # FLOW  (with &replace to fully substitute the domain in the .def)
        # ======================================================================
        f.write("# ============================================================\n")
        f.write(f"# &replace DOMAIN: {domain}\n")
        f.write("# ============================================================\n")
        f.write("FLOW: Flow Analysis 1\n")
        f.write(f"  &replace DOMAIN: {domain}\n")
        f.write( "    Coord Frame = Coord 0\n")
        f.write( "    Domain Type = Fluid\n")
        f.write(f"    Location = {dom_loc}\n")

        # -- Wall boundary (preserved from .def) --------------------------------
        f.write(f"\n    BOUNDARY: {bc_wall}\n")
        f.write( "      Boundary Type = WALL\n")
        f.write( "      Create Other Side = Off\n")
        f.write( "      Interface Boundary = Off\n")
        f.write(f"      Location = {bc_wall_loc}\n")
        f.write( "      BOUNDARY CONDITIONS:\n")
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = No Slip Wall\n")
        f.write( "        END\n")
        f.write( "        WALL ROUGHNESS:\n")
        f.write( "          Option = Smooth Wall\n")
        f.write( "        END\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # -- Motive inlet (Total Pressure) --------------------------------------
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
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = Total Pressure\n")
        f.write( "          Relative Pressure = Pmotin\n")
        f.write( "        END\n")
        f.write( "        TURBULENCE:\n")
        f.write( "          Option = Medium Intensity and Eddy Viscosity Ratio\n")
        f.write( "        END\n")
        f.write( "      END\n")
        # Volume fractions — FLUID: block directly under BOUNDARY: (verified)
        _write_vf_fluid(f, "MotiveCO2",  "1.0")
        _write_vf_fluid(f, "SuctionCO2", "0.0")
        f.write( "    END\n")

        # -- Suction inlet (Static Pressure) ------------------------------------
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
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = Static Pressure\n")   # ← Static, not Total
        f.write( "          Relative Pressure = Psucin\n")
        f.write( "        END\n")
        f.write( "        TURBULENCE:\n")
        f.write( "          Option = Medium Intensity and Eddy Viscosity Ratio\n")
        f.write( "        END\n")
        f.write( "      END\n")
        _write_vf_fluid(f, "MotiveCO2",  "0.0")
        _write_vf_fluid(f, "SuctionCO2", "1.0")
        f.write( "    END\n")

        # -- Outlet (Opening — allows reverse flow) -----------------------------
        f.write(f"\n    BOUNDARY: {bc_out}\n")
        f.write( "      Boundary Type = OPENING\n")      # ← OPENING, not OUTLET
        f.write( "      Interface Boundary = Off\n")
        f.write(f"      Location = {bc_out_loc}\n")
        f.write( "      BOUNDARY CONDITIONS:\n")
        f.write( "        FLOW DIRECTION:\n")
        f.write( "          Option = Normal to Boundary Condition\n")
        f.write( "        END\n")
        f.write( "        FLOW REGIME:\n")
        f.write( "          Option = Subsonic\n")
        f.write( "        END\n")
        f.write( "        MASS AND MOMENTUM:\n")
        f.write( "          Option = Opening Pressure and Direction\n")  # ← verified
        f.write( "          Relative Pressure = Pout\n")
        f.write( "        END\n")
        f.write( "        TURBULENCE:\n")
        f.write( "          Option = Medium Intensity and Eddy Viscosity Ratio\n")
        f.write( "        END\n")
        f.write( "      END\n")
        _write_vf_fluid(f, "MotiveCO2",  "Zero Gradient")
        _write_vf_fluid(f, "SuctionCO2", "Zero Gradient")
        f.write( "    END\n")

        # -- Symmetry planes ----------------------------------------------------
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

        # -- DOMAIN MODELS ------------------------------------------------------
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
        # Reference pressure placement verified from working .def
        f.write( "      REFERENCE PRESSURE:\n")
        f.write( "        Reference Pressure = 0 [bar]\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # -- FLUID DEFINITIONS --------------------------------------------------
        # Option = Material Library  ← verified token (not "Material", not "Mixture Component")
        # MORPHOLOGY: sub-block      ← verified (not inline Morphology Option =)
        for phase, _ in _phases:
            mat = f"{phase}CO2"
            f.write(f"\n    FLUID DEFINITION: {mat}\n")
            f.write(f"      Material = {mat}\n")
            f.write( "      Option = Material Library\n")
            f.write( "      MORPHOLOGY:\n")
            f.write( "        Option = Continuous Fluid\n")
            f.write( "      END\n")
            f.write( "    END\n")

        # -- FLUID MODELS -------------------------------------------------------
        f.write("\n    FLUID MODELS:\n")

        # Step 1: Declare every AV as Fluid Dependent at the top of FLUID MODELS
        # (verified pattern from working CCL — all 8 AVs declared before any FLUID block)
        f.write("      # -- AV declarations (Fluid Dependent) -------------------\n")
        for phase, _ in _phases:
            for key in _AV_FUNC_CATALOGUE:
                av_name = f"{phase}{key}"
                f.write(f"      ADDITIONAL VARIABLE: {av_name}\n")
                f.write( "        Option = Fluid Dependent\n")
                f.write( "      END\n")

        f.write( "      COMBUSTION MODEL:\n")
        f.write( "        Option = None\n")
        f.write( "      END\n")

        # Step 2: Per-fluid FLUID: block with Algebraic Equation for that phase's AVs
        # (verified: Motive AVs → FLUID: MotiveCO2, Suction AVs → FLUID: SuctionCO2)
        f.write("      # -- Per-fluid AV algebraic equations --------------------\n")
        for phase, _ in _phases:
            mat = f"{phase}CO2"
            f.write(f"      FLUID: {mat}\n")
            for key in _AV_FUNC_CATALOGUE:
                av_name   = f"{phase}{key}"
                func_call = f"fn{phase}{key}(Absolute Pressure)"
                f.write(f"        ADDITIONAL VARIABLE: {av_name}\n")
                f.write(f"          Additional Variable Value = {func_call}\n")
                f.write( "          Option = Algebraic Equation\n")
                f.write( "        END\n")
            f.write( "      END\n")   # FLUID

        f.write( "      HEAT TRANSFER MODEL:\n")
        f.write(f"        Fluid Temperature = {T_fluid:.1f} [K]\n")
        f.write( "        Homogeneous Model = On\n")
        f.write( "        Option = Isothermal\n")
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

        # -- FLUID PAIR (top-level domain block — verified) ---------------------
        f.write("\n    FLUID PAIR: MotiveCO2 | SuctionCO2\n")
        f.write( "      INTERPHASE TRANSFER MODEL:\n")
        f.write( "        Option = None\n")
        f.write( "      END\n")
        f.write( "      MASS TRANSFER:\n")
        f.write( "        Option = None\n")
        f.write( "      END\n")
        f.write( "    END\n")

        # -- MULTIPHASE MODELS (top-level domain block — verified) --------------
        # Homogeneous Model = On  is a FLAG, not "Option = Homogeneous"
        f.write("\n    MULTIPHASE MODELS:\n")
        f.write( "      Homogeneous Model = On\n")
        f.write( "      FREE SURFACE MODEL:\n")
        f.write( "        Option = None\n")
        f.write( "      END\n")
        f.write( "    END\n")

        f.write( "  END\n")   # DOMAIN
        f.write( "END\n")    # FLOW

    n_av  = 2 * len(_AV_FUNC_CATALOGUE)
    n_mat = 2 * len(_MAT_FUNC_CATALOGUE)
    print(f"[Pipeline] CCL written: {filename}  "
          f"({n_mat} material funcs + {n_av} AV funcs, UTF-8, &replace)")


# ================================================================================
# 4.  CSE POST-PROCESSING SCRIPT
# ================================================================================

def write_two_fluid_cse(
    filename: str,
    config:   dict,
    x_min:    float = 0.0,
    x_max:    float = 0.0835,
    n_slices: int   = 100,
) -> None:
    """
    Generate a CFX-Post CSE script for Two-Fluid axial profile extraction.

    Volume Fraction variable names in CFX-Post follow the fluid definition
    names exactly:  Volume Fraction.MotiveCO2  and  Volume Fraction.SuctionCO2

    AV names in evaluate() must match ADDITIONAL VARIABLE names in the CCL
    (alphanumeric, no underscores).

    Mixture reconstruction (Perl arithmetic):
        AlphaVmix = VFMotive * MotiveAlphaV + VFSuction * SuctionAlphaV
        SoSmix    = VFMotive * MotiveSoS    + VFSuction * SuctionSoS
    """
    csv_name  = os.path.splitext(os.path.basename(filename))[0] + ".csv"
    p_mot_bar = config["P_mot_in"] / 1e5
    t_mot_c   = config["T_mot_in"] - 273.15
    p_suc_bar = config["P_suc_in"] / 1e5
    p_out_bar = config["P_out"]    / 1e5

    cse = f"""# COMMAND FILE:
#   CFX Post Version = 24.2
#   Two-Fluid Barotropic CO2 Ejector -- axial profile extraction
#   Motive : {p_mot_bar:.3f} bar / {t_mot_c:.2f} C  (Total Pressure inlet)
#   Suction: {p_suc_bar:.3f} bar  (Static Pressure inlet)
#   Outlet : {p_out_bar:.3f} bar  (Opening)
#   Fluid names: MotiveCO2, SuctionCO2
#   AV names   : MotiveAlphaV, SuctionSoS, etc. (alphanumeric)
# END

!$x_min = {x_min};
!$x_max = {x_max};
!$n     = {n_slices};

!for ($i = 1; $i < $n; $i++) {{

    !$x_val = $x_min + ($x_max - $x_min) * $i / $n;
    !$row   = $i + 1;

    ISOSURFACE: X_slice
        Domain List = /DOMAIN GROUP:All Domains
        Variable    = X
        Value       = $x_val
        Apply Instancing Transform = On
        Range       = Global
    END

    # Primitive flow variables
    !($p_stat, $u) = evaluate("ave(Absolute Pressure)\\@X_slice");
    !($rho,    $u) = evaluate("ave(Density)\\@X_slice");
    !($vel,    $u) = evaluate("ave(Velocity)\\@X_slice");

    # Volume fractions — CFX-Post variable name includes fluid definition name
    !($vf_mot, $u) = evaluate("ave(Volume Fraction.MotiveCO2)\\@X_slice");
    !($vf_suc, $u) = evaluate("ave(Volume Fraction.SuctionCO2)\\@X_slice");

    # Per-phase thermodynamic AVs (alphanumeric names match CCL exactly)
    !($mot_av,  $u) = evaluate("ave(MotiveAlphaV)\\@X_slice");
    !($suc_av,  $u) = evaluate("ave(SuctionAlphaV)\\@X_slice");
    !($mot_sos, $u) = evaluate("ave(MotiveSoS)\\@X_slice");
    !($suc_sos, $u) = evaluate("ave(SuctionSoS)\\@X_slice");
    !($mot_xv,  $u) = evaluate("ave(MotiveXv)\\@X_slice");
    !($suc_xv,  $u) = evaluate("ave(SuctionXv)\\@X_slice");
    !($mot_xl,  $u) = evaluate("ave(MotiveXlsat)\\@X_slice");
    !($suc_xl,  $u) = evaluate("ave(SuctionXlsat)\\@X_slice");

    # Mixture reconstruction (volume-fraction weighted)
    !$av_mix  = $vf_mot * $mot_av  + $vf_suc * $suc_av;
    !$sos_mix = $vf_mot * $mot_sos + $vf_suc * $suc_sos;
    !if ($sos_mix > 1.0) {{ $mach = $vel / $sos_mix; }} else {{ $mach = 0.0; }}

    !if ($i == 1) {{
        TABLE: TwoFluid_Profile
          Table Exists = True
          TABLE CELLS:
            A1 = "X [m]",           False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            B1 = "Pstat [Pa]",      False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            C1 = "Rho_mix [kg/m3]", False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            D1 = "Vel [m/s]",       False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            E1 = "SoS_mix [m/s]",   False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            F1 = "Mach [-]",        False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            G1 = "VF_Motive [-]",   False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            H1 = "VF_Suction [-]",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            I1 = "AlphaV_mot [-]",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            J1 = "AlphaV_suc [-]",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            K1 = "AlphaV_mix [-]",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            L1 = "Xv_mot [-]",      False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            M1 = "Xv_suc [-]",      False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            N1 = "Xlsat_mot [-]",   False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
            O1 = "Xlsat_suc [-]",   False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
          END
        END
    !}}

    TABLE: TwoFluid_Profile
      Table Exists = True
      TABLE CELLS:
        A$row = "$x_val",   False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        B$row = "$p_stat",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        C$row = "$rho",     False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        D$row = "$vel",     False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        E$row = "$sos_mix", False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        F$row = "$mach",    False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        G$row = "$vf_mot",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        H$row = "$vf_suc",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        I$row = "$mot_av",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        J$row = "$suc_av",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        K$row = "$av_mix",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        L$row = "$mot_xv",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        M$row = "$suc_xv",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        N$row = "$mot_xl",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
        O$row = "$suc_xl",  False, False, False, Left, True, 0, Font Name, 1|1, %10.4e, True, ffffff, 000000, True
      END
    END

!}}

>table save={csv_name}, name=TwoFluid_Profile
"""
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(cse)
    print(f"[Pipeline] CSE written: {filename}  ({n_slices} slices, 15 cols, UTF-8)")


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

    Returns
    -------
    (df_mot, df_suc, excel_mot, excel_suc, ccl, cse, plot_mot, plot_suc)
    """
    case  = _build_case_name(config)
    fluid = config["Fluid"]

    excel_mot = f"{case}_Motive_Table.xlsx"
    excel_suc = f"{case}_Suction_Table.xlsx"
    ccl_name  = f"{case}_Physics.ccl"
    cse_name  = f"{case}_PostProcess.cse"

    df_mot = generate_equilibrium_table(
        P_in=config["P_mot_in"], T_in=config["T_mot_in"],
        fluid=fluid, label="Motive",
    )
    df_suc = generate_equilibrium_table(
        P_in=config["P_suc_in"], T_in=config["T_suc_in"],
        fluid=fluid, label="Suction",
    )

    if df_mot is None or df_suc is None:
        raise RuntimeError("Table generation failed.")

    print_table_summary(df_mot, "Motive")
    print_table_summary(df_suc, "Suction")

    df_mot.to_excel(excel_mot, index=False, sheet_name="Motive_Isentrope")
    df_suc.to_excel(excel_suc, index=False, sheet_name="Suction_Isentrope")
    print(f"[Pipeline] Excel: {excel_mot}, {excel_suc}")

    plot_mot = f"{case}_SoS_Motive.png"
    plot_suc = f"{case}_SoS_Suction.png"
    try:
        fig = plot_sos_comparison(df_mot, label="Motive",  save_path=plot_mot)
        plt.close(fig)
        fig = plot_sos_comparison(df_suc, label="Suction", save_path=plot_suc)
        plt.close(fig)
    except Exception as exc:
        print(f"  [Warning] Verification plot failed: {exc}")
        plot_mot = plot_suc = None

    write_two_fluid_ccl(ccl_name, df_mot, df_suc, config)
    write_two_fluid_cse(cse_name, config)

    return df_mot, df_suc, excel_mot, excel_suc, ccl_name, cse_name, plot_mot, plot_suc


def run_two_fluid_pipeline_in_folder(config: dict, base_dir: str = ".") -> str:
    """
    Run pipeline, move outputs to case folder, write run_case.sh.
    Returns absolute path to the bash script.
    """
    df_mot, df_suc, excel_mot, excel_suc, ccl, cse, plot_mot, plot_suc = \
        run_two_fluid_pipeline(config)

    case     = _build_case_name(config)
    case_dir = os.path.join(base_dir, case)
    os.makedirs(case_dir, exist_ok=True)

    for fname in [excel_mot, excel_suc, ccl, cse, plot_mot, plot_suc]:
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
#   Uses &replace DOMAIN — fully substitutes domain in .def
#   Reference Pressure = 0 bar (Absolute Pressure tables)
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