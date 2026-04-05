"""
BaroTFM.py
==========
Driver for the VCM Barotropic CO2 ejector pipeline.

All mesh location strings and boundary names are set to match
the latest geometry configuration.

Operating point:
    Motive : 115.599 bar / 45.161 C  (Total Pressure inlet)
    Suction:  37.472 bar              (Stagnation / Total Pressure inlet)
    Outlet :  46.998 bar              (Static Pressure Outlet)
"""

import os
import sys
sys.path.append(os.getcwd())

import two_fluid_pipeline as tfp


# ================================================================================
# SIMULATION CONFIGURATION
# ================================================================================

sim_config = {
    # -- Fluid ------------------------------------------------------------------
    "Fluid": "HEOS::CO2",

    # -- Motive inlet (total / stagnation conditions) ---------------------------
    "P_mot_in": 115.5989069e5,          # [Pa]
    "T_mot_in": 45.16075038 + 273.15,   # [K]

    # -- Suction inlet (static pressure) ----------------------------------------
    # NOTE: suction uses Static Pressure BC (verified from working .def)
    # T_suc_in is used only for the isentrope table generation, not in CCL
    "P_suc_in": 37.47196492e5,          # [Pa]
    "T_suc_in": 27.79283469 + 273.15,   # [K]

    # -- Outlet opening pressure -------------------------------------------------
    "P_out": 46.99754738e5,             # [Pa]

    # -- CFX domain (latest geometry configuration) -----------------------------
    "DOMAIN_NAME":      "Ejector",
    "DOMAIN_LOCATIONS": "DIFF,EXIT,MC,MC_DIFF,MC_DIV,MN,MN_in,SN",

    # -- Boundary names and mesh face tags (updated geometry) -------------------
    # Boundary NAMES  → must match .def string-for-string (spaces OK, not CEL)
    # Boundary LOCATIONS → mesh face tags from the .def Location = lines
    "MOT_BC_NAME":     "Inlet MN",
    "MOT_BC_LOCATION": "inlet_mn",

    "SUC_BC_NAME":     "Inlet SN",
    "SUC_BC_LOCATION": "inlet_sn",

    "OUT_BC_NAME":     "Outlet",
    "OUT_BC_LOCATION": "outlet",

    "WALL_BC_NAME": "Ejector Default",
    "WALL_LOCATIONS": (
        "Primitive 2D AI,Primitive 2D AJ,Primitive 2D AK,Primitive 2D AL,"
        "Primitive 2D AM,Primitive 2D AN,Primitive 2D AO,Primitive 2D AP,"
        "walls zone2d diff,walls zone2d exit,walls zone2d mc_diff,walls zone2d mc_div"
    ),

    "SYMM1_NAME":     "Symm 1",
    "SYMM1_LOCATION": "periodic.1 A",
    "SYMM2_NAME":     "Symm 2",
    "SYMM2_LOCATION": "periodic.1 B",

    # -- Isothermal fluid temperature (from working .def) -----------------------
    "FLUID_TEMPERATURE": 330.0,         # [K]

    # -- Solver / output control ------------------------------------------------
    # RAMPING_STEPS: list of (iteration_threshold, fraction) tuples.
    #   fraction is the proportion of the range Pout -> Psucin applied at each stage.
    #   Steps with fraction=1.0 are stripped — full Psucin is the implicit fallback.
    #   Empty list (or key absent) = no ramp, immediate full Psucin from iteration 1.
    #
    # Generated CEL:
    #   Pstart     = Pout   (discharge pressure — ramp anchor)
    #   PsucinRamp = if(aitern <= 1500, Pstart + 0.33*(Psucin-Pstart),
    #                   if(aitern <= 3000, Pstart + 0.66*(Psucin-Pstart), Psucin))
    "SOLVER_SETTINGS": {
        "MAX_ITER":        10000,
        "BACKUP_INTERVAL": 500,
        "RAMPING_STEPS": [
            (1500,  0.33),   # 33% of range at iteration 1500
            (3000,  0.66),   # 66% of range at iteration 3000
            (10000, 1.00),   # full target — stripped, becomes else-branch
        ],
    },

    # -- Cluster / solver -------------------------------------------------------
    "CFX_BIN":   "/software/ansys/v242/CFX/bin",
    "NODES":     "node-4-16*40",
    "MEM_FLAGS": "-S 1.9 -sizepar 1.5",
}


# ================================================================================
# RUN
# ================================================================================

run_script = tfp.run_two_fluid_pipeline_in_folder(sim_config)
