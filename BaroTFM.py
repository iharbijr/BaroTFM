"""
Master_TwoFluid.py
==================
Driver for the Two-Fluid Barotropic CO2 ejector pipeline.

All mesh location strings and boundary names are set to match
the values parsed from the working base_setup.def exactly.

Operating point:
    Motive : 115.599 bar / 45.161 C  (Total Pressure inlet)
    Suction:  37.472 bar              (Static Pressure inlet)
    Outlet :  46.998 bar              (Opening boundary)
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

    # -- CFX domain (must match base_setup.def exactly) -------------------------
    "DOMAIN_NAME":     "Ejector",
    "DOMAIN_LOCATION": "DIFF,EXIT,MC,MC_DIFF,MC_DIV,MN,MN_in,SN",

    # -- Boundary names and mesh face tags (from working .def) ------------------
    # Boundary NAMES  → must match .def string-for-string (spaces OK, not CEL)
    # Boundary LOCATIONS → mesh face tags from the .def Location = lines
    "MOT_BC_NAME":     "Inlet MN",
    "MOT_BC_LOCATION": "Inlet_MN",

    "SUC_BC_NAME":     "Inlet SN",
    "SUC_BC_LOCATION": "Inlet_SN",

    "OUT_BC_NAME":     "Outlet",
    "OUT_BC_LOCATION": "Outlet",

    "WALL_BC_NAME": "Ejector Default",
    "WALL_LOCATION": (
        "F156.152,F171.37,F172.182,F183.37,F184.37,F185.54,F186.54,"
        "F187.43,F188.43,F189.43,F19.18,F20.18,F21.18,F318.216,"
        "F319.216,F320.216,F329.216,F330.216,F331.216,F332.216,"
        "F333.216,F352.43,F353.43,F354.43,F36.43,F368.18,F373.148,"
        "F374.148,F375.148,F376.148,F390.148,F391.148,F392.148,"
        "F393.148,F409.18,F410.18,F46.54,F55.43,F83.148"
    ),

    "SYMM1_NAME":     "Symm 1",
    "SYMM1_LOCATION": "Sym_1",
    "SYMM2_NAME":     "Symm 2",
    "SYMM2_LOCATION": "Sym_2",

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

# os.system(run_script)              # sequential local run
# os.system(f"sbatch {run_script}")  # SLURM submission


# ================================================================================
# PARAMETRIC SWEEP  (uncomment to use)
# ================================================================================
#
# for P_mot, T_mot in [(110e5, 316.15), (115.6e5, 318.31), (120e5, 320.5)]:
#     for P_suc in [35e5, 37.5e5, 40e5]:
#         cfg = sim_config.copy()
#         cfg["P_mot_in"] = P_mot
#         cfg["T_mot_in"] = T_mot
#         cfg["P_suc_in"] = P_suc
#         tfp.run_two_fluid_pipeline_in_folder(cfg)