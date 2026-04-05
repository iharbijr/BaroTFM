# Ejector-BaroTFM: High-Fidelity Variable Composition Pipeline

A Python-to-Ansys CFX pipeline designed for simulating **Two-Phase Supersonic CO2 Ejectors**. This framework has been upgraded to a **Variable Composition Mixture (VCM)** architecture. It tracks motive and suction streams via species mass fraction transport equations driven by turbulent diffusion, using an Isothermal solver hack coupled with algebraic mass-weighted enthalpy conservation to resolve the mixing zone thermodynamics natively.

## 🚀 Key Technical Features

* **Variable Composition Mixture (VCM)**: Replaces standard multiphase models with a single bulk mixture. Component mass fractions ($Y_{mot}$, $Y_{suc}$) are governed by transport equations and algebraic constraints, accurately coupling mass diffusion to the GEKO/SST turbulence models.
* **Solver-Level Enthalpy Reconstruction**: Implements a 5-step Algebraic Additional Variable (AV) chain (`Hmix` $\rightarrow$ `XmixRaw` $\rightarrow$ `Xmix` $\rightarrow$ `VfVapMix`, `VfLiqMix`) directly in the domain's `FLUID MODELS` block to reconstruct the mixture thermodynamic state.
* **Explicit Unit-Stripping**: Bypasses known Ansys CFX CEL parser bugs by explicitly non-dimensionalizing all compound algebraic fractions (e.g., dividing by base units `[kg m^-3]`, `[m s^-1]`) before explicitly reapplying the final tensor unit.
* **Energy Matrix Bypass**: Forces an `Isothermal` heat transfer model combined with the undocumented `solve energy = f` expert parameter. This completely disables the internal energy matrix, drastically reducing computational overhead while the AV chain handles the true barotropic thermodynamics.
* **HEM Speed of Sound**: Features a fully-relaxed Homogeneous Equilibrium Model (HEM) formulation (`HEMMech`, `HEMTherm`, `HEMSoS`), including thermal relaxation terms derived from saturation entropy derivatives ($ds/dp$).
* **Safe CCL Generation**: Python pipeline automatically wraps long mesh boundary `Location` arrays using CFX continuation characters (`\`) to prevent legacy parser character-limit overflows.

## 📖 Mathematical Background

### 1. Mixture Energetics
Because the solver's energy equation is disabled, the local static enthalpy $h_{mix}$ is reconstructed algebraically from the transported mass fractions and the local velocity magnitude:

$$h_{mix} = (Y_{mot} h_{0,mot} + Y_{suc} h_{0,suc}) - \frac{1}{2} |\mathbf{U}|^2$$

### 2. Fully-Relaxed Acoustics (HEM)
The equilibrium speed of sound $c_{eq}$ captures the drastic change in compressibility within the two-phase region:

$$\frac{1}{\rho_{mix} c_{eq}^2} = \frac{\alpha_l}{\rho_l c_l^2} + \frac{\alpha_v}{\rho_v c_v^2} + T \left[ \frac{\alpha_l \rho_l}{C_{p,l}} \left( \frac{ds_l}{dp} \right)^2 + \frac{\alpha_v \rho_v}{C_{p,v}} \left( \frac{ds_v}{dp} \right)^2 \right]$$

## 🛠 Repository Structure

* **`two_fluid_utils.py`**: The thermodynamic engine using CoolProp (HEOS) to generate stretched pressure grids and single-phase/HEM building blocks.
* **`two_fluid_pipeline.py`**: The core logic handler that translates Python DataFrames into strictly validated, parser-safe CFX CCL and CSE scripts.
* **`BaroTFM.py`**: The master driver script for single-point runs or parametric sweeps, including GEKO/SST turbulence configuration and boundary location mapping.
* **`BaroTFM.tex`**: Comprehensive technical manual documenting the VCM theory and CFX workaround implementation.