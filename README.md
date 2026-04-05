
# Ejector-BaroTFM: High-Fidelity Two-Fluid Pipeline

A Python-to-Ansys CFX pipeline designed for simulating **Two-Phase Supersonic CO2 Ejectors**. This framework implements a **Barotropic Two-Fluid Model (TFM)** that treats motive and suction streams as distinct continuous phases, using mass-weighted enthalpy conservation to resolve the mixing zone.

## 🚀 Key Technical Features

* **Solver-Level Enthalpy Reconstruction**: Implements a _5-step Algebraic Additional Variable_ (AV) chain (`RhoMix` $\rightarrow$ `Ymot` $\rightarrow$ `Hmix` $\rightarrow$ `Xmix` $\rightarrow$ `VfVapMix`) to reconstruct the mixture thermodynamic state from conserved stagnation enthalpies.
* **Iteration-Based Pressure Ramping**: Utilizes a dynamic, parameterized CEL function linked to the CFX variable `aitern` (accumulated iteration number). This gradually transitions the suction inlet from discharge pressure to the target vacuum, ensuring numerical stability.
* **HEM Speed of Sound**: Features a fully-relaxed Homogeneous Equilibrium Model (HEM) formulation, including thermal relaxation terms derived from saturation entropy derivatives ($ds/dp$).
* **Extended Suction Table Range**: Property tables for the suction fluid are automatically extended to the motive inlet pressure ($P_{mot,in}$). This ensures property validity during compression waves and expansion fans within the mixing section.
* **Validated CEL Compliance**: Enforces a strict alphanumeric naming convention (no underscores) for all Material, Function, and AV names to bypass legacy CFX parser restrictions.

## 📖 Mathematical Background

### 1. Mixture Energetics
While individual fluids expand along isentropic paths, the mixing process is governed by the conservation of total enthalpy. The mixture static enthalpy $h_{mix}$ is computed by subtracting kinetic energy from the mass-weighted stagnation enthalpies:

$$h_{mix} = (Y_{mot} h_{0,mot} + Y_{suc} h_{0,suc}) - \frac{1}{2} |\mathbf{U}|^2$$

### 2. Fully-Relaxed Acoustics (HEM)
The equilibrium speed of sound $c_{eq}$ captures the change in compressibility within the two-phase region:

$$\frac{1}{\rho c_{eq}^2} = \frac{\alpha_l}{\rho_l c_l^2} + \frac{\alpha_v}{\rho_v c_v^2} + T \left[ \frac{\alpha_l \rho_l}{C_{p,l}} \left( \frac{ds_l}{dp} \right)^2 + \frac{\alpha_v \rho_v}{C_{p,v}} \left( \frac{ds_v}{dp} \right)^2 \right]$$

## 🛠 Repository Structure

* **`two_fluid_utils.py`**: The thermodynamic engine using CoolProp (HEOS) to generate stretched pressure grids (dense band $\pm 5$ bar around $P_{crit}$) and HEM building blocks.
* **`two_fluid_pipeline.py`**: The core logic handler that translates Python DataFrames into validated CFX CCL and CSE scripts.
* **`BaroTFM.py`**: The master driver script for single-point runs or parametric sweeps, including boundary location mapping.
* **`BaroTFM.tex`**: Comprehensive technical manual documenting the theory and implementation.

## 💻 Solver Control & Usage

### Pressure Ramping Strategy
The pipeline generates nested `if` expressions to manage the suction inlet pressure (`PsucinRamp`) based on the current iteration count (`aitern`):
* **Stage 1**: `aitern` $\le 1500$ (Startup stabilization).
* **Stage 2**: $1500 < $ `aitern` $\le 3000$ (Intermediate compression).
* **Stage 3**: `aitern` $> 3000$ (Target vacuum).

### Quick Start
1.  Configure your operating point in `BaroTFM.py`.
2.  Run the script to generate the alphanumeric `Physics.ccl` and axial-profile `PostProcess.cse`.
3.  Import the CCL into CFX-Pre and ensure the domain name matches your `DOMAIN_NAME` config.