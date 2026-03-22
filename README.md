# CO2-Ejector-BaroTFM: Two-Fluid Barotropic Pipeline

A specialized Python-to-Ansys CFX pipeline designed for the simulation of two-phase supersonic CO2 ejectors. This framework implements a **Barotropic Two-Fluid Model (TFM)**, treating the motive and suction streams as distinct continuous phases that expand isentropically and mix according to the conservation of total enthalpy.

## 🚀 Key Features

* **Automated CFX Setup**: Generates complete CFX Command Language (CCL) files including material definitions, 1D interpolation functions, and multiphase domain configurations.
* **HEM Speed of Sound**: Implements the Fully-Relaxed Homogeneous Equilibrium Model (HEM) for speed of sound, accounting for mechanical compressibility and thermal relaxation terms ($ds/dp$).
* **Enthalpy-Based Mixture Reconstruction**: Utilizes solver-level Algebraic Additional Variables to reconstruct mixture quality ($x_{mix}$) and vapor volume fractions based on mass-weighted enthalpy conservation.
* **CEL Compliance**: Enforces strict alphanumeric naming conventions for all solver variables to ensure compatibility with the Ansys CFX Expression Language.
* **High-Resolution Thermodynamics**: Uses a stretched pressure grid to resolve sharp property gradients near the CO2 critical point ($73.8$ bar).

## 🛠 Project Structure

* **`two_fluid_utils.py`**: The core thermodynamic engine. It utilizes CoolProp (HEOS) to generate equilibrium isentropes and saturation property tables ($h_l, h_v, \rho_l, \rho_v$).
* **`two_fluid_pipeline.py`**: The bridge between Python and CFX. It handles the generation of 1D data pairs for interpolation and writes the `.ccl` and `.cse` scripts.
* **`BaroTFM.py`**: Documentation generator and master logic for the Barotropic Two-Fluid Model theory.

## 📖 Theoretical Framework

### 1. The Barotropic Assumption
Properties for each fluid ($\rho, \mu$) are pre-calculated as functions of pressure along path-specific isentropes ($s_{mot}$ and $s_{suc}$).

### 2. Enthalpy-Based Mixing (Solver-Level)
The local state of the mixture is determined by the conservation of total enthalpy ($h_0$):
* **Mass Fraction ($Y_{mot}$)**: Derived from the solver's volume fractions and path densities.
* **Static Enthalpy ($h_{mix}$)**: $h_{mix} = (Y_{mot}h_{0,mot} + Y_{suc}h_{0,suc}) - 0.5|\mathbf{U}|^2$.
* **Reconstructed Quality ($x_{mix}$)**: Determined by comparing $h_{mix}$ to the saturation enthalpies $h_l(P)$ and $h_v(P)$.

### 3. HEM Speed of Sound
The speed of sound ($c_{eq}$) in the two-phase region is calculated to include thermal relaxation:
$$\frac{1}{\rho c_{\text{eq}}^2} = \underbrace{\frac{\alpha_l}{\rho_l c_l^2} + \frac{\alpha_v}{\rho_v c_v^2}}_{\text{Mechanical}} + \underbrace{T \left[ \frac{\alpha_l \rho_l}{C_{p,\ell}} \left( \frac{d s_l}{d p} \right)^2 + \frac{\alpha_v \rho_v}{C_{p, v}} \left( \frac{d s_v}{d p} \right)^2 \right]}_{\text{Thermal Relaxation}}$$


## 💻 Usage

1.  **Configure Inlet States**: Set the motive and suction stagnation pressures and temperatures in your master configuration script.
2.  **Generate Tables & CCL**: Run the pipeline to generate the `df_mot` and `df_suc` tables and the corresponding `Physics.ccl` file.
3.  **Solver Execution**:
    * Import the generated CCL into your Ansys CFX-Pre setup.
    * Ensure the domain is set to `Multiphase` with `Homogeneous` velocity coupling.
4.  **Post-Processing**: Run the generated `.cse` script in CFX-Post to extract axial profiles and reconstructed mixture properties.

## ⚠️ Requirements

* Python 3.x
* CoolProp (`HEOS` backend)
* Pandas & NumPy
* Ansys CFX (v19.1 or higher recommended)
