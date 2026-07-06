# Physics-Informed Neural Networks for Fluid Flow Simulation

Traditional numerical simulation of fluid flows is computationally intensive and
time-consuming. This project uses a **physics-informed neural network (PINN)** to
reconstruct the velocity and pressure fields of 2D flow past a cylinder (Re = 100),
by embedding the incompressible Navier–Stokes equations directly into the training
loss. Because the network understands the physical constraints during training, it
learns far faster and more data-efficiently than a purely data-driven approach.

Built as part of an Inspirit AI extracurricular project (2023–2025), in
collaboration with UCSB PhD students.

## Highlights

- **Physics as a loss term** — the Navier–Stokes momentum residuals are computed on
  the network output via automatic differentiation and penalized during training.
- **Continuity by construction** — the network predicts a stream function ψ;
  velocities are u = ∂ψ/∂y, v = −∂ψ/∂x, so ∇·u = 0 holds exactly.
- **Inverse problem** — the convection and viscosity coefficients λ₁, λ₂ are unknown
  trainable scalars, discovered from velocity data alone (true values: 1.0, 0.01).
- **Pressure for free** — the model is never shown pressure data; the pressure field
  emerges purely from enforcing momentum balance.

## Data (online source)

High-fidelity spectral DNS of the cylinder wake from the original PINN paper
(Raissi, Perdikaris & Karniadakis, *J. Comput. Phys.*, 2019), downloaded from
[maziarraissi/PINNs](https://github.com/maziarraissi/PINNs):
100×50 spatial grid × 200 time snapshots of (u, v, p). Training uses only 5,000
randomly scattered velocity samples (~0.5% of available points).

## Run it

```bash
pip install torch scipy numpy
python3 train_pinn.py     # trains ~10 min on CPU, writes results/results.js
open index.html           # interactive results page (works from file://)
```

## Files

| File | Purpose |
|---|---|
| `train_pinn.py` | PINN model, training (Adam + L-BFGS), results export |
| `index.html` | Standalone showcase page: animated PINN-vs-DNS fields, loss curve, learned parameters |
| `data/cylinder_nektar_wake.mat` | Cylinder-wake DNS dataset (downloaded) |
| `results/results.js` | Exported predictions + metrics consumed by the page |
| `results/model.pt` | Trained network weights |
