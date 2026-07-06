"""
Physics-Informed Neural Network (PINN) for incompressible Navier-Stokes.

Learns the velocity and pressure fields of 2D flow past a cylinder (Re=100)
from scattered velocity measurements, while enforcing the Navier-Stokes
equations as a soft constraint. The two PDE coefficients (lambda1: convection,
lambda2: viscosity) are treated as trainable parameters and discovered from
data — their true values are 1.0 and 0.01.

Dataset: cylinder_nektar_wake.mat from the original PINN paper
(Raissi, Perdikaris & Karniadakis, 2019), downloaded from
https://github.com/maziarraissi/PINNs

The network maps (x, y, t) -> (psi, p). Velocities are derived from the
stream function (u = dpsi/dy, v = -dpsi/dx) so continuity (div u = 0) is
satisfied exactly by construction.

Outputs:
  results/results.js  — predictions, ground truth, loss history (for the HTML page)
  results/model.pt    — trained weights
"""

import json
import os
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")
torch.manual_seed(1234)
np.random.seed(1234)

N_TRAIN = 5000          # number of scattered training points (x, y, t, u, v)
LAYERS = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
ADAM_ITERS = 6000
LBFGS_ITERS = 500
SNAPSHOT_STRIDE = 20    # export every 20th timestep to the HTML page


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        mods = []
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            mods.append(lin)
        self.linears = nn.ModuleList(mods)
        # unknown PDE coefficients, discovered during training
        self.lambda1 = nn.Parameter(torch.tensor(0.0))
        self.lambda2 = nn.Parameter(torch.tensor(0.0))
        # input normalization bounds, set after seeing data
        self.register_buffer("lb", torch.zeros(3))
        self.register_buffer("ub", torch.ones(3))

    def forward(self, xyt):
        h = 2.0 * (xyt - self.lb) / (self.ub - self.lb) - 1.0
        for lin in self.linears[:-1]:
            h = torch.tanh(lin(h))
        return self.linears[-1](h)  # (psi, p)

    def uvp_residuals(self, x, y, t):
        """Velocities, pressure and Navier-Stokes momentum residuals."""
        xyt = torch.stack([x, y, t], dim=1)
        out = self.forward(xyt)
        psi, p = out[:, 0], out[:, 1]

        grad = lambda f, wrt: torch.autograd.grad(
            f, wrt, grad_outputs=torch.ones_like(f), create_graph=True
        )[0]

        u = grad(psi, y)
        v = -grad(psi, x)

        u_t, u_x, u_y = grad(u, t), grad(u, x), grad(u, y)
        v_t, v_x, v_y = grad(v, t), grad(v, x), grad(v, y)
        u_xx, u_yy = grad(u_x, x), grad(u_y, y)
        v_xx, v_yy = grad(v_x, x), grad(v_y, y)
        p_x, p_y = grad(p, x), grad(p, y)

        l1, l2 = self.lambda1, self.lambda2
        f_u = u_t + l1 * (u * u_x + v * u_y) + p_x - l2 * (u_xx + u_yy)
        f_v = v_t + l1 * (u * v_x + v * v_y) + p_y - l2 * (v_xx + v_yy)
        return u, v, p, f_u, f_v


DATA_URL = ("https://github.com/maziarraissi/PINNs/raw/master/main/Data/"
            "cylinder_nektar_wake.mat")


def load_data():
    path = os.path.join(HERE, "data", "cylinder_nektar_wake.mat")
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, path)
        print(f"Saved to {path}")
    d = sio.loadmat(path)
    X_star = d["X_star"]            # (5000, 2)
    t_star = d["t"]                 # (200, 1)
    U_star = d["U_star"]            # (5000, 2, 200)
    p_star = d["p_star"]            # (5000, 200)

    N, T = X_star.shape[0], t_star.shape[0]
    XX = np.tile(X_star[:, 0:1], (1, T)).ravel()
    YY = np.tile(X_star[:, 1:2], (1, T)).ravel()
    TT = np.tile(t_star.T, (N, 1)).ravel()
    UU = U_star[:, 0, :].ravel()
    VV = U_star[:, 1, :].ravel()
    return d, XX, YY, TT, UU, VV


def main():
    data, XX, YY, TT, UU, VV = load_data()

    idx = np.random.choice(XX.size, N_TRAIN, replace=False)
    to_t = lambda a: torch.tensor(a[idx], dtype=torch.float32, requires_grad=True)
    x, y, t = to_t(XX), to_t(YY), to_t(TT)
    u_obs = torch.tensor(UU[idx], dtype=torch.float32)
    v_obs = torch.tensor(VV[idx], dtype=torch.float32)

    model = PINN(LAYERS).to(DEVICE)
    xyt_all = np.stack([XX, YY, TT], axis=1)
    model.lb.copy_(torch.tensor(xyt_all.min(0), dtype=torch.float32))
    model.ub.copy_(torch.tensor(xyt_all.max(0), dtype=torch.float32))

    loss_history = []

    def compute_loss():
        u, v, _, f_u, f_v = model.uvp_residuals(x, y, t)
        data_loss = ((u - u_obs) ** 2).mean() + ((v - v_obs) ** 2).mean()
        phys_loss = (f_u ** 2).mean() + (f_v ** 2).mean()
        return data_loss + phys_loss, data_loss, phys_loss

    print(f"Training on {N_TRAIN} points | Adam {ADAM_ITERS} iters + L-BFGS {LBFGS_ITERS} iters")
    start = time.time()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)
    for it in range(ADAM_ITERS):
        opt.zero_grad()
        loss, dl, pl = compute_loss()
        loss.backward()
        opt.step()
        sched.step()
        if it % 50 == 0:
            loss_history.append([it, loss.item()])
        if it % 500 == 0:
            print(f"  adam {it:5d} | loss {loss:.4e} (data {dl:.2e}, phys {pl:.2e}) "
                  f"| l1 {model.lambda1.item():.3f} l2 {model.lambda2.item():.4f} "
                  f"| {time.time()-start:.0f}s")

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_ITERS,
                              history_size=50, line_search_fn="strong_wolfe",
                              tolerance_grad=1e-9, tolerance_change=1e-11)
    lbfgs_it = [ADAM_ITERS]

    def closure():
        lbfgs.zero_grad()
        loss, _, _ = compute_loss()
        loss.backward()
        lbfgs_it[0] += 1
        if lbfgs_it[0] % 50 == 0:
            loss_history.append([lbfgs_it[0], loss.item()])
        return loss

    lbfgs.step(closure)
    loss, dl, pl = compute_loss()
    loss_history.append([lbfgs_it[0], loss.item()])
    print(f"  final      | loss {loss:.4e} (data {dl:.2e}, phys {pl:.2e}) "
          f"| l1 {model.lambda1.item():.3f} l2 {model.lambda2.item():.4f} "
          f"| {time.time()-start:.0f}s")

    export(model, data, loss_history)


def export(model, data, loss_history):
    X_star = data["X_star"]
    t_star = data["t"].ravel()
    U_star = data["U_star"]
    p_star = data["p_star"]
    N = X_star.shape[0]

    snap_ids = list(range(0, t_star.size, SNAPSHOT_STRIDE))
    snapshots = []
    errs_u, errs_v, errs_p = [], [], []
    for k in snap_ids:
        xs = torch.tensor(X_star[:, 0], dtype=torch.float32, requires_grad=True)
        ys = torch.tensor(X_star[:, 1], dtype=torch.float32, requires_grad=True)
        ts = torch.full((N,), float(t_star[k]), requires_grad=True)
        u, v, p, _, _ = model.uvp_residuals(xs, ys, ts)
        u, v, p = u.detach().numpy(), v.detach().numpy(), p.detach().numpy()

        ut, vt, pt = U_star[:, 0, k], U_star[:, 1, k], p_star[:, k]
        errs_u.append(np.linalg.norm(u - ut) / np.linalg.norm(ut))
        errs_v.append(np.linalg.norm(v - vt) / np.linalg.norm(vt))
        # pressure is only identifiable up to a constant — align means before comparing
        p_al = p - p.mean() + pt.mean()
        errs_p.append(np.linalg.norm(p_al - pt) / np.linalg.norm(pt))

        r3 = lambda a: [round(float(z), 4) for z in a]
        snapshots.append({
            "t": round(float(t_star[k]), 3),
            "u_pred": r3(u), "v_pred": r3(v), "p_pred": r3(p_al),
            "u_true": r3(ut), "v_true": r3(vt), "p_true": r3(pt),
        })

    print(f"Mean rel. L2 error over {len(snap_ids)} snapshots: "
          f"u {np.mean(errs_u):.3%}, v {np.mean(errs_v):.3%}, p {np.mean(errs_p):.3%}")

    results = {
        "grid": {"nx": 100, "ny": 50,
                 "x": [round(float(v), 4) for v in X_star[:, 0]],
                 "y": [round(float(v), 4) for v in X_star[:, 1]]},
        "snapshots": snapshots,
        "loss_history": loss_history,
        "lambda1": round(model.lambda1.item(), 5),
        "lambda2": round(model.lambda2.item(), 6),
        "errors": {"u": round(float(np.mean(errs_u)), 5),
                   "v": round(float(np.mean(errs_v)), 5),
                   "p": round(float(np.mean(errs_p)), 5)},
        "n_train": N_TRAIN,
        "layers": LAYERS,
    }

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(HERE, "results", "model.pt"))
    with open(os.path.join(HERE, "results", "results.js"), "w") as f:
        f.write("const RESULTS = ")
        json.dump(results, f, separators=(",", ":"))
        f.write(";\n")
    print("Wrote results/results.js and results/model.pt")


if __name__ == "__main__":
    main()
