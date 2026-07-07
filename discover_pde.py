"""
Sparse PDE discovery on top of the trained PINN (SINDy-style).

train_pinn.py assumes the Navier-Stokes momentum equations and fits two
coefficients. This script drops that assumption: each momentum equation is
modeled as

    u_t = - sum_k  xi_k * Theta_k(u, v, p, derivatives)

where Theta is a 14-term library of candidate terms (convection, pressure
gradients, diffusion, plus 9 decoys like u^2, u*v, and a constant). An L1
penalty on the coefficients xi drives the decoys to zero, so the network
*discovers* which terms belong in the physics — recovering the structure of
the Navier-Stokes equations from raw velocity data.

Warm-starts from results/model.pt (run train_pinn.py first).

Outputs:
  results/discovery.js — coefficients + discovered equations (for the HTML page)
"""

import json
import os
import time

import numpy as np
import torch

from train_pinn import HERE, LAYERS, PINN, load_data

torch.manual_seed(1234)
np.random.seed(1234)

N_TRAIN = 5000
ITERS = 2500        # stage 1: L1-regularized fit of the full library
ITERS_REFIT = 1500  # stage 2: debiased refit of surviving terms only
LR_NET = 1e-4       # gentle fine-tuning of the warm-started network
LR_XI = 5e-3        # library coefficients learn faster
L1_WEIGHT = 3e-5    # sparsity pressure on the library (stage 1 only)
THRESH = 5e-3       # |xi| below this is treated as zero when reporting
# Sequential thresholding (STRidge-style): after stage 1, a term survives only
# if its RMS contribution to the time derivative exceeds this fraction of
# RMS(u_t). Effect size, not raw coefficient, so the small viscous terms
# (coef ~0.01 but large u_xx columns) are kept while weak decoys are pruned.
CONTRIB_FRAC = 0.02

U_TERMS = ["u·u_x", "v·u_y", "p_x", "u_xx", "u_yy",
           "u", "v", "p", "u·v", "u²", "v²", "u_y", "v_x", "1"]
V_TERMS = ["u·v_x", "v·v_y", "p_y", "v_xx", "v_yy",
           "u", "v", "p", "u·v", "u²", "v²", "u_x", "v_y", "1"]
# True Navier-Stokes: u_t = -(u u_x + v u_y) - p_x + 0.01 (u_xx + u_yy)
# In the f = u_t + Theta·xi = 0 convention that means:
TRUE_U = {"u·u_x": 1.0, "v·u_y": 1.0, "p_x": 1.0, "u_xx": -0.01, "u_yy": -0.01}
TRUE_V = {"u·v_x": 1.0, "v·v_y": 1.0, "p_y": 1.0, "v_xx": -0.01, "v_yy": -0.01}


def fields(model, x, y, t):
    """All fields and derivatives needed to assemble the term libraries."""
    xyt = torch.stack([x, y, t], dim=1)
    out = model(xyt)
    psi, p = out[:, 0], out[:, 1]
    g = lambda f, w: torch.autograd.grad(
        f, w, grad_outputs=torch.ones_like(f), create_graph=True)[0]

    u, v = g(psi, y), -g(psi, x)
    d = {"u": u, "v": v, "p": p,
         "u_t": g(u, t), "v_t": g(v, t),
         "u_x": g(u, x), "u_y": g(u, y),
         "v_x": g(v, x), "v_y": g(v, y),
         "p_x": g(p, x), "p_y": g(p, y)}
    d["u_xx"], d["u_yy"] = g(d["u_x"], x), g(d["u_y"], y)
    d["v_xx"], d["v_yy"] = g(d["v_x"], x), g(d["v_y"], y)
    return d


def library(d, terms):
    cols = []
    for name in terms:
        if name == "1":
            cols.append(torch.ones_like(d["u"]))
        elif "·" in name:
            a, b = name.split("·")
            cols.append(d[a] * d[b])
        elif name == "u²":
            cols.append(d["u"] ** 2)
        elif name == "v²":
            cols.append(d["v"] ** 2)
        else:
            cols.append(d[name])
    return torch.stack(cols, dim=1)  # (N, n_terms)


def equation_string(lhs, terms, xi):
    parts = []
    for name, c in zip(terms, xi):
        if abs(c) < THRESH:
            continue
        val = -c  # move to the RHS: u_t = -Theta·xi
        parts.append(f"{'+' if val >= 0 and parts else ''}{val:.4g} {name}"
                     if name != "1" else f"{'+' if val >= 0 and parts else ''}{val:.4g}")
    return f"{lhs} = " + " ".join(parts) if parts else f"{lhs} = 0"


def main():
    data, XX, YY, TT, UU, VV = load_data()
    idx = np.random.choice(XX.size, N_TRAIN, replace=False)
    to_t = lambda a: torch.tensor(a[idx], dtype=torch.float32, requires_grad=True)
    x, y, t = to_t(XX), to_t(YY), to_t(TT)
    u_obs = torch.tensor(UU[idx], dtype=torch.float32)
    v_obs = torch.tensor(VV[idx], dtype=torch.float32)

    model = PINN(LAYERS)
    state = torch.load(os.path.join(HERE, "results", "model.pt"), weights_only=True)
    model.load_state_dict(state)
    print("Warm-started from results/model.pt")

    xi_u = torch.nn.Parameter(torch.zeros(len(U_TERMS)))
    xi_v = torch.nn.Parameter(torch.zeros(len(V_TERMS)))

    start = time.time()
    history = []

    def train(iters, l1_weight, mask_u, mask_v, label):
        opt = torch.optim.Adam([
            {"params": [p for n, p in model.named_parameters()
                        if n not in ("lambda1", "lambda2")], "lr": LR_NET},
            {"params": [xi_u, xi_v], "lr": LR_XI},
        ])
        for it in range(iters):
            opt.zero_grad()
            d = fields(model, x, y, t)
            f_u = d["u_t"] + library(d, U_TERMS) @ (xi_u * mask_u)
            f_v = d["v_t"] + library(d, V_TERMS) @ (xi_v * mask_v)
            data_loss = ((d["u"] - u_obs) ** 2).mean() + ((d["v"] - v_obs) ** 2).mean()
            phys_loss = (f_u ** 2).mean() + (f_v ** 2).mean()
            l1 = (xi_u * mask_u).abs().sum() + (xi_v * mask_v).abs().sum()
            loss = data_loss + phys_loss + l1_weight * l1
            loss.backward()
            opt.step()

            if it % 100 == 0:
                history.append([len(history) * 100, loss.item()])
            if it % 500 == 0 or it == iters - 1:
                active_u = sum(1 for c in (xi_u * mask_u).detach() if abs(c) >= THRESH)
                active_v = sum(1 for c in (xi_v * mask_v).detach() if abs(c) >= THRESH)
                print(f"  {label} {it:5d} | loss {loss:.4e} (data {data_loss:.2e}, "
                      f"phys {phys_loss:.2e}) | active terms u:{active_u} v:{active_v} "
                      f"| {time.time()-start:.0f}s")
        return d

    # ----- stage 1: L1-regularized fit of the full library -----
    ones_u = torch.ones_like(xi_u)
    ones_v = torch.ones_like(xi_v)
    d = train(ITERS, L1_WEIGHT, ones_u, ones_v, "stage1")

    # ----- sequential thresholding on effect size -----
    def survivors(d, terms, xi, dt_key):
        theta = library(d, terms).detach()
        rms_dt = d[dt_key].detach().pow(2).mean().sqrt()
        contrib = (theta * xi.detach()).pow(2).mean(0).sqrt()  # RMS per term
        mask = (contrib >= CONTRIB_FRAC * rms_dt).float()
        return mask

    mask_u = survivors(d, U_TERMS, xi_u, "u_t")
    mask_v = survivors(d, V_TERMS, xi_v, "v_t")
    kept_u = [n for n, m in zip(U_TERMS, mask_u) if m]
    kept_v = [n for n, m in zip(V_TERMS, mask_v) if m]
    print(f"\nSequential threshold — surviving terms:")
    print(f"  u-eq: {kept_u}")
    print(f"  v-eq: {kept_v}")

    # ----- stage 2: debiased refit of survivors (no L1) -----
    with torch.no_grad():
        xi_u.mul_(mask_u)
        xi_v.mul_(mask_v)
    train(ITERS_REFIT, 0.0, mask_u, mask_v, "stage2")

    xi_u = (xi_u * mask_u).detach().numpy()
    xi_v = (xi_v * mask_v).detach().numpy()

    eq_u = equation_string("u_t", U_TERMS, xi_u)
    eq_v = equation_string("v_t", V_TERMS, xi_v)
    print("\nDiscovered equations:")
    print(" ", eq_u)
    print(" ", eq_v)

    def pack(terms, xi, true):
        return [{"term": n, "learned": round(float(c), 5),
                 "true": true.get(n, 0.0), "active": bool(abs(c) >= THRESH)}
                for n, c in zip(terms, xi)]

    result = {
        "u": pack(U_TERMS, xi_u, TRUE_U),
        "v": pack(V_TERMS, xi_v, TRUE_V),
        "eq_u": eq_u,
        "eq_v": eq_v,
        "threshold": THRESH,
        "n_library": len(U_TERMS),
        "loss_history": history,
    }
    with open(os.path.join(HERE, "results", "discovery.js"), "w") as f:
        f.write("const DISCOVERY = ")
        json.dump(result, f, separators=(",", ":"))
        f.write(";\n")
    print("Wrote results/discovery.js")


if __name__ == "__main__":
    main()
