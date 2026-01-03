#!/usr/bin/env python
# coding: utf-8

# # Expériences

# ### Import et device

# In[ ]:


import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pandas as pd
import os
import json
from VAE import VAE
from sampler import pseudo_gibbs_step, metropolis_within_gibbs_step, metropolis_within_gibbs_step_mixture

from utils_plot import savefig, plot_evolution_triplet, show_grid, plot_evolution, show_grid_triplet
from utils_chain import bernoulli_ll_missing, f1_missing
from run_chain import run_chain_with_tracking, run_mwg_multichain, run_chain_with_tracking_mixture
import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Device:", device)

model = VAE().to(device)
model.load_state_dict(torch.load("vae_mnist_paper.pth", map_location=device))
model.eval()
print("VAE loaded.")



def make_random_mask(x, missing_rate: float):
    """
    mask_obs = 1 si observé, 0 si manquanté
    missing_rate = p(missing)
    """
    keep_prob = 1.0 - missing_rate
    return torch.bernoulli(torch.full_like(x, keep_prob))

def init_with_noise(x_true, mask_obs):
    noise = torch.bernoulli(torch.full_like(x_true, 0.5))
    return x_true * mask_obs + noise * (1 - mask_obs)


# ### Dataset et fonctions utiles

# In[ ]:


# MNIST binarisé (comme ton training)
transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float()])
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)

x_true, _ = next(iter(test_loader))
x_true = x_true.to(device)

def make_mask(x, kind="top"):
    m = torch.ones_like(x)
    if kind == "top":
        m[:, :, :14, :] = 0
    elif kind == "bottom":
        m[:, :, 14:, :] = 0
    elif kind == "center":
        m[:, :, 8:20, 8:20] = 0
    elif kind == "random50":
        m = torch.bernoulli(torch.full_like(x, 0.5))
    else:
        raise ValueError("Unknown mask kind")
    return m

def init_with_noise(x_true, mask):
    noise = torch.bernoulli(torch.full_like(x_true, 0.5))
    return x_true * mask + noise * (1 - mask)


import torch.nn.functional as F

import torch.nn.functional as F





# ### Comparaison Pesudo gibbs vs Mgw classique

# In[ ]:


import matplotlib.pyplot as plt
import os
import datetime
import json
import pandas as pd

# ========= CONFIG =========
scenarios = ["top","random50","bottom","center"]

n_iters    = 16000
burn_in    = 2000
thinning   = 20
eval_every = 100
warmup_pg  = 50

# MwG options
use_multichain = False
n_chains = 5

use_adaptive_accept = False
target_accept = 0.23
adapt_lr = 0.005
freeze_adapt_after_burnin = True  # recommandé

idxs_show = [0, 1, 2, 5]

# ========= SAVE DIR =========
run_id = 1
outdir = os.path.join("results", f"exp_recompute_classic_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

# sauvegarde config
config = dict(
    scenarios=scenarios,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every, warmup_pg=warmup_pg,
    use_multichain=use_multichain, n_chains=n_chains,
    use_adaptive_accept=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr,
    freeze_adapt_after_burnin=freeze_adapt_after_burnin
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

all_results = {}
summary_rows = []

for kind in scenarios:
    print("\n" + "="*60)
    print(f"MASK = {kind}")
    print("="*60)

    mask = make_mask(x_true, kind=kind)
    x_init = init_with_noise(x_true, mask)

    # ---- PSEUDO (baseline) ----
    mean_pseudo, summ_p, hist_p = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="pseudo",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every
    )

    # ---- MwG (classic) ----
    if use_multichain:
        mean_mwg, std_mwg, summaries, histories, accs, rhos = run_mwg_multichain(
            model, x_true, x_init, mask,
            n_chains=n_chains,
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=use_adaptive_accept,
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )

        # courbes: chaîne 0 (comme avant)
        hist_m = histories[0]
        summ_m = summaries[0]

        # mixing diag
        std_mwg = std_mwg.to(device)
        miss = (1 - mask).bool()
        mixing_std = std_mwg[miss].mean().item()

        print(f"MwG (multi) acceptance last-window per chain: {[None if a is None else round(a,3) for a in accs]}")
        if use_adaptive_accept:
            print(f"MwG (multi) rho final per chain: {[None if r is None else round(r,3) for r in rhos]}")
        print(f"Inter-chain std on missing pixels (mean): {mixing_std:.4f}")

    else:
        mean_mwg, summ_m, hist_m = run_chain_with_tracking(
            model, x_true, x_init, mask,
            method="mwg",
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=use_adaptive_accept,
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        mixing_std = None

    # ---- scores finaux ----
    mean_pseudo_d = mean_pseudo.to(device)
    mean_mwg_d    = mean_mwg.to(device)

    f1_p = f1_missing(x_true, mean_pseudo_d, mask)
    f1_m = f1_missing(x_true, mean_mwg_d, mask)

    logp_p = summ_p["logp_final"]
    logp_m = summ_m["logp_final"]

    print(f"F1 missing: Pseudo={f1_p:.4f} | MwG={f1_m:.4f}")
    print(f"log p(x_miss_true|x_obs): Pseudo={logp_p:.2f} | MwG={logp_m:.2f}")
    if "acc_final" in summ_m:
        print(f"MwG acceptance (last window): {summ_m['acc_final']:.3f}")
    if use_adaptive_accept and "rho_final" in summ_m:
        print(f"MwG rho_final: {summ_m['rho_final']:.3f}")

    # ---- plot evolution ----
    hist = {
        "steps": hist_p["steps"],
        "f1_pseudo": hist_p["f1"],
        "logp_pseudo": hist_p["logp"],
        "f1_mwg": hist_m["f1"],
        "logp_mwg": hist_m["logp"],
    }
    if "acc" in hist_m:
        hist["acc_mwg"] = hist_m["acc"]
    if use_adaptive_accept and "rho" in hist_m:
        if len(hist_m["rho"]) == len(hist["steps"]):
            hist["rho"] = hist_m["rho"]
        else:
            rho = hist_m["rho"]
            if len(rho) > 0 and len(hist["steps"]) > 0:
                idx = np.linspace(0, len(rho)-1, num=len(hist["steps"])).astype(int)
                hist["rho"] = [rho[i] for i in idx]

    plot_evolution(hist, title=f"Mask={kind} | classic MwG", outdir=outdir, prefix=f"evolution_{kind}")
    plt.close("all")

    # ---- visu grille (CPU pour MPS) ----
    show_grid(
        idxs_show,
        x_true.detach().cpu(),
        x_init.detach().cpu(),
        mean_pseudo.detach().cpu(),
        mean_mwg.detach().cpu(),
        title=f"Mask={kind} | F1 pseudo {f1_p:.3f} vs mwg {f1_m:.3f}",
        outdir=outdir,
        name=f"grid_{kind}.png", labelA="Pseudo Gibbs Mean", labelB="Classic MwG Mean"
    )
    plt.close("all")

    # ---- save histories ----
    pd.DataFrame({"step": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}).to_csv(
        os.path.join(outdir, f"hist_pseudo_{kind}.csv"), index=False
    )
    # hist_m peut venir de multi-chain (chain 0) ou mono-chain
    hist_m_df = {"step": hist_m["steps"], "f1": hist_m["f1"], "logp": hist_m["logp"]}
    if "acc" in hist_m:
        hist_m_df["acc"] = hist_m["acc"]
    if use_adaptive_accept and "rho" in hist_m:
        hist_m_df["rho"] = hist_m["rho"][:len(hist_m["steps"])] if len(hist_m["rho"]) >= len(hist_m["steps"]) else hist_m["rho"]
    pd.DataFrame(hist_m_df).to_csv(
        os.path.join(outdir, f"hist_mwg_{kind}.csv"), index=False
    )

    # ---- summary row ----
    row = dict(
        mask=kind,
        f1_pseudo=float(f1_p), logp_pseudo=float(logp_p),
        f1_mwg=float(f1_m), logp_mwg=float(logp_m),
        mixing_std=mixing_std,
        acc_mwg=float(summ_m["acc_final"]) if "acc_final" in summ_m and summ_m["acc_final"] is not None else None,
        adaptive=use_adaptive_accept,
        multichain=use_multichain
    )
    summary_rows.append(row)
    all_results[kind] = row

# ---- save global summary ----
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(outdir, "summary_all_masks.csv"), index=False)
print("\nSaved:", os.path.join(outdir, "summary_all_masks.csv"))
print(df_sum)

# ---- plots recap ----
plt.figure(figsize=(7,4))
x = np.arange(len(df_sum))
plt.bar(x-0.15, df_sum["f1_pseudo"], width=0.3, label="Pseudo")
plt.bar(x+0.15, df_sum["f1_mwg"], width=0.3, label="MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("F1 missing")
plt.title("F1 by mask | classic MwG")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_f1_all_masks.png")
plt.show()

plt.figure(figsize=(7,4))
plt.bar(x-0.15, df_sum["logp_pseudo"], width=0.3, label="Pseudo")
plt.bar(x+0.15, df_sum["logp_mwg"], width=0.3, label="MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("MC log-likelihood")
plt.title("logp by mask | classic MwG")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_logp_all_masks.png")
plt.show()

print("DONE. OUT:", outdir)


# ### Recherche du meilleur alpha-sigma pour mixture

# In[ ]:


import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
scenarios = ["top", "bottom", "center", "random50"]
n_random_repeats = 5      # répéter seulement random50 (car mask stochastique)
base_seed = 123

# chain params (IMPORTANT: burn_in < n_iters)
n_iters    = 12000
burn_in    = 2000
thinning   = 20
eval_every = 200
warmup_pg  = 50

alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]
sigma_list = [0.05, 0.10, 0.15, 0.25, 0.50]

# filtres de bon sens sur acceptance (optionnel)
acc_min, acc_max = 0.05, 0.50

# =========================================================
# OUTDIR + save config
# =========================================================
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join("results", f"exp_grid_mixture_global_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

def savefig(name, dpi=200):
    path = os.path.join(outdir, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print("saved:", path)

config = dict(
    scenarios=scenarios,
    n_random_repeats=n_random_repeats,
    base_seed=base_seed,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every, warmup_pg=warmup_pg,
    alpha_list=alpha_list, sigma_list=sigma_list,
    acc_min=acc_min, acc_max=acc_max,
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# =========================================================
# Helpers
# =========================================================
def heatmap(df, value_col, title, fname):
    A = sorted(df["alpha"].unique())
    S = sorted(df["rw_sigma"].unique())
    mat = np.full((len(S), len(A)), np.nan)
    for i, s in enumerate(S):
        for j, a in enumerate(A):
            v = df[(df.alpha == a) & (df.rw_sigma == s)][value_col].values
            if len(v):
                mat[i, j] = float(v[0])

    plt.figure(figsize=(7, 5))
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.xticks(range(len(A)), [f"{a:.2f}" for a in A])
    plt.yticks(range(len(S)), [f"{s:.2f}" for s in S])
    plt.xlabel("alpha")
    plt.ylabel("rw_sigma")
    plt.title(title)
    plt.colorbar()
    savefig(fname)
    plt.show()
    plt.close()

def eval_one(alpha, sigma, mask_kind, rep_id=0):
    # si ton make_mask("random50") est stochastic, on fixe la seed pour avoir des repeats contrôlés
    if mask_kind == "random50":
        np.random.seed(base_seed + rep_id)

    mask = make_mask(x_true, kind=mask_kind)
    x_init = init_with_noise(x_true, mask)

    mean_mix, summ_m, hist_m = run_chain_with_tracking_mixture(
        model, x_true, x_init, mask,
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        alpha=alpha,
        rw_sigma=sigma
    )

    # mean_mix est ton posterior mean (CPU) -> F1 sur mean
    mean_mix_d = mean_mix.to(device)
    f1_mean = float(f1_missing(x_true, mean_mix_d, mask))

    return dict(
        mask=mask_kind,
        rep=rep_id,
        alpha=float(alpha),
        rw_sigma=float(sigma),
        f1_mean=f1_mean,
        logp_mc=float(summ_m["logp_final"]),
        acc=float(summ_m.get("acc_final", np.nan)) if summ_m.get("acc_final", None) is not None else np.nan,
        # optionnel: tu peux aussi stocker hist_m si tu veux analyser plus tard
    )

# =========================================================
# RUN GRID
# =========================================================
rows = []
for a in alpha_list:
    for s in sigma_list:
        for kind in scenarios:
            reps = n_random_repeats if kind == "random50" else 1
            for r in range(reps):
                row = eval_one(a, s, kind, rep_id=r)
                rows.append(row)
                print(f"[{kind} rep{r}] a={a:.2f} s={s:.2f} | "
                      f"F1mean={row['f1_mean']:.3f} logp={row['logp_mc']:.1f} acc={row['acc']:.3f}")

df_raw = pd.DataFrame(rows)
df_raw.to_csv(os.path.join(outdir, "grid_raw.csv"), index=False)
print("saved:", os.path.join(outdir, "grid_raw.csv"))

# =========================================================
# AGGREGATE + sélection du best global
# =========================================================
df_agg = (
    df_raw
    .groupby(["alpha", "rw_sigma"], as_index=False)
    .agg(
        f1_mean=("f1_mean", "mean"),
        logp_mc=("logp_mc", "mean"),
        acc=("acc", "mean")
    )
)
df_agg.to_csv(os.path.join(outdir, "grid_agg.csv"), index=False)
print("saved:", os.path.join(outdir, "grid_agg.csv"))

# filtre acceptance (optionnel)
df_sel = df_agg[(df_agg["acc"] >= acc_min) & (df_agg["acc"] <= acc_max)].copy()
if len(df_sel) == 0:
    print("WARNING: aucun point ne passe le filtre acceptance -> je prends tout.")
    df_sel = df_agg.copy()

# règle simple: maximise logp_mc, puis f1_mean
df_sel = df_sel.sort_values(["logp_mc", "f1_mean"], ascending=[False, False])
best = df_sel.iloc[0].to_dict()

with open(os.path.join(outdir, "best.json"), "w") as f:
    json.dump(best, f, indent=2)
print("BEST:", best)

# =========================================================
# HEATMAPS (global + par mask)
# =========================================================
heatmap(df_agg, "f1_mean", "GLOBAL mean F1_mean", "heatmap_global_f1_mean.png")
heatmap(df_agg, "logp_mc", "GLOBAL mean logp_mc", "heatmap_global_logp_mc.png")
heatmap(df_agg, "acc",     "GLOBAL mean acceptance", "heatmap_global_acc.png")

for kind in scenarios:
    d = (
        df_raw[df_raw["mask"] == kind]
        .groupby(["alpha", "rw_sigma"], as_index=False)
        .agg(f1_mean=("f1_mean", "mean"),
             logp_mc=("logp_mc", "mean"),
             acc=("acc", "mean"))
    )
    heatmap(d, "f1_mean", f"{kind} mean F1_mean", f"heatmap_{kind}_f1_mean.png")
    heatmap(d, "logp_mc", f"{kind} mean logp_mc", f"heatmap_{kind}_logp_mc.png")
    heatmap(d, "acc",     f"{kind} mean acceptance", f"heatmap_{kind}_acc.png")

print("DONE. OUT:", outdir)


# ### Comparaison Peusdo vs Mgw classique vs Mgw Mixture

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# CONFIG: classic vs mixture
# =========================
best_alpha = 0
best_sigma = 0.15   # (ou 0.10 si tu veux favoriser logp/acc)

scenarios = ["top","random50","bottom","center"]

n_iters    = 16000
burn_in    = 2000
thinning   = 20
eval_every = 100
warmup_pg  = 50

# classic MwG options
use_multichain = False
n_chains = 5

use_adaptive_accept = False
target_accept = 0.23
adapt_lr = 0.005
freeze_adapt_after_burnin = True

idxs_show = [0, 1, 2, 5]

# ========= SAVE DIR =========
run_id = 2
outdir = os.path.join("results", f"exp_compare_classic_vs_mix_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

# save config
config = dict(
    best_alpha=best_alpha, best_sigma=best_sigma,
    scenarios=scenarios,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every, warmup_pg=warmup_pg,
    use_multichain=use_multichain, n_chains=n_chains,
    use_adaptive_accept=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr,
    freeze_adapt_after_burnin=freeze_adapt_after_burnin
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

summary_rows = []

for kind in scenarios:
    print("\n" + "="*60)
    print(f"MASK = {kind}")
    print("="*60)

    mask = make_mask(x_true, kind=kind)
    x_init = init_with_noise(x_true, mask)

    # -----------------
    # PSEUDO (baseline)
    # -----------------
    mean_pseudo, summ_p, hist_p = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="pseudo",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every
    )

    # -----------------
    # CLASSIC MwG
    # -----------------
    if use_multichain:
        mean_classic, std_mwg, summaries, histories, accs, rhos = run_mwg_multichain(
            model, x_true, x_init, mask,
            n_chains=n_chains,
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=use_adaptive_accept,
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        hist_c = histories[0]
        summ_c = summaries[0]
        std_mwg = std_mwg.to(device)
        miss = (1 - mask).bool()
        mixing_std = std_mwg[miss].mean().item()
    else:
        mean_classic, summ_c, hist_c = run_chain_with_tracking(
            model, x_true, x_init, mask,
            method="mwg",
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=use_adaptive_accept,
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        mixing_std = None

    # -----------------
    # MIXTURE MwG (best)
    # -----------------
    mean_mix, summ_m, hist_m = run_chain_with_tracking_mixture(
        model, x_true, x_init, mask,
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        alpha=best_alpha,
        rw_sigma=best_sigma
    )

    # -----------------
    # Final scores
    # -----------------
    mean_pseudo_d  = mean_pseudo.to(device)
    mean_classic_d = mean_classic.to(device)
    mean_mix_d     = mean_mix.to(device)

    f1_p = f1_missing(x_true, mean_pseudo_d, mask)
    f1_c = f1_missing(x_true, mean_classic_d, mask)
    f1_m = f1_missing(x_true, mean_mix_d, mask)

    logp_p = summ_p["logp_final"]
    logp_c = summ_c["logp_final"]
    logp_m = summ_m["logp_final"]

    acc_c = summ_c.get("acc_final", None)
    acc_m = summ_m.get("acc_final", None)

    print(f"F1 missing:  Pseudo={f1_p:.4f} | Classic={f1_c:.4f} | Mixture={f1_m:.4f}")
    print(f"log p(...):  Pseudo={logp_p:.2f} | Classic={logp_c:.2f} | Mixture={logp_m:.2f}")
    if acc_c is not None:
        print(f"Acceptance:  Classic={acc_c:.3f} | Mixture={acc_m:.3f}")
    if mixing_std is not None:
        print(f"Mixing std (multi-chain): {mixing_std:.4f}")

    # -----------------
    # ONE SINGLE EVOLUTION PLOT (Pseudo vs Classic vs Mixture) + save
    # -----------------
    histP = {"steps": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}
    histC = {"steps": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c:
        histC["acc"] = hist_c["acc"]
    histM = {"steps": hist_m["steps"], "f1": hist_m["f1"], "logp": hist_m["logp"]}
    if "acc" in hist_m:
        histM["acc"] = hist_m["acc"]


    plot_evolution_triplet(histP, histC, histM,
                       title=f"Mask={kind}",
                       outdir=outdir,
                       prefix=f"evolution_triplet_{kind}")
    # -----------------
    # Image grids + save (CPU for MPS)
    # -----------------
    show_grid_triplet(
        idxs_show,
        x_true.detach().cpu(),
        x_init.detach().cpu(),
        mean_pseudo.detach().cpu(),
        mean_classic.detach().cpu(),
        mean_mix.detach().cpu(),
        title=f"Mask={kind}",
        outdir=outdir,
        name=f"grid_triplet_{kind}.png",
        labels=("Pseudo Gibbs", "Classic MwG", f"Mixture"),
    )
    plt.close("all")

    # -----------------
    # Save histories
    # -----------------
    pd.DataFrame({"step": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}).to_csv(
        os.path.join(outdir, f"hist_pseudo_{kind}.csv"), index=False
    )
    pd.DataFrame({"step": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"], **({"acc": hist_c["acc"]} if "acc" in hist_c else {})}).to_csv(
        os.path.join(outdir, f"hist_classic_{kind}.csv"), index=False
    )
    pd.DataFrame({"step": hist_m["steps"], "f1": hist_m["f1"], "logp": hist_m["logp"], **({"acc": hist_m["acc"]} if "acc" in hist_m else {})}).to_csv(
        os.path.join(outdir, f"hist_mix_{kind}.csv"), index=False
    )

    # -----------------
    # Summary row
    # -----------------
    summary_rows.append(dict(
        mask=kind,
        f1_pseudo=float(f1_p), logp_pseudo=float(logp_p),
        f1_classic=float(f1_c), logp_classic=float(logp_c), acc_classic=float(acc_c) if acc_c is not None else None,
        f1_mix=float(f1_m), logp_mix=float(logp_m), acc_mix=float(acc_m) if acc_m is not None else None,
        gain_f1_mix_vs_classic=float(f1_m - f1_c),
        gain_logp_mix_vs_classic=float(logp_m - logp_c),
        alpha=best_alpha, rw_sigma=best_sigma,
        mixing_std=mixing_std,
        multichain=use_multichain,
        adaptive=use_adaptive_accept
    ))

# =========================
# Save global summary + recap plots
# =========================
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(outdir, "summary_all_masks.csv"), index=False)
print("\nSaved:", os.path.join(outdir, "summary_all_masks.csv"))
print(df_sum)

# F1 recap
plt.figure(figsize=(9,4))
x = np.arange(len(df_sum))
plt.bar(x-0.25, df_sum["f1_pseudo"], width=0.25, label="Pseudo")
plt.bar(x,      df_sum["f1_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["f1_mix"], width=0.25, label="Mixture MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("F1 missing")
plt.title(f"F1 by mask | mixture a={best_alpha}, sigma={best_sigma}")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_f1_pseudo_classic_mix.png")
plt.show()

# logp recap
plt.figure(figsize=(9,4))
plt.bar(x-0.25, df_sum["logp_pseudo"], width=0.25, label="Pseudo")
plt.bar(x,      df_sum["logp_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["logp_mix"], width=0.25, label="Mixture MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("MC log-likelihood")
plt.title(f"logp by mask | mixture a={best_alpha}, sigma={best_sigma}")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_logp_pseudo_classic_mix.png")
plt.show()

# gain plots
plt.figure(figsize=(9,4))
plt.bar(x-0.15, df_sum["gain_f1_mix_vs_classic"], width=0.3)
plt.xticks(x, df_sum["mask"])
plt.ylabel("ΔF1 (Mixture - Classic)")
plt.title("Gain F1: mixture vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_f1_mix_vs_classic.png")
plt.show()

plt.figure(figsize=(9,4))
plt.bar(x-0.15, df_sum["gain_logp_mix_vs_classic"], width=0.3)
plt.xticks(x, df_sum["mask"])
plt.ylabel("Δlogp (Mixture - Classic)")
plt.title("Gain logp: mixture vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_logp_mix_vs_classic.png")
plt.show()

print("DONE. OUT:", outdir)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# CONFIG: classic vs mixture
# =========================
best_alpha = 0.25
best_sigma = 0.15   # (ou 0.10 si tu veux favoriser logp/acc)

scenarios = ["top","random50","bottom","center"]

n_iters    = 16000
burn_in    = 2000
thinning   = 20
eval_every = 100
warmup_pg  = 50

# classic MwG options
use_multichain = False
n_chains = 5

use_adaptive_accept = False
target_accept = 0.23
adapt_lr = 0.005
freeze_adapt_after_burnin = True

idxs_show = [0, 1, 2, 5]

# ========= SAVE DIR =========
run_id = 10
outdir = os.path.join("results", f"exp_compare_classic_vs_mix_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

# save config
config = dict(
    best_alpha=best_alpha, best_sigma=best_sigma,
    scenarios=scenarios,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every, warmup_pg=warmup_pg,
    use_multichain=use_multichain, n_chains=n_chains,
    use_adaptive_accept=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr,
    freeze_adapt_after_burnin=freeze_adapt_after_burnin
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

summary_rows = []

for kind in scenarios:
    print("\n" + "="*60)
    print(f"MASK = {kind}")
    print("="*60)

    mask = make_mask(x_true, kind=kind)
    x_init = init_with_noise(x_true, mask)

    # -----------------
    # PSEUDO (baseline)
    # -----------------
    mean_pseudo, summ_p, hist_p = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="pseudo",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every
    )

    # -----------------
    # CLASSIC MwG
    # -----------------
    if use_multichain:
        mean_classic, std_mwg, summaries, histories, accs, rhos = run_mwg_multichain(
            model, x_true, x_init, mask,
            n_chains=n_chains,
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=use_adaptive_accept,
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        hist_c = histories[0]
        summ_c = summaries[0]
        std_mwg = std_mwg.to(device)
        miss = (1 - mask).bool()
        mixing_std = std_mwg[miss].mean().item()
    else:
        mean_classic, summ_c, hist_c = run_chain_with_tracking(
            model, x_true, x_init, mask,
            method="mwg",
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=use_adaptive_accept,
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        mixing_std = None

    # -----------------
    # MIXTURE MwG (best)
    # -----------------
    mean_mix, summ_m, hist_m = run_chain_with_tracking_mixture(
        model, x_true, x_init, mask,
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        alpha=best_alpha,
        rw_sigma=best_sigma
    )

    # -----------------
    # Final scores
    # -----------------
    mean_pseudo_d  = mean_pseudo.to(device)
    mean_classic_d = mean_classic.to(device)
    mean_mix_d     = mean_mix.to(device)

    f1_p = f1_missing(x_true, mean_pseudo_d, mask)
    f1_c = f1_missing(x_true, mean_classic_d, mask)
    f1_m = f1_missing(x_true, mean_mix_d, mask)

    logp_p = summ_p["logp_final"]
    logp_c = summ_c["logp_final"]
    logp_m = summ_m["logp_final"]

    acc_c = summ_c.get("acc_final", None)
    acc_m = summ_m.get("acc_final", None)

    print(f"F1 missing:  Pseudo={f1_p:.4f} | Classic={f1_c:.4f} | Mixture={f1_m:.4f}")
    print(f"log p(...):  Pseudo={logp_p:.2f} | Classic={logp_c:.2f} | Mixture={logp_m:.2f}")
    if acc_c is not None:
        print(f"Acceptance:  Classic={acc_c:.3f} | Mixture={acc_m:.3f}")
    if mixing_std is not None:
        print(f"Mixing std (multi-chain): {mixing_std:.4f}")

    # -----------------
    # ONE SINGLE EVOLUTION PLOT (Pseudo vs Classic vs Mixture) + save
    # -----------------
    histP = {"steps": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}
    histC = {"steps": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c:
        histC["acc"] = hist_c["acc"]
    histM = {"steps": hist_m["steps"], "f1": hist_m["f1"], "logp": hist_m["logp"]}
    if "acc" in hist_m:
        histM["acc"] = hist_m["acc"]


    plot_evolution_triplet(histP, histC, histM,
                       title=f"Mask={kind}",
                       outdir=outdir,
                       prefix=f"evolution_triplet_{kind}")
    # -----------------
    # Image grids + save (CPU for MPS)
    # -----------------
    show_grid_triplet(
        idxs_show,
        x_true.detach().cpu(),
        x_init.detach().cpu(),
        mean_pseudo.detach().cpu(),
        mean_classic.detach().cpu(),
        mean_mix.detach().cpu(),
        title=f"Mask={kind}",
        outdir=outdir,
        name=f"grid_triplet_{kind}.png",
        labels=("Pseudo Gibbs", "Classic MwG", f"Mixture"),
    )
    plt.close("all")

    # -----------------
    # Save histories
    # -----------------
    pd.DataFrame({"step": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}).to_csv(
        os.path.join(outdir, f"hist_pseudo_{kind}.csv"), index=False
    )
    pd.DataFrame({"step": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"], **({"acc": hist_c["acc"]} if "acc" in hist_c else {})}).to_csv(
        os.path.join(outdir, f"hist_classic_{kind}.csv"), index=False
    )
    pd.DataFrame({"step": hist_m["steps"], "f1": hist_m["f1"], "logp": hist_m["logp"], **({"acc": hist_m["acc"]} if "acc" in hist_m else {})}).to_csv(
        os.path.join(outdir, f"hist_mix_{kind}.csv"), index=False
    )

    # -----------------
    # Summary row
    # -----------------
    summary_rows.append(dict(
        mask=kind,
        f1_pseudo=float(f1_p), logp_pseudo=float(logp_p),
        f1_classic=float(f1_c), logp_classic=float(logp_c), acc_classic=float(acc_c) if acc_c is not None else None,
        f1_mix=float(f1_m), logp_mix=float(logp_m), acc_mix=float(acc_m) if acc_m is not None else None,
        gain_f1_mix_vs_classic=float(f1_m - f1_c),
        gain_logp_mix_vs_classic=float(logp_m - logp_c),
        alpha=best_alpha, rw_sigma=best_sigma,
        mixing_std=mixing_std,
        multichain=use_multichain,
        adaptive=use_adaptive_accept
    ))

# =========================
# Save global summary + recap plots
# =========================
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(outdir, "summary_all_masks.csv"), index=False)
print("\nSaved:", os.path.join(outdir, "summary_all_masks.csv"))
print(df_sum)

# F1 recap
plt.figure(figsize=(9,4))
x = np.arange(len(df_sum))
plt.bar(x-0.25, df_sum["f1_pseudo"], width=0.25, label="Pseudo")
plt.bar(x,      df_sum["f1_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["f1_mix"], width=0.25, label="Mixture MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("F1 missing")
plt.title(f"F1 by mask | mixture a={best_alpha}, sigma={best_sigma}")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_f1_pseudo_classic_mix.png")
plt.show()

# logp recap
plt.figure(figsize=(9,4))
plt.bar(x-0.25, df_sum["logp_pseudo"], width=0.25, label="Pseudo")
plt.bar(x,      df_sum["logp_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["logp_mix"], width=0.25, label="Mixture MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("MC log-likelihood")
plt.title(f"logp by mask | mixture a={best_alpha}, sigma={best_sigma}")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_logp_pseudo_classic_mix.png")
plt.show()

# gain plots
plt.figure(figsize=(9,4))
plt.bar(x-0.15, df_sum["gain_f1_mix_vs_classic"], width=0.3)
plt.xticks(x, df_sum["mask"])
plt.ylabel("ΔF1 (Mixture - Classic)")
plt.title("Gain F1: mixture vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_f1_mix_vs_classic.png")
plt.show()

plt.figure(figsize=(9,4))
plt.bar(x-0.15, df_sum["gain_logp_mix_vs_classic"], width=0.3)
plt.xticks(x, df_sum["mask"])
plt.ylabel("Δlogp (Mixture - Classic)")
plt.title("Gain logp: mixture vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_logp_mix_vs_classic.png")
plt.show()

print("DONE. OUT:", outdir)


# In[ ]:


# ============================================================
# EXP 7 — Posterior samples diversity / mode-switching diagnostics
#   Pseudo vs Classic MwG vs Mixture MwG
#   - save N posterior samples
#   - visualize multiple samples (not just posterior mean)
#   - quantify diversity on missing region + logp distribution
# ============================================================

import os, json, datetime, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------
# helpers: robust log Bernoulli on OBS pixels (for MH) + MISSING pixels (for eval)
# -------------------------
def bernoulli_logprob_pixels(x, probs, eps=1e-6):
    """
    returns per-pixel log p(x|probs) (B,784) for Bernoulli, robust with clamp
    """
    x = x.view(x.size(0), -1)
    p = probs.view(probs.size(0), -1).clamp(eps, 1 - eps)
    # log Bernoulli = x log p + (1-x) log(1-p)
    return x * torch.log(p) + (1 - x) * torch.log(1 - p)

def log_p_xobs_given_z(x_curr, probs, mask_obs):
    """
    log p(x_obs | z) where mask_obs=1 on observed pixels.
    Uses current x_curr for observed pixels.
    returns (B,)
    """
    ll_pix = bernoulli_logprob_pixels(x_curr, probs)  # (B,784)
    m = mask_obs.view(mask_obs.size(0), -1)
    return (ll_pix * m).sum(dim=1)

def log_p_xmiss_true_given_probs(x_true, probs, mask_obs):
    """
    evaluation metric: log p(x_miss_true | probs) on missing pixels only
    returns (B,)
    """
    ll_pix = bernoulli_logprob_pixels(x_true, probs)  # (B,784)
    miss = (1 - mask_obs).view(mask_obs.size(0), -1)
    return (ll_pix * miss).sum(dim=1)

def log_standard_normal(z):
    # log N(z;0,I) summed over dims -> (B,)
    return (-0.5 * (z**2 + math.log(2*math.pi))).sum(dim=1)

def hamming_missing(x_a, x_b, mask_obs):
    """
    x_a, x_b: (B,1,28,28) binary {0,1}
    returns average Hamming distance on missing pixels (scalar)
    """
    miss = (1 - mask_obs).bool()
    diff = (x_a != x_b) & miss
    return diff.float().mean().item()

# -------------------------
# core samplers for EXP7
# -------------------------
@torch.no_grad()
def collect_samples_pseudo(model, x_true, x_init, mask_obs, *, n_iters=16000, burn_in=2000, sample_every=200):
    x = x_init.clone()
    samples_probs = []
    samples_bin = []
    logp_samples = []

    for t in range(n_iters):
        x_out, _ = pseudo_gibbs_step(model, x, mask_obs)   # probs
        x = torch.bernoulli(x_out)

        if t >= burn_in and ((t - burn_in) % sample_every == 0):
            probs = x_out
            samples_probs.append(probs.detach().cpu())
            samples_bin.append(torch.bernoulli(probs).detach().cpu())
            lp = log_p_xmiss_true_given_probs(x_true, probs, mask_obs).mean().item()
            logp_samples.append(lp)

    return {
        "samples_probs": torch.stack(samples_probs, dim=0),  # (S,B,1,28,28)
        "samples_bin":   torch.stack(samples_bin, dim=0),
        "logp_samples":  np.array(logp_samples),
        "acc_samples":   None,
    }

@torch.no_grad()
def collect_samples_classic(model, x_true, x_init, mask_obs, *,
                            n_iters=16000, burn_in=2000, sample_every=200, warmup_pg=50,
                            adaptive_accept=False, target_accept=0.23, adapt_lr=0.005, freeze_adapt_after_burnin=True):
    # init x
    x = x_init.clone()

    # warmup pseudo-gibbs for better init
    for _ in range(warmup_pg):
        x_out, _ = pseudo_gibbs_step(model, x, mask_obs)
        x = torch.bernoulli(x_out)

    mu, logvar = model.encode(x.view(-1, 784))
    z = model.reparameterize(mu, logvar)

    samples_probs = []
    samples_bin = []
    logp_samples = []
    acc_samples = []

    # optional adaptive scaling (if your metropolis_within_gibbs_step supports it)
    log_rho = torch.tensor(0.0, device=x.device)

    for t in range(n_iters):
        if adaptive_accept and (not freeze_adapt_after_burnin or t < burn_in):
            x_out, z, acc, log_rho = metropolis_within_gibbs_step(
                model, x, z, mask_obs,
                return_accept=True,
                adaptive=True,
                log_rho=log_rho,
                target_accept=target_accept,
                adapt_lr=adapt_lr
            )
        else:
            rho = float(torch.exp(log_rho).item()) if adaptive_accept else 1.0
            x_out, z, acc = metropolis_within_gibbs_step(
                model, x, z, mask_obs,
                return_accept=True,
                adaptive=False,
                proposal_scale=rho
            )

        x = torch.bernoulli(x_out)

        if t >= burn_in and ((t - burn_in) % sample_every == 0):
            probs = x_out
            samples_probs.append(probs.detach().cpu())
            samples_bin.append(torch.bernoulli(probs).detach().cpu())
            lp = log_p_xmiss_true_given_probs(x_true, probs, mask_obs).mean().item()
            logp_samples.append(lp)
            acc_samples.append(float(acc))

    return {
        "samples_probs": torch.stack(samples_probs, dim=0),
        "samples_bin":   torch.stack(samples_bin, dim=0),
        "logp_samples":  np.array(logp_samples),
        "acc_samples":   np.array(acc_samples),
    }

@torch.no_grad()
def collect_samples_mixture(model, x_true, x_init, mask_obs, *,
                            alpha=0.5, rw_sigma=0.25,
                            n_iters=16000, burn_in=2000, sample_every=200, warmup_pg=50):
    """
    Mixture proposal on z:
      with prob alpha: independence proposal z ~ q(z|x_curr)  (same as classic MwG)
      with prob 1-alpha: random-walk z' = z + rw_sigma * eps   (symmetric)
    Always MH-correct wrt p(z) p(x_obs|z).
    """
    x = x_init.clone()

    # warmup pseudo-gibbs
    for _ in range(warmup_pg):
        x_out, _ = pseudo_gibbs_step(model, x, mask_obs)
        x = torch.bernoulli(x_out)

    mu, logvar = model.encode(x.view(-1, 784))
    z = model.reparameterize(mu, logvar)

    samples_probs = []
    samples_bin = []
    logp_samples = []
    acc_samples = []

    for t in range(n_iters):
        # ----- propose z -----
        if torch.rand(()) < alpha:
            # independence from q(z|x_curr)
            mu_prop, logvar_prop = model.encode(x.view(-1, 784))
            z_prop = model.reparameterize(mu_prop, logvar_prop)

            # compute MH ratio: p(x_obs|z)p(z) / q(z|x)  (classic independence MH)
            x_rec_prop = model.decode(z_prop)
            x_rec_cur  = model.decode(z)

            log_num = log_p_xobs_given_z(x, x_rec_prop, mask_obs) + log_standard_normal(z_prop)
            log_den = log_p_xobs_given_z(x, x_rec_cur,  mask_obs) + log_standard_normal(z)

            log_q_prop = (-0.5 * (math.log(2*math.pi) + logvar_prop + (z_prop - mu_prop)**2 / torch.exp(logvar_prop))).sum(dim=1)

            mu_cur, logvar_cur = model.encode(x.view(-1, 784))
            log_q_cur = (-0.5 * (math.log(2*math.pi) + logvar_cur + (z - mu_cur)**2 / torch.exp(logvar_cur))).sum(dim=1)

            log_alpha = (log_num - log_q_prop) - (log_den - log_q_cur)

        else:
            # random-walk symmetric
            z_prop = z + rw_sigma * torch.randn_like(z)

            x_rec_prop = model.decode(z_prop)
            x_rec_cur  = model.decode(z)

            log_num = log_p_xobs_given_z(x, x_rec_prop, mask_obs) + log_standard_normal(z_prop)
            log_den = log_p_xobs_given_z(x, x_rec_cur,  mask_obs) + log_standard_normal(z)
            log_alpha = log_num - log_den

        # ----- accept/reject -----
        alpha_acc = torch.exp(torch.clamp(log_alpha, max=0.0))
        u = torch.rand_like(alpha_acc)
        accept = (u < alpha_acc).float().unsqueeze(1)
        z = z_prop * accept + z * (1 - accept)
        acc = accept.mean().item()

        # ----- sample missing pixels from p(x|z) and clamp observed -----
        x_rec = model.decode(z).view_as(x)
        x_next = x * mask_obs + x_rec * (1 - mask_obs)
        x = torch.bernoulli(x_next)

        if t >= burn_in and ((t - burn_in) % sample_every == 0):
            probs = x_next
            samples_probs.append(probs.detach().cpu())
            samples_bin.append(torch.bernoulli(probs).detach().cpu())
            lp = log_p_xmiss_true_given_probs(x_true, probs, mask_obs).mean().item()
            logp_samples.append(lp)
            acc_samples.append(float(acc))

    return {
        "samples_probs": torch.stack(samples_probs, dim=0),
        "samples_bin":   torch.stack(samples_bin, dim=0),
        "logp_samples":  np.array(logp_samples),
        "acc_samples":   np.array(acc_samples),
    }

# -------------------------
# visuals
# -------------------------
def plot_sample_grid(x_true, x_init, samples_probs, *, idx=0, n_show=8, title=""):
    """
    Show: True | Init | S samples from samples_probs for ONE image idx (binarized for readability)
    samples_probs: (S,B,1,28,28) on CPU
    """
    S = samples_probs.size(0)
    n_show = min(n_show, S)
    cols = 2 + n_show
    fig, axes = plt.subplots(1, cols, figsize=(2.0*cols, 2.4))

    axes[0].imshow(x_true[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("True", fontweight="bold"); axes[0].axis("off")

    axes[1].imshow(x_init[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Init", fontweight="bold"); axes[1].axis("off")

    for j in range(n_show):
        img = (samples_probs[j, idx] > 0.5).float()
        axes[2+j].imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[2+j].set_title(f"S{j}", fontsize=10)
        axes[2+j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def summarize_diversity(samples_probs, samples_bin, mask_obs, x_true):
    """
    Returns dict of scalar diagnostics:
      - mean_var_missing: mean per-pixel variance on missing region (over samples)
      - mean_hamming_step: mean Hamming between consecutive binary samples (missing region)
      - logp_mean/std: mean/std of logp_samples (computed outside)
    """
    # per-pixel variance over samples (S,B,1,28,28) -> (B,1,28,28)
    var_pix = samples_probs.float().var(dim=0)  # variance across samples
    miss = (1 - mask_obs).bool().cpu()
    mean_var_missing = var_pix[miss].mean().item()

    # consecutive Hamming distance on missing region
    S = samples_bin.size(0)
    if S >= 2:
        ham = []
        for s in range(1, S):
            ham.append(hamming_missing(samples_bin[s-1], samples_bin[s], mask_obs.cpu()))
        mean_hamming_step = float(np.mean(ham))
    else:
        mean_hamming_step = float("nan")

    return dict(
        mean_var_missing=mean_var_missing,
        mean_hamming_step=mean_hamming_step
    )

# ============================================================
# RUN EXP 7
# ============================================================
# config
scenarios = ["top","bottom","center","random50"]
n_iters = 16000
burn_in = 2000
sample_every = 200   # -> ~70 samples after burn-in
warmup_pg = 50

# classic options
use_adaptive_accept = False
target_accept = 0.23
adapt_lr = 0.005
freeze_adapt_after_burnin = True

# mixture best
best_alpha = 0.5
best_sigma = 0.15

# which images to visualize
viz_idxs = [0, 1, 2]  # a few digits
n_show_samples = 8

# save dir
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join("results", f"exp7_samples_diversity_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(dict(
        scenarios=scenarios,
        n_iters=n_iters, burn_in=burn_in, sample_every=sample_every, warmup_pg=warmup_pg,
        classic=dict(adaptive=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr, freeze=freeze_adapt_after_burnin),
        mixture=dict(alpha=best_alpha, rw_sigma=best_sigma),
        viz_idxs=viz_idxs, n_show_samples=n_show_samples
    ), f, indent=2)

rows = []

# ensure CPU versions for plotting grids (MPS friendly)
x_true_cpu = x_true.detach().cpu()

for kind in scenarios:
    print("\n" + "="*70)
    print(f"MASK = {kind}")
    print("="*70)

    mask = make_mask(x_true, kind=kind)
    x_init = init_with_noise(x_true, mask)

    mask_cpu = mask.detach().cpu()
    x_init_cpu = x_init.detach().cpu()

    # ---- collect samples ----
    res_p = collect_samples_pseudo(model, x_true, x_init, mask, n_iters=n_iters, burn_in=burn_in, sample_every=sample_every)
    res_c = collect_samples_classic(
        model, x_true, x_init, mask,
        n_iters=n_iters, burn_in=burn_in, sample_every=sample_every, warmup_pg=warmup_pg,
        adaptive_accept=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr, freeze_adapt_after_burnin=freeze_adapt_after_burnin
    )
    res_m = collect_samples_mixture(
        model, x_true, x_init, mask,
        alpha=best_alpha, rw_sigma=best_sigma,
        n_iters=n_iters, burn_in=burn_in, sample_every=sample_every, warmup_pg=warmup_pg
    )

    # ---- diagnostics ----
    div_p = summarize_diversity(res_p["samples_probs"], res_p["samples_bin"], mask, x_true)
    div_c = summarize_diversity(res_c["samples_probs"], res_c["samples_bin"], mask, x_true)
    div_m = summarize_diversity(res_m["samples_probs"], res_m["samples_bin"], mask, x_true)

    logp_p = res_p["logp_samples"]; logp_c = res_c["logp_samples"]; logp_m = res_m["logp_samples"]
    acc_c = res_c["acc_samples"];   acc_m = res_m["acc_samples"]

    print(f"[Pseudo]   logp mean={logp_p.mean():.2f} std={logp_p.std():.2f} | var_missing={div_p['mean_var_missing']:.4f} | ham_step={div_p['mean_hamming_step']:.4f}")
    print(f"[Classic]  logp mean={logp_c.mean():.2f} std={logp_c.std():.2f} | var_missing={div_c['mean_var_missing']:.4f} | ham_step={div_c['mean_hamming_step']:.4f} | acc={np.nanmean(acc_c):.3f}")
    print(f"[Mixture]  logp mean={logp_m.mean():.2f} std={logp_m.std():.2f} | var_missing={div_m['mean_var_missing']:.4f} | ham_step={div_m['mean_hamming_step']:.4f} | acc={np.nanmean(acc_m):.3f}")

    # ---- save raw samples (optional but useful) ----
    # (warning: can be big; ok for ~70 samples * 64 images)
    torch.save(res_p["samples_probs"], os.path.join(outdir, f"samples_probs_pseudo_{kind}.pt"))
    torch.save(res_c["samples_probs"], os.path.join(outdir, f"samples_probs_classic_{kind}.pt"))
    torch.save(res_m["samples_probs"], os.path.join(outdir, f"samples_probs_mix_{kind}.pt"))

    # ---- save per-sample logp / acc ----
    pd.DataFrame({
        "sample_id": np.arange(len(logp_p)),
        "logp_pseudo": logp_p,
        "logp_classic": logp_c,
        "logp_mix": logp_m,
        "acc_classic": acc_c if acc_c is not None else np.nan,
        "acc_mix": acc_m if acc_m is not None else np.nan
    }).to_csv(os.path.join(outdir, f"logp_acc_samples_{kind}.csv"), index=False)

    # ---- plot logp distributions ----
    plt.figure(figsize=(8,4))
    plt.hist(logp_p, bins=20, alpha=0.6, label="Pseudo")
    plt.hist(logp_c, bins=20, alpha=0.6, label="Classic")
    plt.hist(logp_m, bins=20, alpha=0.6, label="Mixture")
    plt.title(f"Mask={kind} — distribution of log p(x_miss_true | x_obs) over posterior samples")
    plt.xlabel("logp per sample (batch mean)")
    plt.ylabel("count")
    plt.grid(True, alpha=0.2)
    plt.legend()
    savefig(outdir, f"logp_hist_{kind}.png")
    plt.show()
    plt.close("all")

    # ---- visualize sample grids for a few images ----
    for idx in viz_idxs:
        # Pseudo
        plot_sample_grid(x_true_cpu, x_init_cpu, res_p["samples_probs"], idx=idx, n_show=n_show_samples,
                         title=f"Mask={kind} | Pseudo | idx={idx}")
        savefig(outdir, f"samples_grid_{kind}_idx{idx}_pseudo.png")
        plt.close("all")

        # Classic
        plot_sample_grid(x_true_cpu, x_init_cpu, res_c["samples_probs"], idx=idx, n_show=n_show_samples,
                         title=f"Mask={kind} | Classic MwG | idx={idx}")
        savefig(outdir, f"samples_grid_{kind}_idx{idx}_classic.png")
        plt.close("all")

        # Mixture
        plot_sample_grid(x_true_cpu, x_init_cpu, res_m["samples_probs"], idx=idx, n_show=n_show_samples,
                         title=f"Mask={kind} | Mixture MwG (a={best_alpha}, s={best_sigma}) | idx={idx}")
        savefig(outdir, f"samples_grid_{kind}_idx{idx}_mix.png")
        plt.close("all")

    # ---- summary row ----
    rows.append(dict(
        mask=kind,
        logp_mean_pseudo=float(logp_p.mean()), logp_std_pseudo=float(logp_p.std()),
        logp_mean_classic=float(logp_c.mean()), logp_std_classic=float(logp_c.std()),
        logp_mean_mix=float(logp_m.mean()), logp_std_mix=float(logp_m.std()),
        var_missing_pseudo=float(div_p["mean_var_missing"]),
        var_missing_classic=float(div_c["mean_var_missing"]),
        var_missing_mix=float(div_m["mean_var_missing"]),
        ham_step_pseudo=float(div_p["mean_hamming_step"]),
        ham_step_classic=float(div_c["mean_hamming_step"]),
        ham_step_mix=float(div_m["mean_hamming_step"]),
        acc_classic=float(np.nanmean(acc_c)) if acc_c is not None else None,
        acc_mix=float(np.nanmean(acc_m)) if acc_m is not None else None,
        alpha=best_alpha, rw_sigma=best_sigma
    ))

# ---- global summary ----
df = pd.DataFrame(rows)
df.to_csv(os.path.join(outdir, "summary_exp7_diversity.csv"), index=False)
print("\nSaved:", os.path.join(outdir, "summary_exp7_diversity.csv"))
print(df)

# recap plots
plt.figure(figsize=(9,4))
x = np.arange(len(df))
plt.bar(x-0.25, df["var_missing_pseudo"],  width=0.25, label="Pseudo")
plt.bar(x,      df["var_missing_classic"], width=0.25, label="Classic")
plt.bar(x+0.25, df["var_missing_mix"],     width=0.25, label="Mixture")
plt.xticks(x, df["mask"])
plt.ylabel("Mean per-pixel variance on missing region (over samples)")
plt.title("EXP7 — posterior diversity (variance) on missing pixels")
plt.grid(True, alpha=0.2)
plt.legend()
savefig(outdir, "recap_var_missing.png")
plt.show()

plt.figure(figsize=(9,4))
plt.bar(x-0.25, df["ham_step_pseudo"],  width=0.25, label="Pseudo")
plt.bar(x,      df["ham_step_classic"], width=0.25, label="Classic")
plt.bar(x+0.25, df["ham_step_mix"],     width=0.25, label="Mixture")
plt.xticks(x, df["mask"])
plt.ylabel("Mean Hamming distance between consecutive samples (missing region)")
plt.title("EXP7 — mode switching proxy (step-to-step changes)")
plt.grid(True, alpha=0.2)
plt.legend()
savefig(outdir, "recap_hamming_step.png")
plt.show()

print("DONE. OUT:", outdir)


# In[ ]:


import os, json, math, datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def savefig(outdir, name):
    path = os.path.join(outdir, name)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print("saved:", path)

@torch.no_grad()
def bernoulli_ll_missing_per_image(x_true, probs, mask_obs, eps=1e-6):
    """
    Retourne log p(x_miss_true | probs) PAR IMAGE (B,) en ne comptant que les pixels manquants.
    Robuste via BCE clamp.
    """
    x = x_true.view(x_true.size(0), -1)
    p = probs.view(probs.size(0), -1).clamp(eps, 1 - eps)
    miss = (1 - mask_obs.view(mask_obs.size(0), -1))  # 1 sur manquant
    ll_pix = -F.binary_cross_entropy(p, x, reduction="none")
    ll_pix = ll_pix * miss
    return ll_pix.sum(dim=1)  # (B,)

@torch.no_grad()
def collect_posterior_samples(
    model, x_true, x_init, mask,
    method="pseudo",              # "pseudo" | "classic" | "mixture"
    n_iters=16000,
    burn_in=2000,
    thinning=200,
    warmup_pg=50,
    # mixture params
    alpha=0.5,
    rw_sigma=0.15,
    # classic params
    proposal_scale=1.0,
):
    """
    Collecte des échantillons (probabilities) après burn-in avec thinning.
    Retour:
      samples_probs: (K,B,1,28,28) sur CPU
      lls_per_image: (K,B) sur CPU   [log p(x_miss_true|x_obs)]
    """
    x = x_init.clone()
    z = None

    if method in ["classic", "mixture"]:
        # warmup PG pour init plus stable
        for _ in range(warmup_pg):
            x_out, _ = pseudo_gibbs_step(model, x, mask)
            x = torch.bernoulli(x_out)
        mu, logvar = model.encode(x.view(-1, 784))
        z = model.reparameterize(mu, logvar)

    samples = []
    lls = []

    for t in range(n_iters):
        if method == "pseudo":
            x_out, _ = pseudo_gibbs_step(model, x, mask)

        elif method == "classic":
            # classic MwG: proposer via encoder (comme ton metropolis_within_gibbs_step actuel)
            x_out, z, _acc = metropolis_within_gibbs_step(
                model, x, z, mask,
                return_accept=True,
                adaptive=False,
                proposal_scale=proposal_scale
            )

        elif method == "mixture":
            # mixture MwG: nécessite ta fonction run/step mixture; si tu as un step dédié, remplace ici.
            # Ici je suppose que tu as une version step-like:
            #   x_out, z, acc = metropolis_within_gibbs_step_mixture(...)
            # Sinon: on peut appeler une "step" mixture que tu as déjà utilisée dans run_chain_with_tracking_mixture.
            x_out, z, _acc = metropolis_within_gibbs_step_mixture(
                model, x, z, mask,
                alpha=alpha,
                rw_sigma=rw_sigma,
                return_accept=True
            )

        else:
            raise ValueError("method must be pseudo|classic|mixture")

        x = torch.bernoulli(x_out)

        keep = (t >= burn_in) and ((t - burn_in) % thinning == 0)
        if keep:
            probs = x_out.detach()
            ll_img = bernoulli_ll_missing_per_image(x_true, probs, mask)
            samples.append(probs.cpu())
            lls.append(ll_img.cpu())

    samples_probs = torch.stack(samples, dim=0)   # (K,B,1,28,28)
    lls_per_image = torch.stack(lls, dim=0)       # (K,B)
    return samples_probs, lls_per_image

def plot_logp_hist_batchmean(lls_p, lls_c, lls_m, title="", bins=25):
    """
    Histogramme sur la moyenne batch: chaque sample -> mean over B.
    """
    vp = lls_p.mean(dim=1).numpy()
    vc = lls_c.mean(dim=1).numpy()
    vm = lls_m.mean(dim=1).numpy()

    plt.figure(figsize=(8,4))
    plt.hist(vp, bins=bins, alpha=0.6, label="Pseudo")
    plt.hist(vc, bins=bins, alpha=0.6, label="Classic")
    plt.hist(vm, bins=bins, alpha=0.6, label="Mixture")
    plt.xlabel("logp per sample (batch mean)")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()

def plot_logp_hist_per_image(lls_p, lls_c, lls_m, idx=0, title="", bins=25):
    """
    Histogramme par image idx: chaque sample -> logp(image idx).
    """
    vp = lls_p[:, idx].numpy()
    vc = lls_c[:, idx].numpy()
    vm = lls_m[:, idx].numpy()

    plt.figure(figsize=(8,4))
    plt.hist(vp, bins=bins, alpha=0.6, label="Pseudo")
    plt.hist(vc, bins=bins, alpha=0.6, label="Classic")
    plt.hist(vm, bins=bins, alpha=0.6, label="Mixture")
    plt.xlabel(f"logp per sample (image idx={idx})")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()

def show_mean_vs_samples(samples_probs, x_true, x_init, mask, idx=0, title=""):
    """
    Affiche: True | Init | Mean(probs) | + quelques samples binarisés
    """
    K = samples_probs.size(0)
    mean_probs = samples_probs.mean(dim=0)  # (B,1,28,28)

    # choisis 6 samples espacés
    pick = np.linspace(0, K-1, num=min(6, K)).astype(int)

    cols = 3 + len(pick)  # True, Init, Mean + samples
    fig, axes = plt.subplots(1, cols, figsize=(2.2*cols, 2.2))
    imgs = [
        x_true[idx].cpu(),
        x_init[idx].cpu(),
        mean_probs[idx].cpu(),
    ]
    names = ["True", "Init", "Posterior mean"]
    for s in pick:
        imgs.append((samples_probs[s, idx] > 0.5).float())
        names.append(f"S{s}")

    for j in range(cols):
        axes[j].imshow(imgs[j].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[j].axis("off")
        axes[j].set_title(names[j], fontsize=9)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def show_variance_map(samples_probs, mask, idx=0, title=""):
    """
    Variance pixelwise sur probs, et on l’affiche surtout sur la zone manquante.
    """
    var = samples_probs.var(dim=0)  # (B,1,28,28)
    v = var[idx].squeeze()          # (28,28)
    miss = (1 - mask[idx]).squeeze().cpu()

    plt.figure(figsize=(4,4))
    plt.imshow((v.cpu()*miss).numpy(), cmap="viridis")
    plt.title(title + " — variance on missing")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()



def pick_hard_idx(lls_p, mask, topk=5):
    """
    Trouve des idx difficiles = plus faible logp moyen (pseudo).
    lls_p: (K,B) logp per image for pseudo
    """
    mean_ll = lls_p.mean(dim=0)  # (B,)
    vals, idxs = torch.topk(-mean_ll, k=min(topk, mean_ll.numel()))  # plus négatif = plus dur
    return idxs.tolist()

# exemple: tu prendras idxs_hard[0] ensuite

# =============================
# CONFIG
# =============================
kind = "top"            # "top" | "bottom" | "center" | "random50"
n_iters   = 16000
burn_in   = 2000
thinning  = 200         # garde ~ (16000-2000)/200 ≈ 70 samples
warmup_pg = 50

best_alpha = 0.5
best_sigma = 0.25       # <-- ton choix

# save dir
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = ensure_dir(os.path.join("results", f"exp7_samples_diversity_{kind}_{run_id}"))
print("OUT:", outdir)

with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(dict(kind=kind, n_iters=n_iters, burn_in=burn_in, thinning=thinning,
                   warmup_pg=warmup_pg, alpha=best_alpha, rw_sigma=best_sigma), f, indent=2)

# data
mask = make_mask(x_true, kind=kind)
x_init = init_with_noise(x_true, mask)

# =============================
# COLLECT samples
# =============================
samples_p, lls_p = collect_posterior_samples(
    model, x_true, x_init, mask,
    method="pseudo",
    n_iters=n_iters, burn_in=burn_in, thinning=thinning
)

samples_c, lls_c = collect_posterior_samples(
    model, x_true, x_init, mask,
    method="classic",
    n_iters=n_iters, burn_in=burn_in, thinning=thinning,
    warmup_pg=warmup_pg,
    proposal_scale=1.0
)

samples_m, lls_m = collect_posterior_samples(
    model, x_true, x_init, mask,
    method="mixture",
    n_iters=n_iters, burn_in=burn_in, thinning=thinning,
    warmup_pg=warmup_pg,
    alpha=best_alpha,
    rw_sigma=best_sigma
)

print("K samples kept:", samples_p.size(0))

# =============================
# Pick a hard image index
# =============================
hard_idxs = pick_hard_idx(lls_p, mask, topk=5)
idx = hard_idxs[0]
print("Hard idx candidates:", hard_idxs, "| using idx=", idx)

# =============================
# 1) Histogram batch-mean
# =============================
plot_logp_hist_batchmean(
    lls_p, lls_c, lls_m,
    title=f"Mask={kind} — distribution of log p(x_miss_true | x_obs) over posterior samples"
)
savefig(outdir, f"logp_hist_batchmean_{kind}.png")
plt.close("all")

# =============================
# 2) Histogram per-image
# =============================
plot_logp_hist_per_image(
    lls_p, lls_c, lls_m, idx=idx,
    title=f"Mask={kind} — logp distribution for one image (idx={idx})"
)
savefig(outdir, f"logp_hist_perimage_{kind}_idx{idx}.png")
plt.close("all")

# =============================
# 3) Mean vs samples + variance maps
# =============================
show_mean_vs_samples(samples_p, x_true, x_init, mask, idx=idx, title=f"Mask={kind} | Pseudo | idx={idx}")
savefig(outdir, f"samples_pseudo_{kind}_idx{idx}.png")
plt.close("all")

show_mean_vs_samples(samples_c, x_true, x_init, mask, idx=idx, title=f"Mask={kind} | Classic MwG | idx={idx}")
savefig(outdir, f"samples_classic_{kind}_idx{idx}.png")
plt.close("all")

show_mean_vs_samples(samples_m, x_true, x_init, mask, idx=idx, title=f"Mask={kind} | Mixture MwG (a={best_alpha}, s={best_sigma}) | idx={idx}")
savefig(outdir, f"samples_mixture_{kind}_idx{idx}.png")
plt.close("all")

show_variance_map(samples_p, mask, idx=idx, title=f"Mask={kind} | Pseudo | idx={idx}")
savefig(outdir, f"var_pseudo_{kind}_idx{idx}.png")
plt.close("all")

show_variance_map(samples_c, mask, idx=idx, title=f"Mask={kind} | Classic | idx={idx}")
savefig(outdir, f"var_classic_{kind}_idx{idx}.png")
plt.close("all")

show_variance_map(samples_m, mask, idx=idx, title=f"Mask={kind} | Mixture (a={best_alpha}, s={best_sigma}) | idx={idx}")
savefig(outdir, f"var_mixture_{kind}_idx{idx}.png")
plt.close("all")

# =============================
# Save raw arrays (optional)
# =============================
torch.save(dict(
    samples_p=samples_p, lls_p=lls_p,
    samples_c=samples_c, lls_c=lls_c,
    samples_m=samples_m, lls_m=lls_m,
    idx=idx
), os.path.join(outdir, "raw_samples.pt"))
print("saved:", os.path.join(outdir, "raw_samples.pt"))


# ### Missing at random , comparaison pseudo, classique, mixture

# In[ ]:


# ============================================
# Missing-at-random sweep: p in {0.4,0.6,0.8,0.9}
# Compare: Pseudo vs Classic MwG vs Mixture MwG (best alpha/sigma)
# Saves: configs, per-p histories, evolution plots, grids, recap plots
# ============================================

import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- YOU SET THESE (from your previous best grid) ----------
best_alpha = 0
best_sigma = 0.15

# --------- sweep settings ----------
p_list = [0.4, 0.6, 0.8, 0.9]          # missing probability
scenarios = [f"mar_p{p}" for p in p_list]

# chain params
n_iters     = 16000
burn_in     = 2000
thinning    = 20
eval_every  = 100
warmup_pg   = 50

assert n_iters > burn_in, "ERROR: n_iters must be > burn_in (otherwise no samples are kept)."

# classic MwG options (mono-chain)
use_adaptive_accept = False
target_accept = 0.15
adapt_lr = 0.005
freeze_adapt_after_burnin = True

idxs_show = [0, 1, 2, 5]

# ========= OUTDIR =========
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join("results", f"exp_mar_sweep_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

# save config
config = dict(
    p_list=p_list,
    best_alpha=best_alpha, best_sigma=best_sigma,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every, warmup_pg=warmup_pg,
    use_adaptive_accept=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr,
    freeze_adapt_after_burnin=freeze_adapt_after_burnin,
    idxs_show=idxs_show,
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

summary_rows = []

# ============================================
# MAIN LOOP over p (missing-at-random)
# ============================================
for p in p_list:
    print("\n" + "="*60)
    print(f"MAR missing p = {p}")
    print("="*60)

    # ---- mask MAR ----
    # Assumes you already have: make_random_mask(x_true, missing_rate=...)
    mask = make_random_mask(x_true, missing_rate=p)
    x_init = init_with_noise(x_true, mask)

    # -----------------
    # PSEUDO
    # -----------------
    mean_pseudo, summ_p, hist_p = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="pseudo",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every
    )

    # -----------------
    # CLASSIC MwG (mono)
    # -----------------
    mean_classic, summ_c, hist_c = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="mwg",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        adaptive_accept=use_adaptive_accept,
        target_accept=target_accept,
        adapt_lr=adapt_lr,
        freeze_adapt_after_burnin=freeze_adapt_after_burnin
    )

    # -----------------
    # MIXTURE MwG (best alpha/sigma)
    # -----------------
    mean_mix, summ_m, hist_m = run_chain_with_tracking_mixture(
        model, x_true, x_init, mask,
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        alpha=best_alpha,
        rw_sigma=best_sigma
    )

    # -----------------
    # FINAL SCORES (on posterior mean)
    # NOTE: this is NOT "last iteration", it's the score of the final posterior-mean imputation.
    # -----------------
    mean_pseudo_d  = mean_pseudo.to(device)
    mean_classic_d = mean_classic.to(device)
    mean_mix_d     = mean_mix.to(device)

    f1_p = f1_missing(x_true, mean_pseudo_d, mask)
    f1_c = f1_missing(x_true, mean_classic_d, mask)
    f1_m = f1_missing(x_true, mean_mix_d, mask)

    logp_p = summ_p["logp_final"]
    logp_c = summ_c["logp_final"]
    logp_m = summ_m["logp_final"]

    acc_c = summ_c.get("acc_final", None)
    acc_m = summ_m.get("acc_final", None)

    print(f"F1 missing:  Pseudo={f1_p:.4f} | Classic={f1_c:.4f} | Mixture={f1_m:.4f}")
    print(f"log p(...):  Pseudo={logp_p:.2f} | Classic={logp_c:.2f} | Mixture={logp_m:.2f}")
    if acc_c is not None or acc_m is not None:
        print(f"Acceptance:  Classic={None if acc_c is None else round(acc_c,3)} | Mixture={None if acc_m is None else round(acc_m,3)}")

    # -----------------
    # EVOLUTION PLOTS (triplet) + save
    # -----------------
    histP = {"steps": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}

    histC = {"steps": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c:
        histC["acc"] = hist_c["acc"]

    histM = {"steps": hist_m["steps"], "f1": hist_m["f1"], "logp": hist_m["logp"]}
    if "acc" in hist_m:
        histM["acc"] = hist_m["acc"]

    prefix = f"evolution_mar_p{p:.2f}".replace(".", "p")
    plot_evolution_triplet(
        histP, histC, histM,
        title=f"MAR p={p:.2f} | a={best_alpha}, s={best_sigma}",
        outdir=outdir,
        prefix=prefix,
        legends=("Pseudo", "Classic MwG", "Mixture MwG")
    )

    # -----------------
    # ONE GRID WITH 3 METHODS (5 columns) + save
    # -----------------
    show_grid_triplet(
        idxs_show,
        x_true.detach().cpu(),
        x_init.detach().cpu(),
        mean_pseudo.detach().cpu(),
        mean_classic.detach().cpu(),
        mean_mix.detach().cpu(),
        title=f"MAR p={p:.2f} | F1 P/C/M = {f1_p:.3f}/{f1_c:.3f}/{f1_m:.3f}",
        outdir=outdir,
        name=f"grid_triplet_mar_p{p:.2f}.png".replace(".", "p"),
        labels=("Pseudo", "Classic", "Mixture")
    )
    plt.close("all")

    # -----------------
    # SAVE HISTORIES (CSV)
    # -----------------
    pd.DataFrame({"step": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}).to_csv(
        os.path.join(outdir, f"hist_pseudo_mar_p{p:.2f}.csv".replace(".", "p")), index=False
    )

    df_c = {"step": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c: df_c["acc"] = hist_c["acc"]
    pd.DataFrame(df_c).to_csv(
        os.path.join(outdir, f"hist_classic_mar_p{p:.2f}.csv".replace(".", "p")), index=False
    )

    df_m = {"step": hist_m["steps"], "f1": hist_m["f1"], "logp": hist_m["logp"]}
    if "acc" in hist_m: df_m["acc"] = hist_m["acc"]
    pd.DataFrame(df_m).to_csv(
        os.path.join(outdir, f"hist_mixture_mar_p{p:.2f}.csv".replace(".", "p")), index=False
    )

    # -----------------
    # SUMMARY ROW
    # gains are computed on FINAL scores (posterior mean at end)
    # -----------------
    summary_rows.append(dict(
        p=float(p),
        f1_pseudo=float(f1_p),   logp_pseudo=float(logp_p),
        f1_classic=float(f1_c),  logp_classic=float(logp_c),  acc_classic=(None if acc_c is None else float(acc_c)),
        f1_mix=float(f1_m),      logp_mix=float(logp_m),      acc_mix=(None if acc_m is None else float(acc_m)),
        gain_f1_mix_vs_classic=float(f1_m - f1_c),
        gain_logp_mix_vs_classic=float(logp_m - logp_c),
        alpha=float(best_alpha), rw_sigma=float(best_sigma),
    ))

# ============================================
# GLOBAL SUMMARY + RECAP PLOTS
# ============================================
df_sum = pd.DataFrame(summary_rows).sort_values("p")
df_sum.to_csv(os.path.join(outdir, "summary_mar_sweep.csv"), index=False)
print("\nSaved:", os.path.join(outdir, "summary_mar_sweep.csv"))
print(df_sum)

# ----- F1 vs p -----
plt.figure(figsize=(8,4))
plt.plot(df_sum["p"], df_sum["f1_pseudo"],  "-o", label="Pseudo")
plt.plot(df_sum["p"], df_sum["f1_classic"], "-o", label="Classic MwG")
plt.plot(df_sum["p"], df_sum["f1_mix"],     "-o", label="Mixture MwG")
plt.xlabel("Missing probability p (MAR)")
plt.ylabel("F1 on missing pixels (final posterior mean)")
plt.title("MAR sweep — F1 vs p")
plt.grid(True, alpha=0.2)
plt.legend()
savefig(outdir, "recap_f1_vs_p.png")
plt.show()
plt.close()

# ----- logp vs p -----
plt.figure(figsize=(8,4))
plt.plot(df_sum["p"], df_sum["logp_pseudo"],  "-o", label="Pseudo")
plt.plot(df_sum["p"], df_sum["logp_classic"], "-o", label="Classic MwG")
plt.plot(df_sum["p"], df_sum["logp_mix"],     "-o", label="Mixture MwG")
plt.xlabel("Missing probability p (MAR)")
plt.ylabel("MC log-likelihood (final)")
plt.title("MAR sweep — logp vs p")
plt.grid(True, alpha=0.2)
plt.legend()
savefig(outdir, "recap_logp_vs_p.png")
plt.show()
plt.close()

# ----- acceptance vs p (classic + mixture) -----
if df_sum["acc_classic"].notna().any() or df_sum["acc_mix"].notna().any():
    plt.figure(figsize=(8,4))
    if df_sum["acc_classic"].notna().any():
        plt.plot(df_sum["p"], df_sum["acc_classic"], "-o", label="Classic MwG acc")
    if df_sum["acc_mix"].notna().any():
        plt.plot(df_sum["p"], df_sum["acc_mix"], "-o", label="Mixture MwG acc")
    plt.xlabel("Missing probability p (MAR)")
    plt.ylabel("Acceptance (last window)")
    plt.title("MAR sweep — acceptance vs p")
    plt.grid(True, alpha=0.2)
    plt.legend()
    savefig(outdir, "recap_acc_vs_p.png")
    plt.show()
    plt.close()

# ----- gains vs p -----
plt.figure(figsize=(8,4))
plt.plot(df_sum["p"], df_sum["gain_f1_mix_vs_classic"], "-o")
plt.xlabel("Missing probability p (MAR)")
plt.ylabel("ΔF1 (Mixture - Classic)")
plt.title("MAR sweep — gain in F1")
plt.grid(True, alpha=0.2)
savefig(outdir, "recap_gain_f1_vs_p.png")
plt.show()
plt.close()

plt.figure(figsize=(8,4))
plt.plot(df_sum["p"], df_sum["gain_logp_mix_vs_classic"], "-o")
plt.xlabel("Missing probability p (MAR)")
plt.ylabel("Δlogp (Mixture - Classic)")
plt.title("MAR sweep — gain in logp")
plt.grid(True, alpha=0.2)
savefig(outdir, "recap_gain_logp_vs_p.png")
plt.show()
plt.close()

print("DONE. OUT:", outdir)


# ### Comparaison Pseudo vs Mgw vs Mgw adaptatif

# In[ ]:


import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# CONFIG: Pseudo vs Classic vs Adaptive MwG
# =========================================
scenarios = ["bottom", "center", "top", "random50"]

n_iters    = 16000
burn_in    = 2000
thinning   = 20
eval_every = 100
warmup_pg  = 50

# multichain (optionnel)
use_multichain = False
n_chains = 5

# adaptive MwG (Uniquement pour la version "adaptive")
use_adaptive_accept = True
target_accept = 0.13
adapt_lr = 0.005
freeze_adapt_after_burnin = True

idxs_show = [0, 1, 2, 5]

# ========= OUTDIR + save config =========
run_id = 1
outdir = os.path.join("results", f"exp_compare_classic_vs_adaptive_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

config = dict(
    scenarios=scenarios,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every, warmup_pg=warmup_pg,
    use_multichain=use_multichain, n_chains=n_chains,
    use_adaptive_accept=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr,
    freeze_adapt_after_burnin=freeze_adapt_after_burnin,
    idxs_show=idxs_show
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

summary_rows = []

for kind in scenarios:
    print("\n" + "="*60)
    print(f"MASK = {kind}")
    print("="*60)

    mask = make_mask(x_true, kind=kind)
    x_init = init_with_noise(x_true, mask)

    # -----------------
    # PSEUDO (baseline)
    # -----------------
    mean_pseudo, summ_p, hist_p = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="pseudo",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every
    )

    # -----------------
    # CLASSIC MwG (non-adaptive)
    # -----------------
    if use_multichain:
        # Si tu veux vraiment multichain "classic", mets adaptive_accept=False ici.
        mean_classic, std_mwg, summaries, histories, accs, rhos = run_mwg_multichain(
            model, x_true, x_init, mask,
            n_chains=n_chains,
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=False,            # <-- CLASSIC
            target_accept=target_accept,      # ignoré si adaptive_accept=False
            adapt_lr=adapt_lr,                # ignoré si adaptive_accept=False
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        hist_c = histories[0]
        summ_c = summaries[0]

        std_mwg = std_mwg.to(device)
        miss = (1 - mask).bool()
        mixing_std = std_mwg[miss].mean().item()
        print(f"Classic MwG (multi) acceptance per chain: {[None if a is None else round(a,3) for a in accs]}")
        print(f"Inter-chain std on missing pixels (mean): {mixing_std:.4f}")

    else:
        mean_classic, summ_c, hist_c = run_chain_with_tracking(
            model, x_true, x_init, mask,
            method="mwg",
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=False,            # <-- CLASSIC
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        mixing_std = None

    # -----------------
    # ADAPTIVE MwG
    # -----------------
    if use_multichain:
        # (optionnel) si tu veux aussi multichain pour l’adaptive : sinon laisse False au-dessus.
        mean_adapt, std_mwg2, summaries2, histories2, accs2, rhos2 = run_mwg_multichain(
            model, x_true, x_init, mask,
            n_chains=n_chains,
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=True,             # <-- ADAPTIVE
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
        hist_a = histories2[0]
        summ_a = summaries2[0]
        print(f"Adaptive MwG (multi) acceptance per chain: {[None if a is None else round(a,3) for a in accs2]}")
        print(f"Adaptive MwG (multi) rho final per chain: {[None if r is None else round(r,3) for r in rhos2]}")

    else:
        mean_adapt, summ_a, hist_a = run_chain_with_tracking(
            model, x_true, x_init, mask,
            method="mwg",
            n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
            warmup_pg=warmup_pg,
            adaptive_accept=True,             # <-- ADAPTIVE
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )

    # -----------------
    # Final scores
    # -----------------
    mean_pseudo_d  = mean_pseudo.to(device)
    mean_classic_d = mean_classic.to(device)
    mean_adapt_d   = mean_adapt.to(device)

    f1_p = f1_missing(x_true, mean_pseudo_d, mask)
    f1_c = f1_missing(x_true, mean_classic_d, mask)
    f1_a = f1_missing(x_true, mean_adapt_d, mask)

    logp_p = summ_p["logp_final"]
    logp_c = summ_c["logp_final"]
    logp_a = summ_a["logp_final"]

    acc_c = summ_c.get("acc_final", None)
    acc_a = summ_a.get("acc_final", None)
    rho_a = summ_a.get("rho_final", None)

    print(f"F1 missing:  Pseudo={f1_p:.4f} | Classic={f1_c:.4f} | Adaptive={f1_a:.4f}")
    print(f"log p(...):  Pseudo={logp_p:.2f} | Classic={logp_c:.2f} | Adaptive={logp_a:.2f}")
    if acc_c is not None or acc_a is not None:
        print(f"Acceptance:  Classic={acc_c if acc_c is not None else float('nan'):.3f} | Adaptive={acc_a if acc_a is not None else float('nan'):.3f}")
    if rho_a is not None:
        print(f"Adaptive rho_final: {rho_a:.3f}")

    # -----------------
    # ONE SINGLE EVOLUTION PLOT (Pseudo vs Classic vs Adaptive) + save
    # -----------------
    histP = {"steps": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}

    histC = {"steps": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c:
        histC["acc"] = hist_c["acc"]

    histA = {"steps": hist_a["steps"], "f1": hist_a["f1"], "logp": hist_a["logp"]}
    if "acc" in hist_a:
        histA["acc"] = hist_a["acc"]

    plot_evolution_triplet(
        histP, histC, histA,
        title=f"Mask={kind} | Pseudo vs Classic vs Adaptive",
        outdir=outdir,
        prefix=f"evolution_triplet_{kind}", legends=("Pseudo", "Classic MwG", "Adaptive MwG")
    )
    plt.close("all")

    # -----------------
    # GRIDS (CPU!) — show_grid sauvegarde déjà via name=
    # -----------------

    show_grid_triplet(
        idxs_show,
        x_true.detach().cpu(),
        x_init.detach().cpu(),
        mean_pseudo.detach().cpu(),
        mean_classic.detach().cpu(),
        mean_adapt.detach().cpu(),
        title=f"Mask={kind}",
        outdir=outdir,
        name=f"grid_triplet_{kind}.png",
        labels=("Pseudo Gibbs", "Classic MwG", f"Adaptive MwG"),
    )
    plt.close("all")
    # -----------------
    # SAVE histories
    # -----------------
    pd.DataFrame({"step": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}).to_csv(
        os.path.join(outdir, f"hist_pseudo_{kind}.csv"), index=False
    )

    hc = {"step": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c:
        hc["acc"] = hist_c["acc"]
    pd.DataFrame(hc).to_csv(os.path.join(outdir, f"hist_classic_{kind}.csv"), index=False)

    ha = {"step": hist_a["steps"], "f1": hist_a["f1"], "logp": hist_a["logp"]}
    if "acc" in hist_a:
        ha["acc"] = hist_a["acc"]
    if "rho" in hist_a:
        ha["rho"] = hist_a["rho"][:len(hist_a["steps"])]
    pd.DataFrame(ha).to_csv(os.path.join(outdir, f"hist_adaptive_{kind}.csv"), index=False)

    # -----------------
    # SUMMARY row
    # -----------------
    summary_rows.append(dict(
        mask=kind,
        f1_pseudo=float(f1_p),   logp_pseudo=float(logp_p),
        f1_classic=float(f1_c),  logp_classic=float(logp_c),  acc_classic=float(acc_c) if acc_c is not None else None,
        f1_adapt=float(f1_a),    logp_adapt=float(logp_a),    acc_adapt=float(acc_a) if acc_a is not None else None,
        rho_final=float(rho_a) if rho_a is not None else None,
        gain_f1_adapt_vs_classic=float(f1_a - f1_c),
        gain_logp_adapt_vs_classic=float(logp_a - logp_c),
        mixing_std=mixing_std,
        multichain=use_multichain,
        target_accept=target_accept,
        adapt_lr=adapt_lr
    ))

# =========================
# GLOBAL SUMMARY + recap plots
# =========================
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(outdir, "summary_all_masks.csv"), index=False)
print("\nSaved:", os.path.join(outdir, "summary_all_masks.csv"))
print(df_sum)

x = np.arange(len(df_sum))

plt.figure(figsize=(9,4))
plt.bar(x-0.25, df_sum["f1_pseudo"],  width=0.25, label="Pseudo")
plt.bar(x,      df_sum["f1_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["f1_adapt"],   width=0.25, label="Adaptive MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("F1 missing")
plt.title("F1 by mask | Classic vs Adaptive MwG")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_f1_pseudo_classic_adaptive.png")
plt.show()
plt.close()

plt.figure(figsize=(9,4))
plt.bar(x-0.25, df_sum["logp_pseudo"],  width=0.25, label="Pseudo")
plt.bar(x,      df_sum["logp_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["logp_adapt"],   width=0.25, label="Adaptive MwG")
plt.xticks(x, df_sum["mask"])
plt.ylabel("MC log-likelihood")
plt.title("logp by mask | Classic vs Adaptive MwG")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_logp_pseudo_classic_adaptive.png")
plt.show()
plt.close()

plt.figure(figsize=(9,4))
plt.bar(x, df_sum["gain_f1_adapt_vs_classic"], width=0.4)
plt.xticks(x, df_sum["mask"])
plt.ylabel("ΔF1 (Adaptive - Classic)")
plt.title("Gain F1: adaptive vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_f1_adapt_vs_classic.png")
plt.show()
plt.close()

plt.figure(figsize=(9,4))
plt.bar(x, df_sum["gain_logp_adapt_vs_classic"], width=0.4)
plt.xticks(x, df_sum["mask"])
plt.ylabel("Δlogp (Adaptive - Classic)")
plt.title("Gain logp: adaptive vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_logp_adapt_vs_classic.png")
plt.show()
plt.close()

print("DONE. OUT:", outdir)


# ### Comparaison Pseudo vs Mgw classique vs Mgw multi-chaîne

# In[ ]:


import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

scenarios = ["top", "random50", "bottom", "center"]

n_iters     = 16000
burn_in     = 2000
thinning    = 20
eval_every  = 100
warmup_pg   = 50

# classic MwG (mono-chaîne)
use_adaptive_accept = False
target_accept = 0.15
adapt_lr = 0.005
freeze_adapt_after_burnin = True

# multi-chain MwG
n_chains = 5

idxs_show = [0, 1, 2, 5]

# ========= SAVE DIR =========
run_id = 1
outdir = os.path.join("results", f"exp_compare_classic_vs_multichain_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

# save config
config = dict(
    scenarios=scenarios,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every, warmup_pg=warmup_pg,
    use_adaptive_accept=use_adaptive_accept, target_accept=target_accept, adapt_lr=adapt_lr,
    freeze_adapt_after_burnin=freeze_adapt_after_burnin,
    n_chains=n_chains,
    idxs_show=idxs_show,
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

summary_rows = []

for kind in scenarios:
    print("\n" + "="*60)
    print(f"MASK = {kind}")
    print("="*60)

    mask = make_mask(x_true, kind=kind)
    x_init = init_with_noise(x_true, mask)

    # -----------------
    # PSEUDO (baseline)
    # -----------------
    mean_pseudo, summ_p, hist_p = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="pseudo",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every
    )

    # -----------------
    # CLASSIC MwG (mono chaîne)
    # -----------------
    mean_classic, summ_c, hist_c = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="mwg",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        adaptive_accept=use_adaptive_accept,
        target_accept=target_accept,
        adapt_lr=adapt_lr,
        freeze_adapt_after_burnin=freeze_adapt_after_burnin
    )

    # -----------------
    # MULTI-CHAIN MwG
    # -----------------
    mean_multi, std_multi, summaries_mc, histories_mc, accs_mc, rhos_mc = run_mwg_multichain(
        model, x_true, x_init, mask,
        n_chains=n_chains,
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        adaptive_accept=use_adaptive_accept,
        target_accept=target_accept,
        adapt_lr=adapt_lr,
        freeze_adapt_after_burnin=freeze_adapt_after_burnin
    )

    # pour les courbes multi-chain : soit tu prends la chaîne 0,
    # soit tu peux moyenner les histories (plus long). Ici: chain 0.
    hist_mc = histories_mc[0]
    summ_mc = summaries_mc[0]

    # mixing diag (inter-chain std sur pixels manquants)
    std_multi_d = std_multi.to(device)
    miss = (1 - mask).bool()
    mixing_std = std_multi_d[miss].mean().item()

    print(f"Multi-chain acceptance last-window per chain: {[None if a is None else round(a,3) for a in accs_mc]}")
    print(f"Inter-chain std on missing pixels (mean): {mixing_std:.4f}")

    # -----------------
    # Final scores (sur mean impute)
    # -----------------
    mean_pseudo_d  = mean_pseudo.to(device)
    mean_classic_d = mean_classic.to(device)
    mean_multi_d   = mean_multi.to(device)

    f1_p  = f1_missing(x_true, mean_pseudo_d, mask)
    f1_c  = f1_missing(x_true, mean_classic_d, mask)
    f1_mc = f1_missing(x_true, mean_multi_d, mask)

    logp_p  = summ_p["logp_final"]
    logp_c  = summ_c["logp_final"]
    logp_mc = summ_mc["logp_final"]

    acc_c  = summ_c.get("acc_final", None)
    acc_mc = summ_mc.get("acc_final", None)

    print(f"F1 missing:  Pseudo={f1_p:.4f} | Classic={f1_c:.4f} | Multi={f1_mc:.4f}")
    print(f"log p(...):  Pseudo={logp_p:.2f} | Classic={logp_c:.2f} | Multi={logp_mc:.2f}")
    if acc_c is not None and acc_mc is not None:
        print(f"Acceptance:  Classic={acc_c:.3f} | Multi(chain0)={acc_mc:.3f}")

    # -----------------
    # ONE SINGLE EVOLUTION PLOT (Pseudo vs Classic vs Multi) + save
    # -----------------
    histP = {"steps": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}

    histC = {"steps": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c:
        histC["acc"] = hist_c["acc"]

    histMC = {"steps": hist_mc["steps"], "f1": hist_mc["f1"], "logp": hist_mc["logp"]}
    if "acc" in hist_mc:
        histMC["acc"] = hist_mc["acc"]

    # IMPORTANT: ton plot_evolution_triplet fait des plt.show().
    # Si tu veux aussi sauver proprement: fais-le DANS utils.py juste avant show().

    plot_evolution_triplet(histP, histC, histMC,
                       title=f"Mask={kind}",
                       outdir=outdir,
                       prefix=f"evolution_triplet_{kind}", legends=("Pseudo", "Classic MwG", f"Multi-chain (n={n_chains})"))
    # -----------------
    # Image grids + save (CPU for MPS)
    # show_grid: (True, Init, Posterior mean, Last sample)
    # Ici, on lui donne mean_pseudo vs mean_classic etc.
    # -> Comme show_grid attend (mean_probs, last_bin), tu peux lui passer:
    #    mean_probs = mean_* , last_bin = mean_* (si tu n’as pas last sample stocké)
    #    OU mieux: modifie show_grid pour faire 4 colonnes "True/Init/AlgoA/AlgoB".
    # Ici on reste minimal: on compare mean vs mean (ça marche visuellement).
    # -----------------

    # Pseudo vs Classic (on met pseudo comme "mean_probs" et classic comme "last_bin")

    show_grid_triplet(
        idxs_show,
        x_true.detach().cpu(),
        x_init.detach().cpu(),
        mean_pseudo.detach().cpu(),
        mean_classic.detach().cpu(),
        mean_multi.detach().cpu(),
        title=f"Mask={kind}",
        outdir=outdir,
        name=f"grid_triplet_{kind}.png",
        labels=("Pseudo Gibbs", "Classic MwG", f"Multi (n={n_chains})"),
    )
    plt.close("all")
    # -----------------
    # Save histories (CSV)
    # -----------------
    pd.DataFrame({"step": hist_p["steps"], "f1": hist_p["f1"], "logp": hist_p["logp"]}).to_csv(
        os.path.join(outdir, f"hist_pseudo_{kind}.csv"), index=False
    )

    df_c = {"step": hist_c["steps"], "f1": hist_c["f1"], "logp": hist_c["logp"]}
    if "acc" in hist_c: df_c["acc"] = hist_c["acc"]
    pd.DataFrame(df_c).to_csv(os.path.join(outdir, f"hist_classic_{kind}.csv"), index=False)

    df_mc = {"step": hist_mc["steps"], "f1": hist_mc["f1"], "logp": hist_mc["logp"]}
    if "acc" in hist_mc: df_mc["acc"] = hist_mc["acc"]
    pd.DataFrame(df_mc).to_csv(os.path.join(outdir, f"hist_multichain_chain0_{kind}.csv"), index=False)

    # -----------------
    # Summary row
    # -----------------
    summary_rows.append(dict(
        mask=kind,
        f1_pseudo=float(f1_p), logp_pseudo=float(logp_p),
        f1_classic=float(f1_c), logp_classic=float(logp_c), acc_classic=float(acc_c) if acc_c is not None else None,
        f1_multi=float(f1_mc), logp_multi=float(logp_mc), acc_multi=float(acc_mc) if acc_mc is not None else None,
        gain_f1_multi_vs_classic=float(f1_mc - f1_c),
        gain_logp_multi_vs_classic=float(logp_mc - logp_c),
        mixing_std=float(mixing_std),
        n_chains=int(n_chains),
        adaptive=bool(use_adaptive_accept)
    ))

# =========================
# Save global summary + recap plots
# =========================
df_sum = pd.DataFrame(summary_rows)
df_sum.to_csv(os.path.join(outdir, "summary_all_masks.csv"), index=False)
print("\nSaved:", os.path.join(outdir, "summary_all_masks.csv"))
print(df_sum)

# F1 recap
plt.figure(figsize=(9,4))
x = np.arange(len(df_sum))
plt.bar(x-0.25, df_sum["f1_pseudo"], width=0.25, label="Pseudo")
plt.bar(x,      df_sum["f1_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["f1_multi"], width=0.25, label=f"Multi-chain (n={n_chains})")
plt.xticks(x, df_sum["mask"])
plt.ylabel("F1 missing")
plt.title("F1 by mask | Pseudo vs Classic vs Multi-chain")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_f1_pseudo_classic_multichain.png")
plt.show()

# logp recap
plt.figure(figsize=(9,4))
plt.bar(x-0.25, df_sum["logp_pseudo"], width=0.25, label="Pseudo")
plt.bar(x,      df_sum["logp_classic"], width=0.25, label="Classic MwG")
plt.bar(x+0.25, df_sum["logp_multi"], width=0.25, label=f"Multi-chain (n={n_chains})")
plt.xticks(x, df_sum["mask"])
plt.ylabel("MC log-likelihood")
plt.title("logp by mask | Pseudo vs Classic vs Multi-chain")
plt.legend()
plt.grid(True, alpha=0.2)
savefig(outdir, "summary_logp_pseudo_classic_multichain.png")
plt.show()

# gains
plt.figure(figsize=(9,4))
plt.bar(x, df_sum["gain_f1_multi_vs_classic"], width=0.35)
plt.xticks(x, df_sum["mask"])
plt.ylabel("ΔF1 (Multi - Classic)")
plt.title("Gain F1: multi-chain vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_f1_multi_vs_classic.png")
plt.show()

plt.figure(figsize=(9,4))
plt.bar(x, df_sum["gain_logp_multi_vs_classic"], width=0.35)
plt.xticks(x, df_sum["mask"])
plt.ylabel("Δlogp (Multi - Classic)")
plt.title("Gain logp: multi-chain vs classic")
plt.grid(True, alpha=0.2)
savefig(outdir, "gain_logp_multi_vs_classic.png")
plt.show()

print("DONE. OUT:", outdir)


# ### Mean vs last

# In[ ]:


# ============================================
# Mean vs Last study (MwG + Pseudo) by mask
# - Plots: F1_mean vs F1_last for each mask
# - Saves: CSV + barplots + grids (mean vs last)
# Requires:
#   - run_chain_mean_last (yours)
#   - make_random_mask, make_mask (structured top/bottom/center), init_with_noise
#   - savefig(outdir, name) from your utils
# ============================================

import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# -------------------------
# CONFIG
# -------------------------
scenarios = ["top", "bottom", "center", "random50"]
p_missing_random = 0.5

n_iters   = 12000
burn_in   = 2000
thinning  = 20
warmup_pg = 50

proposal_scale = 1.0       # classic MwG
adaptive_mwg   = False     # keep False if unstable

idxs_show = [0, 1, 2, 5]

# -------------------------
# OUTDIR
# -------------------------
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join("results", f"exp_mean_vs_last_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

config = dict(
    scenarios=scenarios,
    p_missing_random=p_missing_random,
    n_iters=n_iters, burn_in=burn_in, thinning=thinning, warmup_pg=warmup_pg,
    proposal_scale=proposal_scale,
    adaptive_mwg=adaptive_mwg,
    idxs_show=idxs_show
)
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# -------------------------
# Helper: grid Mean vs Last (generic)
# -------------------------
def show_mean_vs_last_grid(
    idxs, x_true, x_init, mean_probs, last_bin,
    title, outdir, name, label_mean="Mean", label_last="Last"
):
    cols = 4
    rows = len(idxs)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows))
    colnames = ["True", "Masked init", label_mean, label_last]

    for i, idx in enumerate(idxs):
        imgs = [
            x_true[idx].detach().cpu(),
            x_init[idx].detach().cpu(),
            mean_probs[idx].detach().cpu(),
            last_bin[idx].detach().cpu(),
        ]
        for j in range(cols):
            ax = axes[i, j] if rows > 1 else axes[j]
            ax.imshow(imgs[j].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(colnames[j], fontsize=10, fontweight="bold")

    plt.suptitle(title)
    plt.tight_layout()
    savefig(outdir, name)   # <-- your utils.savefig(outdir, name)
    plt.show()
    plt.close()

# -------------------------
# MAIN LOOP
# -------------------------
rows = []

for kind in scenarios:
    print("\n" + "=" * 60)
    print("MASK =", kind)
    print("=" * 60)

    # mask + init
    if kind == "random50":
        mask = make_random_mask(x_true, missing_rate=p_missing_random)
    else:
        mask = make_mask(x_true, kind=kind)  # structured
    x_init = init_with_noise(x_true, mask)

    # --- PSEUDO ---
    mean_p, last_p, f1m_p, f1l_p, logp_p, acc_p, kept_p, dt_p = run_chain_mean_last(
        model, x_true, x_init, mask,
        method="pseudo",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning
    )

    # --- CLASSIC MwG ---
    mean_c, last_c, f1m_c, f1l_c, logp_c, acc_c, kept_c, dt_c = run_chain_mean_last(
        model, x_true, x_init, mask,
        method="mwg",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning,
        warmup_pg=warmup_pg,
        adaptive=adaptive_mwg,
        proposal_scale=proposal_scale
    )
    print(f1_m, f1l_p, logp_c, acc_c, kept_c, dt_c, kept_c, kept_p, dt_p, logp_p)
    print(f"Pseudo  : F1_mean={f1m_p:.4f} | F1_last={f1l_p:.4f} | logp={logp_p:.2f} | kept={kept_p} | t={dt_p:.1f}s")
    print(f"Classic : F1_mean={f1m_c:.4f} | F1_last={f1l_c:.4f} | logp={logp_c:.2f} | acc={acc_c} | kept={kept_c} | t={dt_c:.1f}s")

    # save grids: mean vs last for each method
    show_mean_vs_last_grid(
        idxs_show,
        x_true, x_init,
        mean_p, last_p,
        title=f"Mask={kind} | PSEUDO mean vs last | F1mean={f1m_p:.3f} F1last={f1l_p:.3f}",
        outdir=outdir,
        name=f"grid_mean_vs_last_pseudo_{kind}.png",
        label_mean="Pseudo mean", label_last="Pseudo last"
    )
    show_mean_vs_last_grid(
        idxs_show,
        x_true, x_init,
        mean_c, last_c,
        title=f"Mask={kind} | CLASSIC MwG mean vs last | F1mean={f1m_c:.3f} F1last={f1l_c:.3f} acc={acc_c}",
        outdir=outdir,
        name=f"grid_mean_vs_last_classic_{kind}.png",
        label_mean="Classic mean", label_last="Classic last"
    )

    rows.append(dict(
        mask=kind,
        # pseudo
        f1_mean_pseudo=float(f1m_p),
        f1_last_pseudo=float(f1l_p),
        logp_pseudo=float(logp_p),
        time_pseudo_s=float(dt_p),
        kept_pseudo=int(kept_p),
        # classic
        f1_mean_classic=float(f1m_c),
        f1_last_classic=float(f1l_c),
        logp_classic=float(logp_c),
        acc_classic=float(acc_c) if acc_c is not None else None,
        time_classic_s=float(dt_c),
        kept_classic=int(kept_c),
        # deltas (mean-last)
        gap_pseudo=float(f1m_p - f1l_p),
        gap_classic=float(f1m_c - f1l_c),
    ))

df = pd.DataFrame(rows)
csv_path = os.path.join(outdir, "mean_vs_last_summary.csv")
df.to_csv(csv_path, index=False)
print("\nSaved:", csv_path)
print(df)

# -------------------------
# PLOTS: F1 mean vs last by mask
# -------------------------
x = np.arange(len(df))
w = 0.20

# Pseudo: mean vs last
plt.figure(figsize=(10,4))
plt.bar(x - 1.5*w, df["f1_mean_pseudo"], width=w, label="Pseudo mean")
plt.bar(x - 0.5*w, df["f1_last_pseudo"], width=w, label="Pseudo last")
plt.bar(x + 0.5*w, df["f1_mean_classic"], width=w, label="Classic mean")
plt.bar(x + 1.5*w, df["f1_last_classic"], width=w, label="Classic last")
plt.xticks(x, df["mask"])
plt.ylabel("F1 on missing pixels")
plt.title("Mean vs Last — Pseudo vs Classic (by mask)")
plt.grid(True, alpha=0.2)
plt.legend()
savefig(outdir, "bar_f1_mean_vs_last_pseudo_classic.png")
plt.show()
plt.close()

# gap plot: (mean - last)
plt.figure(figsize=(10,4))
plt.bar(x - 0.15, df["gap_pseudo"], width=0.3, label="Pseudo (mean-last)")
plt.bar(x + 0.15, df["gap_classic"], width=0.3, label="Classic (mean-last)")
plt.xticks(x, df["mask"])
plt.ylabel("F1(mean) - F1(last)")
plt.title("Gap mean-last (positive means averaging helps)")
plt.grid(True, alpha=0.2)
plt.legend()
savefig(outdir, "bar_gap_mean_minus_last.png")
plt.show()
plt.close()

print("DONE. OUT:", outdir)


# In[ ]:





# ### Etude hyper param warump_pg, thinning, burn_in

# In[ ]:


import os, json, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
mask_kind = "random50"     # "random50" / "top" / "bottom" / "center"
missing_rate = 0.5         # used only if random50

base = dict(
    n_iters=16000,
    burn_in=2000,
    thinning=20,
    eval_every=100,
    warmup_pg=50,
)

sweep_warmup   = [0, 10, 50, 100]          # MwG only
sweep_burn_in  = [0, 500, 2000, 5000]      # MwG only
sweep_thinning = [1, 5, 10, 20, 50]        # MwG only

# MwG options
use_adaptive_accept = False
target_accept = 0.15
adapt_lr = 0.005
freeze_adapt_after_burnin = True

# ---------------------------
# OUTDIR
# ---------------------------
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join("results", f"exp_mwg_hyper_overlay_{mask_kind}_{run_id}")
os.makedirs(outdir, exist_ok=True)
print("OUT:", outdir)

with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(dict(
        mask_kind=mask_kind, missing_rate=missing_rate,
        base=base,
        sweep_warmup=sweep_warmup, sweep_burn_in=sweep_burn_in, sweep_thinning=sweep_thinning,
        mwg=dict(
            use_adaptive_accept=use_adaptive_accept,
            target_accept=target_accept,
            adapt_lr=adapt_lr,
            freeze_adapt_after_burnin=freeze_adapt_after_burnin
        )
    ), f, indent=2)

# ---------------------------
# Mask / init FIXED
# ---------------------------
if mask_kind == "random50":
    mask = make_random_mask(x_true, missing_rate=missing_rate)
else:
    mask = make_mask(x_true, kind=mask_kind)

x_init = init_with_noise(x_true, mask)

# ---------------------------
# Core runner (MwG only)
# ---------------------------
def run_mwg_once(params):
    n_iters    = int(params["n_iters"])
    burn_in    = int(params["burn_in"])
    thinning   = int(params["thinning"])
    eval_every = int(params["eval_every"])
    warmup_pg  = int(params["warmup_pg"])

    assert n_iters > burn_in, f"n_iters must be > burn_in (got {n_iters} <= {burn_in})"
    assert thinning >= 1, "thinning must be >= 1"

    mean_m, summ_m, hist_m = run_chain_with_tracking(
        model, x_true, x_init, mask,
        method="mwg",
        n_iters=n_iters, burn_in=burn_in, thinning=thinning, eval_every=eval_every,
        warmup_pg=warmup_pg,
        adaptive_accept=use_adaptive_accept,
        target_accept=target_accept,
        adapt_lr=adapt_lr,
        freeze_adapt_after_burnin=freeze_adapt_after_burnin
    )
    return summ_m, hist_m

# ---------------------------
# Overlay plot helper
# ---------------------------
def overlay_plots(hists, title_prefix, save_prefix):
    """
    hists: dict label -> hist_m
    Each hist_m: {"steps":..., "f1":..., "logp":..., "acc":...}
    """
    # ---- F1 ----
    plt.figure(figsize=(9,4))
    for label, h in hists.items():
        plt.plot(h["steps"], h["f1"], "-o", markersize=3, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("F1 on missing pixels")
    plt.title(title_prefix + " — F1 evolution (MwG)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outdir, f"{save_prefix}_f1_overlay.png")
    plt.show()
    plt.close()

    # ---- logp ----
    plt.figure(figsize=(9,4))
    for label, h in hists.items():
        plt.plot(h["steps"], h["logp"], "-o", markersize=3, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("log p(x_miss_true | x_obs)  (MC est.)")
    plt.title(title_prefix + " — logp evolution (MwG)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outdir, f"{save_prefix}_logp_overlay.png")
    plt.show()
    plt.close()

    # ---- acceptance ----
    any_acc = any(("acc" in h and len(h["acc"]) == len(h["steps"])) for h in hists.values())
    if any_acc:
        plt.figure(figsize=(9,3))
        for label, h in hists.items():
            if "acc" in h and len(h["acc"]) == len(h["steps"]):
                plt.plot(h["steps"], h["acc"], "-o", markersize=3, label=label)
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance (avg over window)")
        plt.title(title_prefix + " — acceptance (MwG)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(outdir, f"{save_prefix}_acc_overlay.png")
        plt.show()
        plt.close()

# ---------------------------
# Sweep runner (generic)
# ---------------------------
def run_sweep(param_name, values):
    """
    Runs MwG for each value, stores:
      - overlay plots (F1/logp/acc)
      - CSV: per-value final scores
      - CSV: long format history (step, f1, logp, acc, label)
    """
    hists = {}
    rows_final = []
    rows_hist = []

    for v in values:
        params = dict(base)
        params[param_name] = int(v)

        label = f"{param_name}={v}"
        print("\n---", label, "---")

        summ, hist = run_mwg_once(params)

        # keep only what we need
        h = {"steps": hist["steps"], "f1": hist["f1"], "logp": hist["logp"]}
        if "acc" in hist:
            h["acc"] = hist["acc"]
        hists[label] = h

        # final scores
        rows_final.append(dict(
            param=param_name,
            value=float(v),
            n_iters=params["n_iters"],
            burn_in=params["burn_in"],
            thinning=params["thinning"],
            eval_every=params["eval_every"],
            warmup_pg=params["warmup_pg"],
            f1_final=float(summ["f1_final"]) if summ.get("f1_final") is not None else np.nan,
            logp_final=float(summ["logp_final"]) if summ.get("logp_final") is not None else np.nan,
            acc_final=float(summ.get("acc_final")) if summ.get("acc_final") is not None else np.nan,
        ))

        # long history
        for i, step in enumerate(h["steps"]):
            rows_hist.append(dict(
                label=label,
                param=param_name,
                value=float(v),
                step=int(step),
                f1=float(h["f1"][i]),
                logp=float(h["logp"][i]),
                acc=float(h["acc"][i]) if ("acc" in h and i < len(h["acc"])) else np.nan
            ))

    # save CSVs
    df_final = pd.DataFrame(rows_final).sort_values("value")
    df_hist = pd.DataFrame(rows_hist).sort_values(["value","step"])

    df_final.to_csv(os.path.join(outdir, f"summary_{param_name}.csv"), index=False)
    df_hist.to_csv(os.path.join(outdir, f"history_{param_name}_long.csv"), index=False)

    # overlay plots
    title = f"{mask_kind} | sweep {param_name} (MwG only)"
    overlay_plots(hists, title_prefix=title, save_prefix=f"sweep_{param_name}")

    # recap final (one point per value)
    plt.figure(figsize=(7,4))
    plt.plot(df_final["value"], df_final["f1_final"], "-o")
    plt.xlabel(param_name); plt.ylabel("F1 final")
    plt.title(f"{mask_kind} | final F1 vs {param_name} (MwG)")
    plt.grid(True, alpha=0.3)
    savefig(outdir, f"recap_final_f1_vs_{param_name}.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(df_final["value"], df_final["logp_final"], "-o")
    plt.xlabel(param_name); plt.ylabel("logp final")
    plt.title(f"{mask_kind} | final logp vs {param_name} (MwG)")
    plt.grid(True, alpha=0.3)
    savefig(outdir, f"recap_final_logp_vs_{param_name}.png")
    plt.show()
    plt.close()

    if df_final["acc_final"].notna().any():
        plt.figure(figsize=(7,4))
        plt.plot(df_final["value"], df_final["acc_final"], "-o")
        plt.xlabel(param_name); plt.ylabel("acc final")
        plt.title(f"{mask_kind} | final acceptance vs {param_name} (MwG)")
        plt.grid(True, alpha=0.3)
        savefig(outdir, f"recap_final_acc_vs_{param_name}.png")
        plt.show()
        plt.close()

    return df_final, df_hist

# ---------------------------
# RUN ALL 3 SWEEPS
# ---------------------------
df_wu_final, df_wu_hist = run_sweep("warmup_pg", sweep_warmup)
df_b_final,  df_b_hist  = run_sweep("burn_in",  sweep_burn_in)
df_th_final, df_th_hist = run_sweep("thinning", sweep_thinning)

print("DONE. OUT:", outdir)

