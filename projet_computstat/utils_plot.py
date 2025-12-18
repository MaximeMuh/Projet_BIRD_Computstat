import os
import matplotlib.pyplot as plt


def savefig(outdir, name, dpi=200):
    path = os.path.join(outdir, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print("saved:", path)


def plot_evolution_triplet(hist_p, hist_c, hist_m, title="", outdir=None, prefix="triplet", legends=None):
    """
    On trace F1 et logp pour trois méthodes, et l'acceptance si elle existe.
    Chaque hist_* contient "steps", "f1", "logp", et parfois "acc".
    """
    def _save(name):
        if outdir is None:
            return
        savefig(outdir, name)

    lab_p = "Pseudo" if legends is None else legends[0]
    lab_c = "Classic MwG" if legends is None else legends[1]
    lab_m = "Mixture MwG" if legends is None else legends[2]

    plt.figure(figsize=(9, 4))
    plt.plot(hist_p["steps"], hist_p["f1"], "--o", markersize=3, label=lab_p)
    plt.plot(hist_c["steps"], hist_c["f1"], "-o",  markersize=3, label=lab_c)
    plt.plot(hist_m["steps"], hist_m["f1"], "-o",  markersize=3, label=lab_m)
    plt.xlabel("Iteration")
    plt.ylabel("F1 on missing pixels")
    plt.title(title + "  F1 evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save(f"{prefix}_f1.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(hist_p["steps"], hist_p["logp"], "--o", markersize=3, label=lab_p)
    plt.plot(hist_c["steps"], hist_c["logp"], "-o",  markersize=3, label=lab_c)
    plt.plot(hist_m["steps"], hist_m["logp"], "-o",  markersize=3, label=lab_m)
    plt.xlabel("Iteration")
    plt.ylabel("log p(x_miss_true | x_obs)  (MC est.)")
    plt.title(title + "  log-likelihood evolution (higher is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save(f"{prefix}_logp.png")
    plt.show()
    plt.close()

    has_acc = ("acc" in hist_c) or ("acc" in hist_m)
    if has_acc:
        plt.figure(figsize=(9, 3))
        if "acc" in hist_c:
            plt.plot(hist_c["steps"], hist_c["acc"], "-o", markersize=3, label=lab_c + " acc")
        if "acc" in hist_m:
            plt.plot(hist_m["steps"], hist_m["acc"], "-o", markersize=3, label=lab_m + " acc")
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance (avg over window)")
        plt.title(title + "  acceptance")
        plt.grid(True, alpha=0.3)
        plt.legend()
        _save(f"{prefix}_acc.png")
        plt.show()
        plt.close()


def show_grid(idxs, x_true, x_init, imgA, imgB, title, outdir, name="grid.png", labelA="Algo A", labelB="Algo B"):
    """
    On affiche une grille en 4 colonnes: True, init masquée, méthode A, méthode B.
    """
    rows, cols = len(idxs), 4
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows))
    colnames = ["True", "Masked init", labelA, labelB]

    for i, idx in enumerate(idxs):
        imgs = [x_true[idx], x_init[idx], imgA[idx], imgB[idx]]
        for j in range(cols):
            ax = axes[i, j] if rows > 1 else axes[j]
            ax.imshow(imgs[j].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(colnames[j], fontsize=10, fontweight="bold")

    plt.suptitle(title)
    plt.tight_layout()
    savefig(outdir, name)
    plt.show()


def show_grid_triplet(idxs, x_true, x_init, x_a, x_b, x_c, title, outdir,
                      name="grid_triplet.png", labels=("A", "B", "C")):
    """
    On affiche une grille en 5 colonnes: True, init masquée, A, B, C.
    x_a/x_b/x_c peuvent être sur CPU ou GPU, on convertit en CPU pour afficher.
    """
    rows, cols = len(idxs), 5
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 2.6 * rows))
    colnames = ["True", "Masked init", labels[0], labels[1], labels[2]]

    for i, idx in enumerate(idxs):
        imgs = [x_true[idx], x_init[idx], x_a[idx], x_b[idx], x_c[idx]]
        for j in range(cols):
            ax = axes[i, j] if rows > 1 else axes[j]
            ax.imshow(imgs[j].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(colnames[j], fontsize=10, fontweight="bold")

    plt.suptitle(title)
    plt.tight_layout()
    savefig(outdir, name)
    plt.show()


def plot_evolution(hist, title="", outdir=None, prefix="evolution"):
    """
    On trace l'évolution de F1 et logp pour pseudo et MwG.
    On trace aussi l'acceptance si elle est fournie, et rho si on l'a tracké.
    """
    steps = hist["steps"]

    plt.figure(figsize=(9, 4))
    plt.plot(steps, hist["f1_pseudo"], "--o", markersize=3, label="Pseudo-Gibbs")
    plt.plot(steps, hist["f1_mwg"], "-o", markersize=3, label="MwG")
    plt.xlabel("Iteration")
    plt.ylabel("F1 on missing pixels")
    plt.title(title + "  F1 evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if outdir is not None:
        savefig(outdir, f"{prefix}_f1.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(steps, hist["logp_pseudo"], "--o", markersize=3, label="Pseudo-Gibbs")
    plt.plot(steps, hist["logp_mwg"], "-o", markersize=3, label="MwG")
    plt.xlabel("Iteration")
    plt.ylabel("log p(x_miss_true | x_obs)  (MC est.)")
    plt.title(title + "  log-likelihood evolution (higher is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if outdir is not None:
        savefig(outdir, f"{prefix}_logp.png")
    plt.show()
    plt.close()

    if "acc_mwg" in hist:
        plt.figure(figsize=(9, 3))
        plt.plot(steps, hist["acc_mwg"], "-o", markersize=3)
        plt.xlabel("Iteration")
        plt.ylabel("MwG acceptance (avg over window)")
        plt.title(title + "  MwG acceptance")
        plt.grid(True, alpha=0.3)
        if outdir is not None:
            savefig(outdir, f"{prefix}_acc.png")
        plt.show()
        plt.close()

    if "rho" in hist:
        plt.figure(figsize=(9, 3))
        plt.plot(steps, hist["rho"], "-o", markersize=3)
        plt.xlabel("Iteration")
        plt.ylabel("proposal_scale (rho)")
        plt.title(title + "  adaptive rho")
        plt.grid(True, alpha=0.3)
        if outdir is not None:
            savefig(outdir, f"{prefix}_rho.png")
        plt.show()
        plt.close()