import time, math
import numpy as np
import torch

from sampler import (
    log_standard_normal_pdf, log_bernoulli_pdf, log_normal_pdf,
    pseudo_gibbs_step, metropolis_within_gibbs_step, metropolis_within_gibbs_step_mixture
)
from utils_chain import bernoulli_ll_missing, f1_missing


device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))


@torch.no_grad()
def run_chain_with_tracking(model, x_true, x_init, mask, *,
                            method="pseudo", n_iters=8000, burn_in=2000, thinning=20, eval_every=200,
                            warmup_pg=50, adaptive_accept=False, target_accept=0.20, adapt_lr=0.02,
                            freeze_adapt_after_burnin=True):
    """
    On lance une chaîne et on garde une moyenne postérieure (sur les probs).
    En plus on enregistre des points (F1/logp) tous les eval_every, pour tracer des courbes.
    """
    x = x_init.clone()

    # Pour MwG on a aussi une variable latente z, donc on l'initialise une fois.
    z = None
    log_rho = torch.tensor(0.0, device=x.device)  # rho = exp(log_rho)
    rho_track, acc_window = [], []

    if method == "mwg":
        # On chauffe un peu x avec du pseudo-gibbs avant de calculer z.
        for _ in range(warmup_pg):
            x_out, _ = pseudo_gibbs_step(model, x, mask)
            x = torch.bernoulli(x_out)

        # On part de z ~ q(z|x) pour que la chaîne démarre "proprement".
        mu, logvar = model.encode(x.view(-1, 784))
        z = model.reparameterize(mu, logvar)

    # sum_mean sert à accumuler les probs, logS sert au log-mean-exp du loglik sur les samples gardés.
    sum_mean, n_kept, logS = None, 0, None
    steps, f1_list, logp_list, acc_hist = [], [], [], []

    for t in range(n_iters):
        if method == "pseudo":
            # Ici on remplit directement les pixels manquants via le décodeur, sans MH sur z.
            x_out, _ = pseudo_gibbs_step(model, x, mask)
            x = torch.bernoulli(x_out)

        elif method == "mwg":
            # Ici on fait MH sur z, puis on re-sample x_miss avec le décodeur.
            adapt_now = adaptive_accept and (not freeze_adapt_after_burnin or t < burn_in)

            if adapt_now:
                x_out, z, acc, log_rho = metropolis_within_gibbs_step(
                    model, x, z, mask, return_accept=True, adaptive=True,
                    log_rho=log_rho, target_accept=target_accept, adapt_lr=adapt_lr
                )
                rho_track.append(float(torch.exp(log_rho).item()))
                if t % 200 == 0:
                    print("t", t, "acc", acc, "rho", float(torch.exp(log_rho).item()))
            else:
                rho = float(torch.exp(log_rho).item()) if adaptive_accept else 1.0
                x_out, z, acc = metropolis_within_gibbs_step(
                    model, x, z, mask, return_accept=True, adaptive=False, proposal_scale=rho
                )
                if adaptive_accept:
                    rho_track.append(rho)

            acc_window.append(acc)
            x = torch.bernoulli(x_out)

        else:
            raise ValueError("method must be 'pseudo' or 'mwg'")

        # On ne garde des samples qu'après burn-in, et on espace avec thinning.
        keep = (t >= burn_in) and ((t - burn_in) % thinning == 0)
        if not keep:
            continue

        probs = x_out
        sum_mean = probs.detach().clone() if sum_mean is None else (sum_mean + probs.detach())
        n_kept += 1
        mean_impute = sum_mean / n_kept

        # On met à jour une estimation MC de log p(x_miss_true | x_obs) par log-mean-exp.
        ll = bernoulli_ll_missing(x_true, probs, mask)
        if not torch.isfinite(ll).all():
            print("NON FINITE ll at t=", t, "check probs", probs.min().item(), probs.max().item())
        logS = ll.detach().clone() if logS is None else torch.logaddexp(logS, ll.detach())
        logp_batch = (logS - math.log(n_kept)).mean().item()

        # On enregistre juste quelques points pour faire des courbes lisibles.
        if (t % eval_every) == 0:
            steps.append(t)
            f1_list.append(f1_missing(x_true, mean_impute, mask))
            logp_list.append(logp_batch)
            if method == "mwg":
                acc_hist.append(float(np.mean(acc_window)) if len(acc_window) else float("nan"))
                acc_window = []

    # On renvoie la moyenne sur CPU (plus pratique pour les grids/plots).
    mean_impute_cpu = (sum_mean / max(n_kept, 1)).detach().cpu()

    summary = {"f1_final": (f1_list[-1] if len(f1_list) else None),
               "logp_final": (logp_list[-1] if len(logp_list) else None)}
    history = {"steps": steps, "f1": f1_list, "logp": logp_list}

    if method == "mwg":
        summary["acc_final"] = acc_hist[-1] if len(acc_hist) else None
        history["acc"] = acc_hist
        if adaptive_accept:
            summary["rho_final"] = rho_track[-1] if len(rho_track) else float(torch.exp(log_rho).item())
            history["rho"] = (rho_track[::max(1, len(rho_track)//max(1, len(steps)))] if len(steps) else rho_track)

    return mean_impute_cpu, summary, history


@torch.no_grad()
def run_mwg_multichain(model, x_true, x_init, mask, n_chains=5, **kwargs):
    """
    On lance plusieurs chaînes MwG indépendantes, puis on moyenne les reconstructions.
    Ça donne une idée de la stabilité (std entre chaînes).
    """
    chain_means, chain_summaries, chain_histories = [], [], []
    for _ in range(n_chains):
        mean_c, summ_c, hist_c = run_chain_with_tracking(model, x_true, x_init, mask, method="mwg", **kwargs)
        chain_means.append(mean_c)
        chain_summaries.append(summ_c)
        chain_histories.append(hist_c)

    chain_means = torch.stack(chain_means, dim=0)  # (C,B,1,28,28)
    mean_over, std_over = chain_means.mean(0), chain_means.std(0)

    # On récupère les infos finales de chaque chaîne, sans faire plus compliqué.
    accs = [s.get("acc_final", None) for s in chain_summaries]
    rhos = [s.get("rho_final", None) for s in chain_summaries]
    return mean_over, std_over, chain_summaries, chain_histories, accs, rhos


@torch.no_grad()
def run_chain_with_tracking_mixture(model, x_true, x_init, mask, *,
                                   n_iters=16000, burn_in=2000, thinning=20, eval_every=100,
                                   warmup_pg=50, alpha=0.5, rw_sigma=0.1):
    """
    Même format que run_chain_with_tracking, mais on remplace le pas MwG par un pas "mixture".
    """
    x = x_init.clone()

    # On chauffe x avant de calculer z, sinon ça démarre souvent n'importe comment.
    for _ in range(warmup_pg):
        x_out, _ = pseudo_gibbs_step(model, x, mask)
        x = torch.bernoulli(x_out)

    mu, logvar = model.encode(x.view(-1, 784))
    z = model.reparameterize(mu, logvar)

    sum_mean, n_kept, logS = None, 0, None
    steps, f1_list, logp_list, acc_list, acc_window = [], [], [], [], []

    for t in range(n_iters):
        # Ici le proposal sur z mélange indépendance q(z|x) et random-walk autour de z courant.
        x_out, z, acc = metropolis_within_gibbs_step_mixture(
            model, x, z, mask, alpha=alpha, rw_sigma=rw_sigma, return_accept=True
        )
        x = torch.bernoulli(x_out)
        acc_window.append(acc)

        keep = (t >= burn_in) and ((t - burn_in) % thinning == 0)
        if not keep:
            continue

        sum_mean = x_out.detach().clone() if sum_mean is None else (sum_mean + x_out.detach())
        n_kept += 1
        mean_impute = sum_mean / n_kept

        ll = bernoulli_ll_missing(x_true, x_out, mask)
        logS = ll.detach().clone() if logS is None else torch.logaddexp(logS, ll.detach())
        logp_batch = (logS - math.log(n_kept)).mean().item()

        if (t % eval_every) == 0:
            steps.append(t)
            f1_list.append(f1_missing(x_true, mean_impute, mask))
            logp_list.append(logp_batch)
            acc_list.append(float(np.mean(acc_window)) if len(acc_window) else float("nan"))
            acc_window = []

    mean_impute_cpu = (sum_mean / max(n_kept, 1)).detach().cpu()
    summary = {"f1_final": (f1_list[-1] if len(f1_list) else None),
               "logp_final": (logp_list[-1] if len(logp_list) else None),
               "acc_final": (acc_list[-1] if len(acc_list) else None),
               "n_kept": n_kept, "alpha": alpha, "rw_sigma": rw_sigma}
    history = {"steps": steps, "f1": f1_list, "logp": logp_list, "acc": acc_list}
    return mean_impute_cpu, summary, history


@torch.no_grad()
def run_chain_mean_last(model, x_true, x_init, mask_obs, *,
                        method="pseudo", n_iters=8000, burn_in=2000, thinning=20,
                        warmup_pg=50, adaptive=False, target_accept=0.15, adapt_lr=0.005,
                        freeze_adapt_after_burnin=True, proposal_scale=1.0):
    """
    On fait tourner une chaîne et on sort deux sorties simples.
    La moyenne postérieure (sur les probs) et le dernier sample binaire.
    """
    t0 = time.perf_counter()
    x = x_init.clone()
    z, log_rho, acc_window = None, torch.tensor(0.0, device=x.device), []

    if method == "mwg":
        # Même idée: on warmup x, puis on initialise z.
        for _ in range(warmup_pg):
            x_out, _ = pseudo_gibbs_step(model, x, mask_obs)
            x = torch.bernoulli(x_out)
        mu, logvar = model.encode(x.view(-1, 784))
        z = model.reparameterize(mu, logvar)

    sum_mean, n_kept, logS, last_sample_bin = None, 0, None, None

    for t in range(n_iters):
        if method == "pseudo":
            x_out, _ = pseudo_gibbs_step(model, x, mask_obs)
            x = torch.bernoulli(x_out)
        else:
            # Soit on adapte rho au début, soit on fixe proposal_scale.
            adapt_now = adaptive and (not freeze_adapt_after_burnin or t < burn_in)
            if adapt_now:
                x_out, z, acc, log_rho = metropolis_within_gibbs_step(
                    model, x, z, mask_obs, return_accept=True, adaptive=True,
                    log_rho=log_rho, target_accept=target_accept, adapt_lr=adapt_lr
                )
            else:
                rho = float(torch.exp(log_rho).item()) if adaptive else float(proposal_scale)
                x_out, z, acc = metropolis_within_gibbs_step(
                    model, x, z, mask_obs, return_accept=True, adaptive=False, proposal_scale=rho
                )
            acc_window.append(acc)
            x = torch.bernoulli(x_out)

        keep = (t >= burn_in) and ((t - burn_in) % thinning == 0)
        if not keep:
            continue

        sum_mean = x_out.detach().clone() if sum_mean is None else (sum_mean + x_out.detach())
        n_kept += 1

        ll = bernoulli_ll_missing(x_true, x_out, mask_obs)
        logS = ll.clone() if logS is None else torch.logaddexp(logS, ll)
        last_sample_bin = x.detach().cpu()

    mean_impute = (sum_mean / max(n_kept, 1)).detach().cpu()
    # Moyenne postérieure continue
    mean_impute = (mean_impute > 0.5).float()

    f1_mean = f1_missing(x_true, mean_impute.to(device), mask_obs)


    # On calcule les F1 sur la partie manquante, une fois pour la moyenne, une fois pour le dernier sample.
    #f1_mean = f1_missing(x_true, mean_impute.to(device), mask_obs)
    f1_last = f1_missing(x_true, last_sample_bin.to(device), mask_obs)

    logp_mc = float((logS - math.log(max(n_kept, 1))).mean().item()) if logS is not None else float("nan")
    acc_last = (float(np.mean(acc_window[-max(1, len(acc_window)//10):]))
                if (method == "mwg" and len(acc_window) > 0) else None)

    return mean_impute, last_sample_bin, f1_mean, f1_last, logp_mc, acc_last, n_kept, (time.perf_counter() - t0)