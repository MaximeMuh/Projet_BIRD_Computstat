import math
import numpy as np
import torch
import torch.nn.functional as F


def pseudo_gibbs_step(model, x_curr, mask):
    """
    On prend x courant (complété) et on fait 1 "pseudo-Gibbs" :
    1) on encode -> z ~ q(z|x)
    2) on decode -> probs = p(x|z)
    3) on remplace juste les pixels manquants avec ces probs

    x_next = x_obs + x_recon sur les missing
    """
    with torch.no_grad():
        mu, logvar = model.encode(x_curr.view(-1, 784))
        z = model.reparameterize(mu, logvar)
        x_recon = model.decode(z).view_as(x_curr)

    x_next = x_curr * mask + x_recon * (1 - mask)
    return x_next, z


def log_normal_pdf(z, mu, logvar):
    """
    log N(z ; mu, diag(exp(logvar)))  (batch)
    """
    const = -0.5 * torch.log(2 * torch.tensor(np.pi, device=z.device)) - 0.5 * logvar
    dist  = -0.5 * (z - mu) ** 2 / torch.exp(logvar)
    return torch.sum(const + dist, dim=1)


def log_standard_normal_pdf(z):
    """
    log N(z ; 0, I)
    """
    return log_normal_pdf(z, torch.zeros_like(z), torch.zeros_like(z))


def log_bernoulli_pdf(x, probs, mask=None):
    """
    log p(x | probs) pour des Bernoulli indépendantes.
    On l'écrit avec la BCE :
      log p = - BCE(probs, x)
    Si mask est donné, on ne compte que les pixels observés (mask=1).
    """
    bce = F.binary_cross_entropy(probs, x, reduction="none")
    if mask is not None:
        bce = bce * mask.view(x.size(0), -1)
    return -torch.sum(bce, dim=1)


@torch.no_grad()
def metropolis_within_gibbs_step(model, x_curr, z_curr, mask, return_accept=False,
                                proposal_scale=1.0,
                                adaptive=False, log_rho=None, adapt_lr=0.005, target_accept=0.23,
                                adapt_clamp=(math.log(0.25), math.log(4.0))):
    """
    Un pas MwG classique (MH sur z, puis Gibbs sur x_miss).

    Target (à une constante près) :
      pi(z) ∝ p(x_obs | z) p(z)

    Proposal :
      q_rho(z'|x) = N(mu_q(x), rho^2 * diag(exp(logvar_q(x))))

    Ratio MH :
      a = min(1,  pi(z') q(z|x) / (pi(z) q(z'|x))  )
    """
    mu_q, logvar_q = model.encode(x_curr.view(-1, 784))

    if adaptive:
        if log_rho is None:
            log_rho = torch.tensor(0.0, device=mu_q.device)
        rho = torch.exp(log_rho).item()
    else:
        rho = float(proposal_scale)

    eps = torch.randn_like(mu_q)
    z_prop = mu_q + rho * torch.exp(0.5 * logvar_q) * eps

    log_p_z_prop = log_standard_normal_pdf(z_prop)
    log_p_z_curr = log_standard_normal_pdf(z_curr)

    x_recon_prop = model.decode(z_prop)
    x_recon_curr = model.decode(z_curr)

    log_p_x_prop = log_bernoulli_pdf(x_curr.view(-1, 784), x_recon_prop, mask)
    log_p_x_curr = log_bernoulli_pdf(x_curr.view(-1, 784), x_recon_curr, mask)

    # q_rho a la même moyenne mu_q, juste une variance gonflée par rho^2
    logvar_q_rho = logvar_q + 2.0 * torch.log(torch.tensor(rho, device=mu_q.device))
    log_q_z_prop = log_normal_pdf(z_prop, mu_q, logvar_q_rho)
    log_q_z_curr = log_normal_pdf(z_curr, mu_q, logvar_q_rho)

    log_alpha = (log_p_x_prop + log_p_z_prop + log_q_z_curr) - (log_p_x_curr + log_p_z_curr + log_q_z_prop)
    alpha = torch.exp(torch.clamp(log_alpha, max=0.0))

    u = torch.rand_like(alpha)
    accept = (u < alpha)

    z_next = torch.where(accept.unsqueeze(1), z_prop, z_curr)

    # Gibbs : on remplace x_miss avec le décodeur (en probas), on garde x_obs intact
    x_recon_next = model.decode(z_next).view_as(x_curr)
    x_next = x_curr * mask + x_recon_next * (1 - mask)

    acc_rate = accept.float().mean()

    # adaptation simple de rho pour viser une acceptation cible
    if adaptive:
        log_rho = log_rho + adapt_lr * (target_accept - acc_rate)
        lo, hi = adapt_clamp
        log_rho = torch.clamp(log_rho, lo, hi)

    if return_accept:
        return (x_next, z_next, acc_rate.item(), log_rho) if adaptive else (x_next, z_next, acc_rate.item())
    return (x_next, z_next, log_rho) if adaptive else (x_next, z_next)


@torch.no_grad()
def metropolis_within_gibbs_step_mixture(model, x_curr, z_curr, mask, *,
                                        alpha=0.5, rw_sigma=0.5, return_accept=True):
    """
    Même target que MwG, mais proposal = mélange :

      avec proba alpha    : z' ~ q(z|x)            (independence)
      avec proba 1-alpha  : z' = z + sigma * eps   (random-walk)

    Donc :
      q(z'|z) = alpha q_indep(z'|x) + (1-alpha) q_rw(z'|z)

    Et MH :
      a = min(1,  pi(z') q(z|z') / (pi(z) q(z'|z))  )
    """
    mu_curr, logvar_curr = model.encode(x_curr.view(-1, 784))

    B, D = z_curr.shape
    u = torch.rand(B, device=z_curr.device)
    is_indep = (u < alpha).float().unsqueeze(1)

    z_indep = model.reparameterize(mu_curr, logvar_curr)
    z_rw = z_curr + rw_sigma * torch.randn_like(z_curr)
    z_prop = is_indep * z_indep + (1 - is_indep) * z_rw

    log_p_z_prop = log_standard_normal_pdf(z_prop)
    log_p_z_curr = log_standard_normal_pdf(z_curr)

    x_recon_prop = model.decode(z_prop)
    x_recon_curr = model.decode(z_curr)

    log_p_x_prop = log_bernoulli_pdf(x_curr.view(-1, 784), x_recon_prop, mask)
    log_p_x_curr = log_bernoulli_pdf(x_curr.view(-1, 784), x_recon_curr, mask)

    log_target_prop = log_p_x_prop + log_p_z_prop
    log_target_curr = log_p_x_curr + log_p_z_curr

    # densités des deux briques
    log_q_indep_prop = log_normal_pdf(z_prop, mu_curr, logvar_curr)
    log_q_indep_curr = log_normal_pdf(z_curr, mu_curr, logvar_curr)

    logvar_rw = torch.full_like(z_curr, 2 * np.log(rw_sigma + 1e-12))
    log_q_rw_prop = log_normal_pdf(z_prop, z_curr, logvar_rw)
    log_q_rw_curr = log_normal_pdf(z_curr, z_prop, logvar_rw)

    # log q(z'|z) via logsumexp : log( alpha*... + (1-alpha)*... )
    log_a = torch.log(torch.tensor(alpha + 1e-12, device=z_curr.device))
    log_1a = torch.log(torch.tensor(1 - alpha + 1e-12, device=z_curr.device))

    log_q_forward  = torch.logaddexp(log_a + log_q_indep_prop, log_1a + log_q_rw_prop)
    log_q_backward = torch.logaddexp(log_a + log_q_indep_curr, log_1a + log_q_rw_curr)

    log_alpha_mh = (log_target_prop + log_q_backward) - (log_target_curr + log_q_forward)
    alpha_mh = torch.exp(torch.clamp(log_alpha_mh, max=0.0))

    u2 = torch.rand_like(alpha_mh)
    accept = (u2 < alpha_mh).float().unsqueeze(1)

    z_next = accept * z_prop + (1 - accept) * z_curr

    # même fin que MwG : on reconstruit x_miss avec le décodeur
    x_recon_next = model.decode(z_next).view_as(x_curr)
    x_next = x_curr * mask + x_recon_next * (1 - mask)

    if return_accept:
        return x_next, z_next, accept.mean().item()
    return x_next, z_next