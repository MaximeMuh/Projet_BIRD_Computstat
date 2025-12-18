import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score


def bernoulli_ll_missing(x_true, probs, mask_obs, eps=1e-6):
    """
    On calcule log p(x_miss_true | probs) uniquement sur les pixels manquants.
    """
    B = x_true.size(0)

    # On met tout à plat : (B, 784) et on passe en double sur CPU
    x = x_true.detach().cpu().view(B, -1).double()
    p = probs.detach().cpu().view(B, -1).double()

    # On nettoie d'abord p (NaN/inf), puis on clamp pour avoir eps <= p <= 1-eps
    p = torch.nan_to_num(p, nan=0.5, posinf=1.0 - eps, neginf=eps).clamp(eps, 1 - eps)

    # On récupère le masque des pixels manquants (1 = manquant)
    miss = (1 - mask_obs.detach().cpu().view(B, -1)).double()

    # BCE pixel par pixel, puis on prend l'opposé pour avoir un log-likelihood
    bce = F.binary_cross_entropy(p, x, reduction="none")   
    ll_pix = -bce * miss
    ll_pix = torch.nan_to_num(ll_pix, nan=-1e6, posinf=-1e6, neginf=-1e6)

    # On somme sur les pixels => un score par image
    return ll_pix.sum(dim=1)  # (B,)


def f1_missing(x_true, x_hat, mask_obs):
    """
    On calcule le F1 uniquement sur les pixels manquants.
    On binarise x_hat avec un seuil à 0.5.
    """
    miss = (1 - mask_obs).bool().cpu().numpy().reshape(-1)

    yt = x_true.detach().cpu().numpy().reshape(-1)[miss]
    yp = (x_hat.detach().cpu().numpy().reshape(-1)[miss] > 0.5).astype(float)

    return f1_score(yt, yp)