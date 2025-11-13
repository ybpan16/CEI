# src/CEI/method.py

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from pathlib import Path

CHI2_1_median = 0.454936423119572

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def trimmed_median(x, prop=0.10):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan
    k = int(np.floor(prop * n))
    if 2*k >= n:
        return float(np.median(x))
    idx = np.argpartition(x, (k, n-k-1))
    mid = x[idx[k:n-k]]
    return float(np.median(mid))

def trimmed_median_vec(Z, prop=0.10):
    Z = np.asarray(Z, float)
    return np.array([trimmed_median(Z[:, j], prop) for j in range(Z.shape[1])], float)

# ----------------------------------------------------------------------
# Z-scores
# ----------------------------------------------------------------------
def z_qb(xB, xP, side="greater", return_details="all"):
    xB = np.asarray(xB, float)
    xP = np.asarray(xP, float)
    if xB.shape != xP.shape:
        raise ValueError("xB and xP must have the same length")

    SB = np.sum(xB)
    SP = np.sum(xP)

    fB = xB / SB
    fP = xP / SP
    d = fP - fB
    p_pool = (xB + xP) / (SB + SP)

    Var0 = p_pool * (1.0/SB + 1.0/SP)
    Z_qb = d / np.sqrt(np.maximum(Var0, 1e-30))

    if side == "greater":
        pvals = 1.0 - norm.cdf(Z_qb)
    elif side == "less":
        pvals = norm.cdf(Z_qb)
    elif side == "two-sided":
        pvals = 2.0 * (1.0 - norm.cdf(np.abs(Z_qb)))
    else:
        raise ValueError("side must be 'greater', 'less', or 'two-sided'")
    
    if return_details == 'all':
        return Z_qb, pvals, d
    elif return_details == 'Z_qb':
        return Z_qb
    elif return_details == 'pvals':
        return pvals
    elif return_details == 'd':
        return d
    else:
        raise ValueError("return_details must be 'all', 'Z_qb', 'pvals', or 'd'")
    
def z_r(xB, xP, tau=0.5, side="greater", return_details='all'):
    xB = np.asarray(xB, float)
    xP = np.asarray(xP, float)
    if xB.shape != xP.shape:
        raise ValueError("xB and xP must have the same length")
    
    SB = np.sum(xB)
    SP = np.sum(xP)
    
    pB_tilde = (xB + tau) / SB
    pP_tilde = (xP + tau) / SP

    r = np.log2(pP_tilde) - np.log2(pB_tilde)
    var = (1.0/(pP_tilde * SP)) + (1.0/(pB_tilde * SB))
    se = np.sqrt(var) / np.log(2.0)
    Z_r = r / se

    if side == "greater":
        pvals = 1.0 - norm.cdf(Z_r)
    elif side == "less":
        pvals = norm.cdf(Z_r)
    elif side == "two-sided":
        pvals = 2.0 * (1.0 - norm.cdf(np.abs(Z_r)))
    else:
        raise ValueError("side must be 'greater', 'less', or 'two-sided'")

    if return_details == 'all':
        return Z_r, pvals, r
    elif return_details == 'Z_r':
        return Z_r
    elif return_details == 'pvals':
        return pvals
    elif return_details == 'r':
        return r
    else:
        raise ValueError("return_details must be 'all', 'Z_r', 'pvals', or 'r'")

# ----------------------------------------------------------------------
# 2D method (parametric)
# ----------------------------------------------------------------------
def bivnorm_pvals(z1, z2, Sigma, eps=1e-12):
    """Two-sided (elliptical) p-values via χ²₂ tail using m² = zᵀΣ⁻¹z."""
    z = np.stack([z1, z2], axis=-1)
    Sinv = np.linalg.inv(Sigma + eps*np.eye(2))
    m2 = np.einsum('...i,ij,...j->...', z, Sinv, z)
    p = 1.0 - chi2.cdf(m2, df=2)
    return p, m2

def pvals_from_data(zds, zrs, use_unit_var=False):
    zds = np.asarray(zds); zrs = np.asarray(zrs)
    if use_unit_var:
        rho = np.corrcoef(zds, zrs)[0,1]
        denom = 1 - rho**2
        m2 = (zds**2 - 2*rho*zds*zrs + zrs**2) / denom
        p = 1.0 - chi2.cdf(m2, df=2)
        return p, m2, rho
    else:
        Sigma = np.cov(np.vstack([zds, zrs]), bias=False)
        p, m2 = bivnorm_pvals(zds, zrs, Sigma)
        return p, m2, Sigma

# ----------------------------------------------------------------------
# Joint Mahalanobis + permutation
# ----------------------------------------------------------------------
def mahalanobis_distance(Z, mu, target_kappa=500.0, min_ridge=1e-8):
    X = Z - mu
    n = Z.shape[0]
    Sigma = (X.T @ X) / max(n - 1, 1)

    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 0.0, None)
    lam_max, lam_min = float(np.max(w)), float(np.min(w))

    if lam_min <= 0:
        delta = max(min_ridge, lam_max/target_kappa - lam_min + 1e-12)
    else:
        delta = max(min_ridge, max(0.0, lam_max/target_kappa - lam_min))

    w, V = np.linalg.eigh(Sigma + delta * np.eye(Sigma.shape[0]))
    w = np.clip(w, 1e-12, None)
    Winvhalf = V @ (np.diag(1.0 / np.sqrt(w))) @ V.T

    D = X @ Winvhalf.T
    D2 = np.sum(D * D, axis=1)
    return D2

def mahalanobis_pvals(Z, mu, side='greater'):
    D2 = mahalanobis_distance(Z, mu)
    if side == "greater":
        p = 1.0 - chi2.cdf(D2, df=2)
    elif side == "less":
        p = chi2.cdf(D2, df=2)
    elif side == "two-sided":
        p = 2.0 * (1.0 - chi2.cdf(np.abs(D2), df=2))
    else:
        raise ValueError("side must be 'greater', 'less', or 'two-sided'")
    return np.sqrt(D2), p

# Permutation p-values for 2D Mahalanobis
def pvals_2d_permutation(xB, xP, D2_obs, R=200, rng=None):
    xB = np.asarray(xB, int); xP = np.asarray(xP, int)
    n  = xB.size
    if rng is None:
        rng = np.random.default_rng()

    SB, SP = int(xB.sum()), int(xP.sum())
    f_pool = (xB + xP) / float(SB + SP)

    null_pool = np.empty(n * R, float)
    write = 0

    for _ in range(R):
        xB0 = rng.multinomial(SB, f_pool)
        xP0 = rng.multinomial(SP, f_pool)

        Z1, _, _ = z_qb(xB0, xP0)
        Z2, _, _ = z_r(xB0, xP0)
        Z = np.column_stack([Z1, Z2])
        mu = trimmed_median_vec(Z, prop=0.01)

        D2 = mahalanobis_distance(Z, mu)
        null_pool[write:write+n] = D2
        write += n

    null_pool.sort()
    N0 = null_pool.size

    idx = np.searchsorted(null_pool, D2_obs, side='left')
    ge  = N0 - idx

    p_emp = (1.0 + ge) / (1.0 + N0)
    return np.clip(p_emp, 0.0, 1.0)


# ----------------------------------------------------------------------
# ACAT
# ----------------------------------------------------------------------
def acat(pvals, weights=[0.5, 0.5], clip=1e-15):
    P = np.asarray(pvals, float)
    if P.ndim == 1:
        P = P[None, :]
    n, K = P.shape

    if weights is None:
        w = np.ones(K, float)
    else:
        w = np.asarray(weights, float)
        if w.shape != (K,) or np.any(w < 0) or np.any(np.isnan(w)):
            raise ValueError("weights must be nonnegative, shape (K,), no NaNs")

    W = np.broadcast_to(w, (n, K)).copy()
    nan_mask = ~np.isfinite(P)
    W[nan_mask] = 0.0

    row_wsum = W.sum(axis=1)
    p_acat = np.ones(n, float)
    good = row_wsum > 0
    if not np.any(good):
        return p_acat

    W[good] /= row_wsum[good, None]
    P = np.clip(P, clip, 1 - clip)

    T = np.sum(W * np.tan((0.5 - P) * np.pi), axis=1)
    p_acat[good] = 0.5 - np.arctan(T[good]) / np.pi
    return np.clip(p_acat, 0.0, 1.0)

# ----------------------------------------------------------------------
# FDR control
# ----------------------------------------------------------------------
def fdr(pvals, alpha=0.1, method='bh', weights=None):
    p = np.asarray(pvals, float)
    if p.ndim != 1:
        raise ValueError("pvals must be 1-D")
    m_total = p.size
    keep = ~np.isnan(p)
    p_clean = p[keep]
    m = p_clean.size
    if m == 0:
        return np.full(m_total, np.nan), np.zeros(m_total, bool), np.nan

    if weights is not None:
        w = np.asarray(weights, float)
        if w.shape != p.shape or np.any(np.isnan(w)) or np.any(w < 0):
            raise ValueError("weights must be nonnegative, same shape as pvals, no NaNs")
        w = w[keep]
        w = w * (m / np.sum(w))
        p_eff = p_clean / w
    else:
        p_eff = p_clean

    order = np.argsort(p_eff, kind='mergesort')
    p_sorted = p_eff[order]
    ranks = np.arange(1, m+1)

    if method == 'by':
        Hm = np.sum(1.0 / ranks)
        alpha_prime = alpha / Hm
    elif method == 'bh':
        alpha_prime = alpha
    else:
        raise ValueError("method must be 'bh' or 'by'")

    crit = ranks * (alpha_prime / m)
    le = p_sorted <= crit
    if np.any(le):
        k = np.max(np.nonzero(le)[0]) + 1
        thresh = p_sorted[k-1]
    else:
        k = 0
        thresh = np.nan

    q_sorted = (m / ranks) * p_sorted
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    if method == 'by':
        q_sorted *= np.sum(1.0 / ranks)
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    inv = np.empty_like(order); inv[order] = np.arange(m)
    qvals_nonan = q_sorted[inv]

    qvals = np.full(m_total, np.nan, float); qvals[keep] = qvals_nonan

    reject = np.zeros(m_total, bool)
    if k > 0:
        rej_nonan = np.zeros(m, bool)
        rej_nonan[order[:k]] = True
        reject[keep] = rej_nonan

    return qvals, reject, thresh

# ----------------------------------------------------------------------
# High-level analysis function
# ----------------------------------------------------------------------
def run_analysis(
    input_file,
    Bcol=None,
    Pcol=None,
    alpha_2d=0.10,
    alpha_joint=0.10,
    alpha_acat=0.10,
    fdr_method="bh",
    R_perm=400,
    out_2d=None,
    out_joint=None,
    out_acat=None,
):
    """
    Run the full analysis pipeline on one Excel file.
    """
    input_path = Path(input_file)

    data = pd.read_excel(input_path)
    index = data['index'].to_numpy()
    xB = data[Bcol].to_numpy()
    xP = data[Pcol].to_numpy()

    Z_qb, p_qb, d = z_qb(xB, xP)
    Z_r,  p_r,  r = z_r(xB, xP)

    p_2d, _, _ = pvals_from_data(Z_qb, Z_r)

    Z = np.column_stack([Z_qb, Z_r])
    mu = trimmed_median_vec(Z, prop=0.01)
    D2 = mahalanobis_distance(Z, mu)
    Di = np.sqrt(D2)
    p_joint = pvals_2d_permutation(xB, xP, D2, R=R_perm, rng=None)

    p_acat = acat(np.column_stack([p_qb, p_r]), weights=[0.5, 0.5])

    # FDR
    q_2d,    rej_2d,    thre_2d    = fdr(p_2d,    alpha=alpha_2d,    method=fdr_method)
    q_joint, rej_joint, thre_joint = fdr(p_joint, alpha=alpha_joint, method=fdr_method)
    q_acat,  rej_acat,  thre_acat  = fdr(p_acat,  alpha=alpha_acat,  method=fdr_method)

    hits_2d    = rej_2d    & (d > 0)
    hits_joint = rej_joint & (d > 0)
    hits_acat  = rej_acat  & (d > 0)

    out = pd.DataFrame({
        'index': index, 'xB': xB, 'xP': xP,
        'd': d, 'Z_qb': Z_qb, 'p_qb': p_qb,
        'r': r, 'Z_r': Z_r, 'p_r': p_r,
        'p_2d': p_2d, 'q_2d': q_2d, 'hit_2d': hits_2d,
        'Di': Di, 'p_joint': p_joint, 'q_joint': q_joint, 'hit_joint': hits_joint,
        'p_acat': p_acat, 'q_acat': q_acat, 'hit_acat': hits_acat,
    })

    if out_2d is not None:
        calls_2d = out.loc[hits_2d].sort_values('q_2d')
        calls_2d.to_excel(out_2d, index=False)
    if out_joint:
        calls_joint = out.loc[hits_joint].sort_values('q_joint')
        calls_joint.to_excel(out_joint, index=False)
    if out_acat:
        calls_acat  = out.loc[hits_acat].sort_values('q_acat')
        calls_acat.to_excel(out_acat, index=False)

    return "Completed analysis."