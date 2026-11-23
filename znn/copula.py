import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from .data import DEFAULT_FEATURE_CONFIG, catalog_to_XY
from .models import _location_scale, latent
from .qp_output import DEFAULT_ZMAX, _mode, pz_cdf, pz_ppf

__all__ = ["fit_copula", "sample_redshifts", "DEFAULT_NZ_BINS"]

DEFAULT_NZ_BINS = np.linspace(0.0, DEFAULT_ZMAX, 61)  # the n(z) bins: 60 bins of width 0.05 on [0, 3]
_RHO_MAX = 0.99
_PIT_CLIP = 1e-9  # PITs are clipped to [_PIT_CLIP, 1 - _PIT_CLIP] before the normal score
_EIG_FLOOR = 1e-12

# A fitted copula is a plain dict:
#   rho        the free parameter of Eq. neuron-corr (0: independent sampling)
#   active     boolean mask of the latent units used (those active on the training set)
#   mean       <xi>, the training-set mean of the active units
#   whitening  Lambda^-1/2 V^T, stored to act on the right: w = (xi - <xi>) @ whitening
#   bins       edges of the n(z) bins rho was fitted on
#   n_fit      number of objects rho was fitted on (outliers excluded)
#   n_bins     number of n(z) bins with at least one of them
#   pit_std    standard deviation of the u_j of the fit sample (divided out before the fit)

def _pca_basis(xi):
    """Units active on the training set, their mean <xi> and Lambda^-1/2 V^T (Eq. pca), from training latents xi."""
    xi = np.asarray(xi, dtype=np.float64)
    active = np.any(xi != 0, axis=0)
    x = xi[:, active]
    lam, V = np.linalg.eigh(np.atleast_2d(np.cov(x, rowvar=False)))
    lam = np.maximum(lam, _EIG_FLOOR * lam.max())
    return dict(active=active, mean=x.mean(axis=0), whitening=V / np.sqrt(lam))


def _pca_projection(copula, xi):
    """w_j = Lambda^-1/2 V^T (xi_j - <xi>) (Eq. pca), shape (N, n_active)."""
    return (np.asarray(xi, dtype=np.float64)[:, copula["active"]] - copula["mean"]) @ copula["whitening"]


def _unit(w):
    """w_j / |w_j|, with a row of zeros for w_j = 0."""
    norm = np.linalg.norm(w, axis=1, keepdims=True)
    return np.divide(w, norm, out=np.zeros_like(w), where=norm > 0)


def _cos_theta(w1, w2):
    """cos(theta_jj') between the rows of w1 and of w2 (Eq. similarity), shape (N1, N2)."""
    return _unit(np.atleast_2d(w1)) @ _unit(np.atleast_2d(w2)).T


def _sample_g(w, a, eps, rho):
    """
    g_j = sqrt(rho) w_j . a / |w_j| + sqrt(1 - rho) eps_j (Eq. neuron-copula), for the columns (draws) of a (n_active,
    n_draws) and eps (N, n_draws). An object with w_j = 0 gets g_j = eps_j, so that every g_j has unit variance.
    """
    d = _unit(w)
    shared = rho * np.sum(d**2, axis=1, keepdims=True)  # rho, or 0 for w_j = 0
    return np.sqrt(rho) * (d @ a) + np.sqrt(1 - shared) * eps


def _redshifts_from_g(g, mu, scale, nu, zmax=DEFAULT_ZMAX):
    """z_j = CDF_j^-1(Phi(g_j)) (Eq. copula), for g of shape (N, n_draws) and p(z) parameters mu, scale (N,)."""
    return pz_ppf(stats.norm.cdf(g), mu[:, None], scale[:, None], nu, zmax)


def _pit_scores(z_ref, mu, scale, nu, zmax=DEFAULT_ZMAX):
    """u_j = Phi^-1(CDF_j(z_j)) (Eq. pit-score)."""
    return stats.norm.ppf(np.clip(pz_cdf(z_ref, mu, scale, nu, zmax), _PIT_CLIP, 1 - _PIT_CLIP))


def _nz_bin_index(zmode, bins):
    """Index of the n(z) bin of each object, from the mode of its p(z) (a mode on the last edge goes in the last bin)."""
    return np.clip(np.searchsorted(bins, zmode, side="right") - 1, 0, len(bins) - 2)


def _bin_pit_means(u, q, n_bins):
    """u_q = M_q^-1 sum_{j in q} u_j (Eq. pit-bin) and M_q, for all n_bins bins (u_q = 0 for an empty bin)."""
    M_q = np.bincount(q, minlength=n_bins).astype(np.float64)
    u_q = np.bincount(q, weights=u, minlength=n_bins) / np.maximum(M_q, 1)
    return u_q, M_q


def _bin_mean_directions(w, q, n_bins):
    """wbar_q = M_q^-1 sum_{j in q} w_j / |w_j|, shape (n_bins, n_active) (zero rows for empty bins)."""
    d = _unit(w)
    wbar = np.zeros((n_bins, d.shape[1]))
    np.add.at(wbar, q, d)
    M_q = np.bincount(q, minlength=n_bins).astype(np.float64)
    return wbar / np.maximum(M_q, 1)[:, None]


def _copula_bin_covariance(rho, M_q, wbar_q):
    """
    Sigma_qq' = rho wbar_q . wbar_q' + delta_qq' (1 - rho) / M_q (Eq. neuron-copula_var): the covariance of the mean PIT scores
    of the bins predicted by the copula. Its diagonal is Var(g_q) = 1 / M_q + rho S_q, with
    S_q = M_q^-2 sum_{j != j' in q} cos(theta_jj').
    """
    shared = np.sqrt(rho) * wbar_q
    return shared @ shared.T + np.diag((1 - rho) / M_q)


def _neg_log_likelihood(rho, u_q, M_q, wbar_q):
    """-log N(u | 0, Sigma(rho)), up to a constant."""
    L = np.linalg.cholesky(_copula_bin_covariance(rho, M_q, wbar_q))
    y = np.linalg.solve(L, u_q)
    return float(np.sum(np.log(np.diag(L))) + 0.5 * y @ y)


def _fit_rho(u_q, M_q, wbar_q, min_count=2):
    """
    rho maximising the Gaussian likelihood of the measured mean PIT scores of the bins, u ~ N(0, Sigma(rho))
    (Eq. copula_fit), over the bins with at least min_count objects.
    """
    keep = M_q >= min_count
    if not keep.any():
        return 0.0
    u_q, M_q, wbar_q = u_q[keep], M_q[keep], wbar_q[keep]
    res = minimize_scalar(_neg_log_likelihood, bounds=(0.0, _RHO_MAX), args=(u_q, M_q, wbar_q), method="bounded",
                          options=dict(xatol=1e-8))
    return float(res.x)


def _fit_copula(xi, mu, scale, nu, z_ref, outlier=4.0, bins=DEFAULT_NZ_BINS, zmax=DEFAULT_ZMAX):
    """fit_copula on arrays: training latents xi, p(z) parameters mu, scale, nu and reference redshifts z_ref."""
    xi = np.asarray(xi, dtype=np.float64)
    mu, scale, bins = np.asarray(mu), np.asarray(scale), np.asarray(bins, dtype=np.float64)
    copula = _pca_basis(xi)
    u = _pit_scores(z_ref, mu, scale, nu, zmax)
    keep = np.isfinite(z_ref) & np.isfinite(u) & (np.abs(u) < outlier)
    pit_std = float(np.std(u[keep]))
    q = _nz_bin_index(_mode(mu[keep], scale[keep], nu, zmax), bins)
    n_bins = len(bins) - 1
    u_q, M_q = _bin_pit_means(u[keep] / pit_std, q, n_bins)
    wbar_q = _bin_mean_directions(_pca_projection(copula, xi[keep]), q, n_bins)
    copula.update(rho=_fit_rho(u_q, M_q, wbar_q), bins=bins, n_fit=int(keep.sum()), n_bins=int(np.sum(M_q > 0)),
                  pit_std=pit_std)
    return copula


def fit_copula(model, df, bands, ref_band, config=DEFAULT_FEATURE_CONFIG, filter_curves=None, Y=None, outlier=4.0,
               bins=DEFAULT_NZ_BINS, zmax=DEFAULT_ZMAX):
    """
    Fit the copula of a trained model (F, H, nu) on its training set: the same labelled catalogue (and reference
    redshifts Y) that train_model was given (see the top of the module).

    df: the catalogue (see znn.io.read_catalog); bands, ref_band, config, filter_curves as for catalog_to_XY (the
        model's feature settings; load_model_file returns them as **features).
    Y: reference redshifts, in the order of df; by default its "redshift" column.
    outlier: objects whose PIT normal score exceeds this in absolute value are left out of the fit.
    bins: edges of the n(z) bins the objects are grouped in, by the mode of their p(z).

    Returns the copula as a dict (see the top of the module).
    """
    X, Y_catalog = catalog_to_XY(df, bands, ref_band, config, filter_curves=filter_curves)
    z_ref = Y_catalog if Y is None else Y
    if z_ref is None:
        raise ValueError("fit_copula needs reference redshifts: a 'redshift' column in df, or Y")
    z_ref = np.asarray(z_ref, dtype=np.float64)
    mu, scale = _location_scale(model, X)
    return _fit_copula(latent(model, X), mu, scale, model[2], z_ref, outlier, bins, zmax)


def _sample(copula, xi, mu, scale, nu, n_draws, rng, rho=None, chunk=50_000, zmax=DEFAULT_ZMAX):
    """sample_redshifts on arrays: (n_draws, N) draws, float32."""
    rho = copula["rho"] if rho is None else float(rho)
    n = len(mu)
    a = rng.standard_normal((int(np.sum(copula["active"])), n_draws))  # common to all objects
    out = np.empty((n_draws, n), dtype=np.float32)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        # (objects, draws) layout: consecutive chunks of eps are the rows of one (N, n_draws) draw, whatever the chunk
        eps = rng.standard_normal((sl.stop - sl.start, n_draws))
        g = _sample_g(_pca_projection(copula, xi[sl]), a, eps, rho)
        out[:, sl] = _redshifts_from_g(g, mu[sl], scale[sl], nu, zmax).T
    return out


def sample_redshifts(
    model,
    copula,
    df,
    bands,
    ref_band,
    config=DEFAULT_FEATURE_CONFIG,
    filter_curves=None,
    n_draws=100,
    rng=None,
    rho=None,
    chunk=50_000,
    zmax=DEFAULT_ZMAX,
):
    """
    Joint redshift draws for a catalogue from a trained model (F, H, nu) and its copula (the dict fit_copula returns):
    each object's redshift is marginally distributed exactly as its p(z) (the Student-t of predict), and objects that
    are similar in the latent space of F are correlated (see the top of the module). Histogramming one draw of all
    objects gives one n(z) realization.

    df, bands, ref_band, config, filter_curves: the catalogue and the model's feature settings, as for predict.
    n_draws: number of joint draws. rng: a numpy Generator (or a seed); the draws do not depend on chunk.
    rho: overrides the copula's rho (0 gives independent sampling from the same p(z)s).
    chunk: objects at a time, to bound memory with large catalogues.

    Returns an array (n_draws, N) of redshifts, float32, in the order of df.
    """
    rng = np.random.default_rng(rng)
    X, _ = catalog_to_XY(df, bands, ref_band, config, filter_curves=filter_curves)
    mu, scale = _location_scale(model, X)
    return _sample(copula, latent(model, X), mu, scale, model[2], n_draws, rng, rho, chunk, zmax)
