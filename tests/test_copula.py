"""
Tests of znn.copula: correlated redshift draws that leave every object's p(z) unchanged, and the fit of rho. Each
array-level function is tested against the equation of the write-up it implements, on synthetic latent vectors; the
catalogue-level fit_copula and sample_redshifts end to end on a small untrained model.
"""

import numpy as np
import pytest
import tensorflow as tf
from scipy import stats
from tensorflow.keras import models

from scipy.optimize import minimize_scalar

from znn.copula import (
    _bin_mean_directions,
    _bin_pit_means,
    _copula_bin_covariance,
    _cos_theta,
    _fit_copula,
    _fit_rho,
    _neg_log_likelihood,
    _pca_basis,
    _pca_projection,
    _sample,
    _sample_g,
    fit_copula,
    sample_redshifts,
)
from znn.models import ToRedshift, build_model
from znn.qp_output import DEFAULT_ZMAX, _ppf, pz_cdf, pz_ppf

NU = 5.0
# (median z, scale in log(1 + z)): narrow and wide, mid-range and close to each edge, so truncation matters
CASES = [(0.05, 0.02), (0.5, 0.05), (1.0, 0.1), (2.0, 0.3), (2.9, 0.2), (0.0, 0.4)]
MU = np.log1p(np.array([c[0] for c in CASES]))
SCALE = np.array([c[1] for c in CASES])


def _latent_points(n, dim=16, seed=0):
    """Synthetic latent vectors: n points in dim dimensions, a few dead units (always 0) like a ReLU layer."""
    rng = np.random.default_rng(seed)
    phi = np.maximum(rng.standard_normal((n, dim)) @ rng.standard_normal((dim, dim)), 0.0)
    phi[:, :3] = 0.0
    return phi


# --- p(z) CDF and quantile function -------------------------------------------------------------------------------


def test_pz_cdf_and_ppf_are_inverse():
    q = np.linspace(0.001, 0.999, 41)
    for mu, s in zip(MU, SCALE):
        z = pz_ppf(q, mu, s, NU)
        np.testing.assert_allclose(pz_cdf(z, mu, s, NU), q, atol=1e-9)


def test_pz_cdf_edges_and_monotone():
    z = np.linspace(-0.5, DEFAULT_ZMAX + 0.5, 400)
    for mu, s in zip(MU, SCALE):
        c = pz_cdf(z, mu, s, NU)
        assert c[0] == 0.0 and c[-1] == 1.0 and np.all(np.diff(c) >= 0)


def test_pz_cdf_matches_truncated_student_t():
    """Against the definition: (T(log(1 + z)) - T(0)) / (T(log(1 + zmax)) - T(0)), T the Student-t CDF."""
    z = np.array([0.1, 0.4, 0.9, 1.5, 2.5])
    for mu, s in zip(MU[1:4], SCALE[1:4]):  # mid-range objects: the direct formula is accurate there
        T = stats.t(NU, loc=mu, scale=s).cdf
        expected = (T(np.log1p(z)) - T(0.0)) / (T(np.log1p(DEFAULT_ZMAX)) - T(0.0))
        np.testing.assert_allclose(pz_cdf(z, mu, s, NU), expected, rtol=1e-10)


def test_pz_ppf_elementwise_matches_grid_ppf():
    """One quantile level per object (pz_ppf) agrees with the shared-grid _ppf used by package_pdfs."""
    q = np.linspace(0.01, 0.99, 7)
    grid = _ppf(q, MU, SCALE, NU, DEFAULT_ZMAX)
    for j, level in enumerate(q):
        np.testing.assert_array_equal(pz_ppf(np.full(len(MU), level), MU, SCALE, NU), grid[:, j])


# --- similarity in the latent space (Eqs. pca and similarity) -------------------------------------------------------


def _copula(xi, rho):
    """A copula with the PCA basis of the training latents xi and the given rho."""
    return _pca_basis(xi) | dict(rho=rho)


def test_pca_drops_dead_units_and_whitens_the_training_set():
    xi = _latent_points(2000)
    cop = _copula(xi, 0.3)
    assert not cop["active"][:3].any() and cop["active"][3:].all()
    w = _pca_projection(cop, xi)
    assert w.shape == (2000, 13)
    np.testing.assert_allclose(w.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.cov(w, rowvar=False), np.eye(13), atol=1e-8)


def test_similarity_does_not_depend_on_the_scale_of_each_neuron():
    """cos(theta) is the angle in the metric Sigma_xi^-1: rescaling the neurons leaves it unchanged."""
    xi = _latent_points(1000)
    scaled = xi * np.random.default_rng(1).uniform(0.1, 10.0, xi.shape[1])
    cos = _cos_theta(*(2 * [_pca_projection(_copula(xi, 0.3), xi[:20])]))
    cos_scaled = _cos_theta(*(2 * [_pca_projection(_copula(scaled, 0.3), scaled[:20])]))
    np.testing.assert_allclose(cos, cos_scaled, atol=1e-8)
    np.testing.assert_allclose(np.diag(cos), 1.0)


# --- sampling (Eqs. neuron-copula and copula) ----------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 0.6])
def test_marginals_are_each_objects_pz(rho):
    """Whatever rho, the draws of each object follow its own p(z): their PITs are uniform (KS test)."""
    train = _latent_points(200)
    xi = _latent_points(len(MU), seed=1)
    cop = _copula(train, rho)
    xi[0, cop["active"]] = cop["mean"]  # w_j = 0: no direction, sampled independently, still from its own p(z)
    draws = _sample(cop, xi, MU, SCALE, NU, n_draws=4000, rng=np.random.default_rng(1))
    assert draws.shape == (4000, len(MU)) and draws.dtype == np.float32
    assert np.all((draws >= 0) & (draws <= DEFAULT_ZMAX))
    for i in range(len(MU)):
        pit = pz_cdf(draws[:, i].astype(np.float64), MU[i], SCALE[i], NU)
        assert stats.kstest(pit, "uniform").pvalue > 1e-3, i


def test_g_has_unit_variance_and_correlation_rho_cos_theta():
    """Eq. neuron-copula: Var(g_j) = 1 and Corr(g_j, g_j') = rho cos(theta_jj') (Eq. neuron-corr)."""
    rho, n_draws = 0.5, 20_000
    train = _latent_points(500)
    cop = _copula(train, rho)
    w = _pca_projection(cop, _latent_points(6, seed=2))
    w[1] = 2.5 * w[0]  # same direction as object 0: correlation rho
    rng = np.random.default_rng(3)
    g = _sample_g(w, rng.standard_normal((w.shape[1], n_draws)), rng.standard_normal((len(w), n_draws)), rho)
    np.testing.assert_allclose(g.var(axis=1), 1.0, atol=0.05)
    expected = rho * _cos_theta(w, w)
    np.fill_diagonal(expected, 1.0)
    np.testing.assert_allclose(np.corrcoef(g), expected, atol=0.04)
    assert np.corrcoef(g)[0, 1] == pytest.approx(rho, abs=0.03)


def _normal_scores(draws, mu, scale):
    return stats.norm.ppf(np.clip(pz_cdf(draws.astype(np.float64), mu, scale, NU), 1e-9, 1 - 1e-9))


def test_redshift_correlation_follows_g():
    """Through Eq. copula, the normal scores of the redshift draws are the g_j: same object twice, correlation rho."""
    xi = _latent_points(200)
    base = xi[:1]
    draws = _sample(_copula(xi, 0.6), np.vstack([base, base]), MU[[1, 2]], SCALE[[1, 2]], NU, n_draws=6000,
                    rng=np.random.default_rng(4))
    assert np.corrcoef(_normal_scores(draws, MU[[1, 2]], SCALE[[1, 2]]).T)[0, 1] == pytest.approx(0.6, abs=0.04)


@pytest.mark.parametrize("rho_override", [None, 0.0])
def test_rho_zero_is_independent(rho_override):
    """rho = 0 in the copula, or passed to _sample over a copula with rho > 0: independent draws."""
    xi = _latent_points(200)
    cop = _copula(xi, 0.0 if rho_override is None else 0.8)
    draws = _sample(cop, np.vstack([xi[:1], xi[:1]]), MU[[1, 2]], SCALE[[1, 2]], NU, n_draws=5000,
                    rng=np.random.default_rng(6), rho=rho_override)
    assert abs(np.corrcoef(_normal_scores(draws, MU[[1, 2]], SCALE[[1, 2]]).T)[0, 1]) < 0.05


def test_draws_do_not_depend_on_chunk():
    xi = _latent_points(103)
    mu, scale = np.full(103, 0.5), np.full(103, 0.1)
    cop = _copula(xi, 0.3)
    a = _sample(cop, xi, mu, scale, NU, n_draws=5, rng=np.random.default_rng(7), chunk=10)
    b = _sample(cop, xi, mu, scale, NU, n_draws=5, rng=np.random.default_rng(7), chunk=10_000)
    np.testing.assert_array_equal(a, b)


# --- fit of rho (Eqs. pit-bin, neuron-copula_var and copula_fit) -------------------------------------------------------------


def _clustered_points(n, n_clusters=40, seed=0):
    """Latent vectors in clusters of similar activations, and the cluster of each (used as its n(z) bin)."""
    rng = np.random.default_rng(seed)
    centres = np.maximum(rng.standard_normal((n_clusters, 16)), 0.0) + 0.5 * rng.standard_normal((n_clusters, 16))
    c = rng.integers(0, n_clusters, n)
    xi = np.maximum(centres[c] + 0.3 * rng.standard_normal((n, 16)), 0.0)
    xi[:, :3] = 0.0
    return xi, c


def test_bin_covariance_matches_the_pairwise_sums():
    """Sigma_qq' = rho M_q^-1 M_q'^-1 sum_{j in q, j' in q'} cos(theta_jj') + delta_qq' (1 - rho) / M_q, against the
    explicit sums; the diagonal is Var(g_q) = 1 / M_q + rho M_q^-2 sum_{j != j'} cos(theta_jj')."""
    rho, n_bins = 0.3, 5
    xi, q = _clustered_points(300, n_clusters=n_bins)
    w = _pca_projection(_copula(xi, rho), xi)
    M_q = np.bincount(q, minlength=n_bins).astype(float)
    sigma = _copula_bin_covariance(rho, M_q, _bin_mean_directions(w, q, n_bins))
    for k in range(n_bins):
        for l in range(n_bins):
            cos = _cos_theta(w[q == k], w[q == l])
            expected = rho * cos.mean() + (k == l) * (1 - rho) / M_q[k]
            assert sigma[k, l] == pytest.approx(expected, rel=1e-10)
        off_diagonal = (_cos_theta(w[q == k], w[q == k]).sum() - M_q[k]) / M_q[k] ** 2
        assert sigma[k, k] == pytest.approx(1 / M_q[k] + rho * off_diagonal, rel=1e-10)


def test_bin_covariance_is_the_covariance_of_the_mean_pit_scores():
    """Eq. neuron-copula_var against the covariance of u_q over many realizations of Eq. neuron-copula."""
    rho, n_real, n_bins = 0.3, 4000, 5
    xi, q = _clustered_points(200, n_clusters=n_bins, seed=1)
    w = _pca_projection(_copula(xi, rho), xi)
    rng = np.random.default_rng(2)
    g = _sample_g(w, rng.standard_normal((w.shape[1], n_real)), rng.standard_normal((len(w), n_real)), rho)
    u_q = np.array([_bin_pit_means(g[:, r], q, n_bins)[0] for r in range(n_real)])
    M_q = np.bincount(q, minlength=n_bins).astype(float)
    sigma = _copula_bin_covariance(rho, M_q, _bin_mean_directions(w, q, n_bins))
    np.testing.assert_allclose(np.cov(u_q, rowvar=False), sigma, atol=0.1 * np.diag(sigma).max())


def test_fit_is_the_maximum_of_the_gaussian_likelihood():
    """_fit_rho maximises log N(u | 0, Sigma(rho)) (Eq. copula_fit): checked against a fine grid."""
    rng = np.random.default_rng(3)
    xi, q = _clustered_points(3000, n_clusters=30, seed=4)
    w = _pca_projection(_copula(xi, 0.3), xi)
    u_q, M_q = _bin_pit_means(_sample_g(w, rng.standard_normal((13, 1)), rng.standard_normal((len(w), 1)), 0.3)[:, 0],
                              q, 30)
    wbar_q = _bin_mean_directions(w, q, 30)
    grid = np.linspace(0, 0.99, 2000)
    best = grid[np.argmin([_neg_log_likelihood(r, u_q, M_q, wbar_q) for r in grid])]
    assert _fit_rho(u_q, M_q, wbar_q) == pytest.approx(best, abs=1e-3)


def test_fit_ignores_bins_with_fewer_than_two_objects():
    assert _fit_rho(np.array([0.1, 0.0]), np.array([1.0, 0.0]), np.zeros((2, 3))) == 0.0


@pytest.mark.parametrize("rho_true", [0.0, 0.2, 0.5])
def test_fit_recovers_rho(rho_true):
    """PIT scores simulated from the copula itself, binned by cluster: the fit recovers rho."""
    rng = np.random.default_rng(10)
    xi, q = _clustered_points(4000, seed=11)
    w = _pca_projection(_copula(xi, rho_true), xi)
    wbar_q = _bin_mean_directions(w, q, 40)
    estimates = []
    for _ in range(10):  # a has only 13 components, shared by all bins: average over a few fields
        u = _sample_g(w, rng.standard_normal((13, 1)), rng.standard_normal((len(w), 1)), rho_true)[:, 0]
        u_q, M_q = _bin_pit_means(u, q, 40)
        estimates.append(_fit_rho(u_q, M_q, wbar_q))
    assert np.mean(estimates) == pytest.approx(rho_true, abs=0.1)


def test_fit_copula_from_redshifts_end_to_end_on_arrays():
    """_fit_copula from p(z) parameters and reference redshifts drawn with a known rho: rho is recovered. Each
    cluster's p(z)s peak in their own n(z) bin."""
    rng = np.random.default_rng(20)
    n, rho_true = 4000, 0.3
    xi, c = _clustered_points(n, seed=21)
    mu, scale = np.log1p(0.05 * c + 0.025), np.full(n, 0.01)
    fits = []
    for _ in range(10):
        z_ref = _sample(_copula(xi, rho_true), xi, mu, scale, NU, n_draws=1, rng=rng)[0].astype(np.float64)
        fitted = _fit_copula(xi, mu, scale, NU, z_ref)
        assert fitted["n_fit"] > 0.99 * n and fitted["active"].sum() == 13 and fitted["n_bins"] == 40
        fits.append(fitted["rho"])
    assert np.mean(fits) == pytest.approx(rho_true, abs=0.1)


def test_fit_leaves_out_outliers():
    rng = np.random.default_rng(30)
    xi = _latent_points(500, seed=31)
    mu, scale = np.full(500, 0.5), np.full(500, 0.05)
    z_ref = np.expm1(mu + scale * rng.standard_normal(500))
    z_ref[:10] = 2.9  # catastrophic: far in the tail
    assert _fit_copula(xi, mu, scale, NU, z_ref)["n_fit"] <= 490


# --- catalogue level, with a small untrained model ------------------------------------------------------------------


def _make_model(n_bins, seed=0):
    tf.keras.utils.set_random_seed(seed)
    base = build_model(input_shape=(n_bins, 3))
    F = models.Model(base.inputs[0], ToRedshift(0.5, 0.3)(base.outputs[0]))
    G = build_model(input_shape=(n_bins, 3))
    return F, G, NU


def test_fit_and_sample_on_a_catalogue(catalog, bands, feature_config):
    model = _make_model(feature_config["n_bins"])
    cop = fit_copula(model, catalog, bands, "i", feature_config, outlier=np.inf)
    assert isinstance(cop, dict) and 0 <= cop["rho"] < 1 and cop["n_fit"] <= len(catalog)

    draws = sample_redshifts(model, cop, catalog, bands, "i", feature_config, n_draws=7, rng=0)
    assert draws.shape == (7, len(catalog)) and np.all((draws >= 0) & (draws <= DEFAULT_ZMAX))
    again = sample_redshifts(model, cop, catalog, bands, "i", feature_config, n_draws=7, rng=0, chunk=3)
    np.testing.assert_array_equal(draws, again)


def test_fit_copula_needs_redshifts(catalog, bands, feature_config):
    model = _make_model(feature_config["n_bins"])
    with pytest.raises(ValueError, match="reference redshifts"):
        fit_copula(model, catalog.drop(columns="redshift"), bands, "i", feature_config)
