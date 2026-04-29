import os
import math
import sys
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from sklearn.model_selection import GroupKFold


SEED = 42
BOOTSTRAP_B = 200
ALPHA_GRID = np.linspace(0.5, 10, 30) 
GG_INIT_ALPHA = 1.0
RESULTS_DIR = "results"
SUMMARY_CSV = "real_data_gg_summary.csv"
USE_OPTUNA = True
USE_OPTUNA_GG_BOOSTING = True
RUN_DATASETS_IN_SUBPROCESS = True
US_FUNDAMENTALS_MAX_OBS_PER_GROUP = 8
ZILLOW_MAX_OBS_PER_GROUP = 30
AIRBNB_NYC_MAX_OBS_PER_GROUP = 15


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    g_test: np.ndarray


def import_gpboost():
    fallback = os.path.join(os.path.dirname(__file__), "GPBoost_full backup", "python-package")
    if os.path.isdir(fallback) and fallback not in sys.path:
        sys.path.insert(0, fallback)
    import gpboost as gpb
    return gpb


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[valid]
    y_p = y_pred[valid]
    return {
        "rmse": float(np.sqrt(np.mean((y_t - y_p) ** 2))),
        "mae": float(np.mean(np.abs(y_t - y_p))),
        "corr": float(np.corrcoef(y_t, y_p)[0, 1]) if len(y_t) > 2 else np.nan,
    }


def bootstrap_group_se(group, stat_fn, B=BOOTSTRAP_B, seed=SEED):
    group = np.asarray(group)
    unique_groups = np.unique(group)
    if len(unique_groups) < 2:
        return np.nan
    idx_by_group = {gid: np.flatnonzero(group == gid) for gid in unique_groups}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(B)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_idx = []
        sampled_group = []
        for new_gid, gid in enumerate(sampled_groups):
            idx = idx_by_group[gid]
            sampled_idx.append(idx)
            sampled_group.append(np.full(len(idx), new_gid, dtype=int))
        idx = np.concatenate(sampled_idx)
        boot_group = np.concatenate(sampled_group)
        val = stat_fn(idx, boot_group)
        if np.isfinite(val):
            vals.append(float(val))
    if len(vals) < 2:
        return np.nan
    return float(np.std(vals, ddof=1))


def summarize_prediction_metrics(y_true, y_pred, group):
    vals = metrics(y_true, y_pred)
    vals["rmse_se"] = bootstrap_group_se(
        group,
        lambda idx, _: metrics(y_true[idx], y_pred[idx])["rmse"],
    )
    vals["mae_se"] = bootstrap_group_se(
        group,
        lambda idx, _: metrics(y_true[idx], y_pred[idx])["mae"],
    )
    vals["corr_se"] = bootstrap_group_se(
        group,
        lambda idx, _: metrics(y_true[idx], y_pred[idx])["corr"],
    )
    return vals


def format_estimate_with_se(value, se, digits=5):
    if not np.isfinite(value):
        return "NaN"
    value_str = f"{value:.{digits}g}"
    if np.isfinite(se):
        return f"{value_str} ({se:.3g})"
    return value_str


def highlight_metric_flags(summary, metric_col, se_col, higher_is_better=False, z=1.96):
    is_best = pd.Series(False, index=summary.index)
    is_secondary = pd.Series(False, index=summary.index)

    for _, idx in summary.groupby("dataset").groups.items():
        idx = list(idx)
        vals = pd.to_numeric(summary.loc[idx, metric_col], errors="coerce")
        ses = pd.to_numeric(summary.loc[idx, se_col], errors="coerce")
        finite = np.isfinite(vals.to_numpy())
        if not finite.any():
            continue

        finite_idx = np.asarray(idx)[finite]
        finite_vals = vals.loc[finite_idx]
        best_idx = finite_vals.idxmax() if higher_is_better else finite_vals.idxmin()
        best_val = float(summary.loc[best_idx, metric_col])
        is_best.loc[best_idx] = True

        for row_idx in finite_idx:
            if row_idx == best_idx:
                continue
            se = float(summary.loc[row_idx, se_col])
            if not np.isfinite(se):
                continue
            val = float(summary.loc[row_idx, metric_col])
            if higher_is_better:
                close_enough = val >= best_val - z * se
            else:
                close_enough = val <= best_val + z * se
            if close_enough:
                is_secondary.loc[row_idx] = True

    return is_best, is_secondary


def apply_highlight_markup(text, is_best=False, is_secondary=False):
    if is_best:
        return f"**{text}**"
    if is_secondary:
        return f"*{text}*"
    return text


def split_within_groups(group, test_size=0.2, seed=SEED, consecutive=False):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for gid in np.unique(group):
        idx = np.flatnonzero(group == gid)
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue
        n_test = max(1, int(np.floor(test_size * len(idx))))
        if consecutive:
            start = int(rng.integers(0, len(idx) - n_test + 1))
            chosen = idx[start:start + n_test]
        else:
            chosen = rng.choice(idx, size=n_test, replace=False)
        chosen_set = set(np.asarray(chosen).tolist())
        test_idx.extend(np.asarray(chosen).tolist())
        train_idx.extend([i for i in idx.tolist() if i not in chosen_set])
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def posterior_group_stats(group, y, f, alpha, delta):
    uniq, inv = np.unique(group, return_inverse=True)
    n_i = np.bincount(inv).astype(float)
    s_i = np.bincount(inv, weights=np.exp(f) * y).astype(float)
    shape = delta + n_i * alpha
    rate = delta + s_i
    return shape, rate, {gid: i for i, gid in enumerate(uniq)}


def predict_grouped_from_beta(x, group, beta, alpha, shape, rate, id_to_idx, delta):
    return predict_grouped_from_f(x @ beta, group, alpha, shape, rate, id_to_idx, delta)


def predict_grouped_from_f(f, group, alpha, shape, rate, id_to_idx, delta):
    pred = np.empty(len(group), dtype=float)
    for i, gid in enumerate(group):
        if gid in id_to_idx:
            k = shape[id_to_idx[gid]]
            r = rate[id_to_idx[gid]]
        else:
            k = delta
            r = delta
        pred[i] = alpha * np.exp(-f[i]) * r / max(k - 1.0, 1e-12)
    return pred


def gamma_fe_objective(beta, x, y, alpha):
    f = x @ beta
    return float(np.sum(math.lgamma(alpha) - alpha * f - (alpha - 1.0) * np.log(y) + np.exp(f) * y))


def gamma_fe_grad(beta, x, y, alpha):
    f = x @ beta
    return x.T @ (-alpha + np.exp(f) * y)


def gamma_fe_test_nll(y, f, alpha):
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    return float(np.sum(gammaln(alpha) - alpha * f - (alpha - 1.0) * np.log(y) + np.exp(f) * y))


def grouped_gg_test_nll(y, group, f, alpha, delta):
    y = np.asarray(y, dtype=float)
    group = np.asarray(group)
    f = np.asarray(f, dtype=float)
    _, inv = np.unique(group, return_inverse=True)
    n_i = np.bincount(inv).astype(float)
    sum_f = np.bincount(inv, weights=f).astype(float)
    sum_log_y = np.bincount(inv, weights=np.log(y)).astype(float)
    sum_exp_f_y = np.bincount(inv, weights=np.exp(f) * y).astype(float)
    return float(np.sum(
        n_i * gammaln(alpha)
        - delta * np.log(delta)
        + gammaln(delta)
        - gammaln(delta + n_i * alpha)
        - alpha * sum_f
        - (alpha - 1.0) * sum_log_y
        + (delta + n_i * alpha) * np.log(delta + sum_exp_f_y)
    ))


def gamma_normal_mc_test_nll(y, latent_mean, latent_var, alpha, n_samples=4096, seed=SEED, chunk_size=512):
    y = np.asarray(y, dtype=float)
    latent_mean = np.asarray(latent_mean, dtype=float)
    latent_var = np.maximum(np.asarray(latent_var, dtype=float), 0.0)
    alpha = float(alpha)
    if len(y) == 0 or not np.isfinite(alpha) or alpha <= 0:
        return np.nan

    rng = np.random.default_rng(seed)
    half_samples = max(1, int(np.ceil(n_samples / 2)))
    log_alpha = np.log(alpha)
    const = alpha * log_alpha - gammaln(alpha)
    nll = 0.0
    for start in range(0, len(y), chunk_size):
        end = min(start + chunk_size, len(y))
        eps = rng.normal(size=(end - start, half_samples))
        eps = np.concatenate([eps, -eps], axis=1)[:, :n_samples]
        z = latent_mean[start:end, None] + np.sqrt(latent_var[start:end, None]) * eps
        yi = y[start:end, None]
        log_pdf = const - alpha * z + (alpha - 1.0) * np.log(yi) - alpha * yi * np.exp(-z)
        nll -= float(np.sum(logsumexp(log_pdf, axis=1) - np.log(log_pdf.shape[1])))
    return nll


def fit_global_gamma_fe(x_train, y_train, alpha_grid):
    beta0 = np.zeros(x_train.shape[1], dtype=float)
    best = None
    for alpha in alpha_grid:
        beta0[0] = -np.log(np.mean(y_train) / float(alpha)) #maybe this is a better initialization
        opt = minimize(
            fun=lambda b: gamma_fe_objective(b, x_train, y_train, float(alpha)),
            x0=beta0,
            jac=lambda b: gamma_fe_grad(b, x_train, y_train, float(alpha)),
            method="L-BFGS-B",
            options={"maxiter": 250},
        )
        beta_hat = opt.x if opt.success else beta0
        negll = gamma_fe_objective(beta_hat, x_train, y_train, float(alpha))
        if best is None or negll < best["negll"]:
            best = {"alpha": float(alpha), "beta": beta_hat, "negll": float(negll)}
    return best


def fit_groupwise_gamma_fe(x_train, y_train, g_train, alpha, beta_fallback):
    models = {}
    p = x_train.shape[1]
    for gid in np.unique(g_train):
        idx = np.flatnonzero(g_train == gid)
        if len(idx) <= 2:
            models[int(gid)] = beta_fallback.copy()
            continue
        xg = x_train[idx]
        yg = y_train[idx]
        beta0 = np.zeros(p, dtype=float)
        beta0[0] = -np.log(np.mean(yg) / float(alpha))#maybe this is a better initialization than all zeros
        opt = minimize(
            fun=lambda b: gamma_fe_objective(b, xg, yg, alpha),
            x0=beta0,
            jac=lambda b: gamma_fe_grad(b, xg, yg, alpha),
            method="L-BFGS-B",
            options={"maxiter": 120},
        )
        models[int(gid)] = (opt.x if opt.success else beta_fallback).copy()
    return models


def predict_groupwise_gamma_fe(x, g, alpha, models, beta_fallback):
    pred = np.empty(x.shape[0], dtype=float)
    for i, gid in enumerate(g):
        beta = models.get(int(gid), beta_fallback)
        pred[i] = alpha * np.exp(-(x[i] @ beta))
    return pred


def fit_gg_linear(
    x_train,
    y_train,
    g_train,
    init_alpha=GG_INIT_ALPHA,
    starts=None,
    maxit=240,
    try_lbfgs_fallback=True,
):
    gpb = import_gpboost()
    def _fit_once(alpha0, optimizer_cov="nelder_mead", optimizer_coef="nelder_mead"):
        gp_model = gpb.GPModel(
            group_data=g_train.astype(np.int32, copy=False),
            likelihood="gamma_gamma",
            likelihood_additional_param=float(alpha0),
            num_data=len(y_train),
            free_raw_data=False,
        )
        gp_model._user_likelihood = "gamma_gamma"
        gp_model.fit(
            y=y_train.astype(np.float64, copy=False),
            X=x_train.astype(np.float64, copy=False),
            params={
                "optimizer_cov": optimizer_cov,
                "optimizer_coef": optimizer_coef,
                "init_cov_pars": np.array([1.0], dtype=float),
                "init_aux_pars": np.array([float(alpha0)], dtype=float),
                "estimate_aux_pars": True,
                "init_coef": np.zeros(x_train.shape[1], dtype=float),
                "maxit": int(maxit),
                "trace": False,
            },
        )
        return {
            "alpha_init": float(alpha0),
            "alpha": float(np.asarray(gp_model.get_aux_pars(format_pandas=False)).reshape(-1)[0]),
            "negll": float(gp_model.get_current_neg_log_likelihood()),
            "delta": float(np.asarray(gp_model.get_cov_pars(format_pandas=False)).reshape(-1)[0]),
            "beta": np.asarray(gp_model.get_coef(format_pandas=False), dtype=float).reshape(-1),
            "gp_model": gp_model,
        }

    if starts is None:
        starts = [float(init_alpha), max(0.2, 0.5 * float(init_alpha)), max(1.5, 2.0 * float(init_alpha))]
    else:
        starts = [max(float(a), 1e-6) for a in starts]
    fits = []
    for a0 in starts:
        try:
            fits.append(_fit_once(a0))
        except Exception:
            continue

    # If alpha does not move in the robust Nelder-Mead pass, try one L-BFGS pass.
    if try_lbfgs_fallback and fits and all(np.isclose(r["alpha"], r["alpha_init"], rtol=0.0, atol=1e-10) for r in fits):
        for a0 in starts:
            try:
                fits.append(_fit_once(a0, optimizer_cov="lbfgs", optimizer_coef="lbfgs"))
            except Exception:
                continue

    if not fits:
        raise RuntimeError("All Gamma-Gamma linear fits failed")
    best = min(fits, key=lambda r: r["negll"])
    if all(np.isclose(r["alpha"], r["alpha_init"], rtol=0.0, atol=1e-10) for r in fits):
        print("[warn] gg_gpboost linear: alpha did not move from init in any restart; returning best negll among starts")
    return best


def fit_gamma_normal_linear(x_train, y_train, g_train, maxit=240):
    gpb = import_gpboost()
    gp_model = gpb.GPModel(
        group_data=g_train.astype(np.int32, copy=False),
        likelihood="gamma",
        num_data=len(y_train),
        free_raw_data=False,
    )
    gp_model.fit(
        y=y_train.astype(np.float64, copy=False),
        X=x_train.astype(np.float64, copy=False),
        params={
            "optimizer_cov": "nelder_mead",
            "optimizer_coef": "nelder_mead",
            "init_cov_pars": np.array([1.0], dtype=float),
            "estimate_aux_pars": True,
            "init_coef": np.zeros(x_train.shape[1], dtype=float),
            "maxit": int(maxit),
            "trace": False,
        },
    )
    aux = np.asarray(gp_model.get_aux_pars(format_pandas=False), dtype=float).reshape(-1)
    cov = np.asarray(gp_model.get_cov_pars(format_pandas=False), dtype=float).reshape(-1)
    return {
        "alpha": float(aux[0]) if len(aux) else np.nan,
        "negll": float(gp_model.get_current_neg_log_likelihood()),
        "cov_par": float(cov[0]) if len(cov) else np.nan,
        "beta": np.asarray(gp_model.get_coef(format_pandas=False), dtype=float).reshape(-1),
        "gp_model": gp_model,
    }


def fit_gamma_normal_boosted(x_train, y_train, g_train, num_boost_round=200, learning_rate=0.05, tree_params=None):
    gpb = import_gpboost()
    gp_model = gpb.GPModel(
        group_data=g_train.astype(np.int32, copy=False),
        likelihood="gamma",
        num_data=len(y_train),
        free_raw_data=False,
    )
    gp_model.set_optim_params({
        "estimate_aux_pars": True,
    })
    params = {
        "objective": "gamma",
        "metric": "gamma",
        "learning_rate": float(learning_rate),
        "num_leaves": 3,
        "min_data_in_leaf": 10,
        "max_depth": 1,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "verbose": -1,
    }
    if tree_params:
        params.update(tree_params)
    booster = gpb.train(
        params=params,
        train_set=gpb.Dataset(
            x_train.astype(np.float64, copy=False),
            label=y_train.astype(np.float64, copy=False),
            free_raw_data=False,
        ),
        gp_model=gp_model,
        num_boost_round=int(num_boost_round),
    )
    f_train = np.asarray(
        booster.predict(
            data=x_train.astype(np.float64, copy=False),
            group_data_pred=g_train.astype(np.int32, copy=False),
            pred_latent=True,
            ignore_gp_model=True,
        ),
        dtype=float,
    ).reshape(-1)
    try:
        negll = float(gp_model.get_current_neg_log_likelihood())
    except Exception:
        negll = np.nan
    aux = np.asarray(gp_model.get_aux_pars(format_pandas=False), dtype=float).reshape(-1)
    cov = np.asarray(gp_model.get_cov_pars(format_pandas=False), dtype=float).reshape(-1)
    return {
        "alpha": float(aux[0]) if len(aux) else np.nan,
        "negll": negll,
        "cov_par": float(cov[0]) if len(cov) else np.nan,
        "booster": booster,
        "f_train": f_train,
    }


def fit_gg_boosted(x_train, y_train, g_train, init_alpha=GG_INIT_ALPHA, num_boost_round=200, learning_rate=0.05, tree_params=None):
    gpb = import_gpboost()
    gp_model = gpb.GPModel(
        group_data=g_train.astype(np.int32, copy=False),
        likelihood="gamma_gamma",
        likelihood_additional_param=float(init_alpha),
        num_data=len(y_train),
        free_raw_data=False,
    )
    gp_model._user_likelihood = "gamma_gamma"
    gp_model.set_optim_params({
        "optimizer_cov": "gradient_descent",
        "optimizer_coef": "gradient_descent",
        "init_cov_pars": np.array([1.0], dtype=float),
        "init_aux_pars": np.array([float(init_alpha)], dtype=float),
        "estimate_aux_pars": True,
        "lr_cov": 1e-3,
        "lr_coef": 1e-3,
        "trace": False,
    })
    params = {
        "objective": "gamma_gamma",
        "metric": "test_neg_log_likelihood",
        "boost_from_average": False,
        "learning_rate": float(learning_rate),
        "num_leaves": 3,
        "min_data_in_leaf": 10,
        "max_depth": 1,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "verbose": -1,
    }
    if tree_params:
        params.update(tree_params)
    booster = gpb.train(
        params=params,
        train_set=gpb.Dataset(
            x_train.astype(np.float64, copy=False),
            label=y_train.astype(np.float64, copy=False),
            free_raw_data=False,
        ),
        gp_model=gp_model,
        num_boost_round=int(num_boost_round),
    )
    f_train = np.asarray(
        booster.predict(
            data=x_train.astype(np.float64, copy=False),
            group_data_pred=g_train.astype(np.int32, copy=False),
            pred_latent=True,
            ignore_gp_model=True,
        ),
        dtype=float,
    ).reshape(-1)
    try:
        negll = float(gp_model.get_current_neg_log_likelihood())
    except Exception:
        negll = np.nan
    return {
        "alpha": float(np.asarray(gp_model.get_aux_pars(format_pandas=False)).reshape(-1)[0]),
        "negll": negll,
        "delta": float(np.asarray(gp_model.get_cov_pars(format_pandas=False)).reshape(-1)[0]),
        "booster": booster,
        "f_train": f_train,
    }


def tune_gg_boosting(x_train, y_train, g_train, init_alpha=GG_INIT_ALPHA, n_trials=8, n_splits=5, max_rounds=200, early_stopping=20):
    gpb = import_gpboost()
    gp_model = gpb.GPModel(
        group_data=g_train.astype(np.int32, copy=False),
        likelihood="gamma_gamma",
        likelihood_additional_param=float(init_alpha),
        num_data=len(y_train),
        free_raw_data=False,
    )
    gp_model._user_likelihood = "gamma_gamma"
    gp_model.set_optim_params({
        "optimizer_cov": "gradient_descent",
        "optimizer_coef": "gradient_descent",
        "init_cov_pars": np.array([1.0], dtype=float),
        "trace": False,
        "init_aux_pars": np.array([float(init_alpha)], dtype=float),
        "estimate_aux_pars": True,
        "lr_cov": 1e-3,
        "lr_coef": 1e-3,
    })
    folds = list(GroupKFold(n_splits=n_splits).split(x_train, y_train, groups=g_train))
    search_space = {
        "learning_rate": [0.03, 0.06],
        "min_data_in_leaf": [8, 20],
        "max_depth": [1, 1],
        "num_leaves": [2, 3],
        "lambda_l2": [0.0, 2.0],
        "max_bin": [127, 255],
        "feature_fraction": [0.8, 1.0],
    }
    opt = gpb.tune_pars_TPE_algorithm_optuna(
        search_space=search_space,
        n_trials=int(n_trials),
        X=x_train.astype(np.float64, copy=False),
        y=y_train.astype(np.float64, copy=False),
        gp_model=gp_model,
        max_num_boost_round=int(max_rounds),
        early_stopping_rounds=int(early_stopping),
        folds=folds,
        metric="test_neg_log_likelihood",
        cv_seed=4,
        tpe_seed=1,
        verbose_eval=1,
        params={"objective": "gamma_gamma", "boost_from_average": False, "line_search_step_length": False, "verbose": -1},
        use_gp_model_for_validation=True,
        train_gp_model_cov_pars=True,
    )
    best_params = dict(opt.get("best_params", {}))
    best_rounds = int(opt.get("best_num_boost_round", opt.get("best_iter", 100)))
    return best_params, best_rounds


def tune_plain_boosting(x_train, y_train, g_train, n_trials=8, n_splits=5, max_rounds=200, early_stopping=20):
    gpb = import_gpboost()
    folds = list(GroupKFold(n_splits=n_splits).split(x_train, y_train, groups=g_train))
    search_space = {
        "learning_rate": [0.03, 0.08],
        "min_data_in_leaf": [8, 20],
        "max_depth": [1, 2],
        "num_leaves": [2, 6],
        "lambda_l2": [0.0, 2.0],
        "max_bin": [127, 255],
        "feature_fraction": [0.8, 1.0],
        "line_search_step_length": [False, True],
    }
    opt = gpb.tune_pars_TPE_algorithm_optuna(
        search_space=search_space,
        n_trials=int(n_trials),
        X=x_train.astype(np.float64, copy=False),
        y=y_train.astype(np.float64, copy=False),
        gp_model=None,
        max_num_boost_round=int(max_rounds),
        early_stopping_rounds=int(early_stopping),
        folds=folds,
        metric="rmse",
        cv_seed=4,
        tpe_seed=1,
        verbose_eval=1,
        params={"objective": "regression_l2", "verbose": -1},
    )
    best_params = dict(opt.get("best_params", {}))
    best_rounds = int(opt.get("best_num_boost_round", opt.get("best_iter", 100)))
    return best_params, best_rounds


def fit_plain_boosting_global(
    x_train,
    y_train,
    num_boost_round=200,
    learning_rate=0.05,
    tree_params=None,
    categorical_feature=None,
):
    gpb = import_gpboost()
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": float(learning_rate),
        "num_leaves": 3,
        "min_data_in_leaf": 10,
        "max_depth": 1,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "verbose": -1,
    }
    if tree_params:
        params.update(tree_params)
    booster = gpb.train(
        params=params,
        train_set=gpb.Dataset(
            x_train.astype(np.float64, copy=False),
            label=y_train.astype(np.float64, copy=False),
            categorical_feature=categorical_feature,
            free_raw_data=False,
        ),
        num_boost_round=int(num_boost_round),
    )
    return booster


def standardize_train_test(x_train, x_test):
    x_train = x_train.copy().astype(float)
    x_test = x_test.copy().astype(float)
    for j in range(1, x_train.shape[1]):
        mean_j = x_train[:, j].mean()
        std_j = x_train[:, j].std()
        if std_j > 0:
            x_train[:, j] = (x_train[:, j] - mean_j) / std_j
            x_test[:, j] = (x_test[:, j] - mean_j) / std_j
    return x_train, x_test


def impute_train_test_medians(x_train, x_test):
    x_train = x_train.copy().astype(float)
    x_test = x_test.copy().astype(float)
    for j in range(1, x_train.shape[1]):
        train_col = x_train[:, j]
        test_col = x_test[:, j]
        if not (np.isnan(train_col).any() or np.isnan(test_col).any()):
            continue
        median_j = np.nanmedian(train_col)
        if not np.isfinite(median_j):
            median_j = 0.0
        x_train[:, j] = np.where(np.isnan(train_col), median_j, train_col)
        x_test[:, j] = np.where(np.isnan(test_col), median_j, test_col)
    return x_train, x_test


def sample_groups(df, group_col, max_groups, seed=SEED):
    if max_groups is None:
        return df
    group_ids = pd.Series(df[group_col].astype(str).unique())
    if len(group_ids) <= max_groups:
        return df
    keep = set(group_ids.sample(n=int(max_groups), random_state=seed).tolist())
    return df[df[group_col].astype(str).isin(keep)].copy()


def cap_observations_per_group(df, group_col, max_obs_per_group, seed=SEED):
    if max_obs_per_group is None:
        return df
    max_obs_per_group = int(max_obs_per_group)
    return (
        df.sample(frac=1.0, random_state=seed)
        .groupby(group_col, group_keys=False)
        .head(max_obs_per_group)
        .reset_index(drop=True)
    )


def cap_contiguous_observations_per_group(
    df,
    group_col,
    order_col,
    max_obs_per_group,
    seed=SEED,
):
    if max_obs_per_group is None:
        return df
    max_obs_per_group = int(max_obs_per_group)
    rng = np.random.default_rng(seed)
    chunks = []
    for _, group in df.sort_values([group_col, order_col]).groupby(group_col, sort=False):
        if len(group) <= max_obs_per_group:
            chunks.append(group)
            continue
        start = int(rng.integers(0, len(group) - max_obs_per_group + 1))
        chunks.append(group.iloc[start:start + max_obs_per_group])
    return pd.concat(chunks, axis=0).reset_index(drop=True)


def load_us_fundamentals(
    base_dir,
    max_rows=120000,
    min_obs_per_group=None,
    max_obs_per_group=None,
    min_groups=2,
):
    df = pd.read_csv(os.path.join(base_dir, "assets_forecast_panel.csv"))
    df = df.dropna(subset=["company_name", "feature_year", "y_assets_next"]).copy()
    df = df[df["y_assets_next"] > 0].copy()
    df["feature_year"] = pd.to_numeric(df["feature_year"], errors="coerce")
    if min_obs_per_group is not None or max_obs_per_group is not None:
        group_sizes = df.groupby("company_name")["company_name"].transform("size")
        keep = np.ones(len(df), dtype=bool)
        if min_obs_per_group is not None:
            keep &= group_sizes.to_numpy() >= int(min_obs_per_group)
        if max_obs_per_group is not None:
            keep &= group_sizes.to_numpy() <= int(max_obs_per_group)
        df = df[keep].copy()
        num_groups = df["company_name"].nunique()
        if num_groups < min_groups:
            raise ValueError(
                "us_fundamentals group-size filter "
                f"[{min_obs_per_group}, {max_obs_per_group}] leaves only {num_groups} groups"
            )
    else:
        df = cap_observations_per_group(
            df,
            "company_name",
            max_obs_per_group=US_FUNDAMENTALS_MAX_OBS_PER_GROUP,
            seed=SEED,
        )
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    df = df.sort_values(["company_name", "feature_year"]).reset_index(drop=True)
    y = df["y_assets_next"].astype(float).to_numpy()
    g = pd.factorize(df["company_name"])[0].astype(np.int32)
    feature_cols = [
        "y_assets", "assets_current", "liabilities", "liabilities_current", "equity", "cash", "net_income",
        "operating_income", "ppe_net", "cf_operating", "cf_investing", "cf_financing",
        "current_assets_to_assets", "liabilities_to_assets", "current_liabilities_to_assets", "equity_to_assets",
        "cash_to_assets", "net_income_to_assets", "operating_income_to_assets", "ppe_to_assets",
        "cf_operating_to_assets", "cf_investing_to_assets", "cf_financing_to_assets",
    ]
    present = [c for c in feature_cols if c in df.columns]
    x_df = df[present].apply(pd.to_numeric, errors="coerce")
    year = pd.to_numeric(df["feature_year"], errors="coerce").astype(float)
    x = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            pd.Series(year, index=df.index, name="feature_year"),
            x_df,
        ],
        axis=1,
    ).astype(float).to_numpy()
    return x, y, g


def load_zillow(base_dir, max_rows=120000, max_obs_per_group=ZILLOW_MAX_OBS_PER_GROUP):
    df = pd.read_csv(os.path.join(base_dir, "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"))
    id_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    date_cols = [c for c in df.columns if c[:4].isdigit()]
    df = df[df["RegionType"].eq("msa")].dropna(subset=["RegionID", "SizeRank", "StateName"]).copy()
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="date",
        value_name="zhvi",
    )
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df["zhvi"] = pd.to_numeric(long_df["zhvi"], errors="coerce")
    long_df = long_df.dropna(subset=["date", "zhvi"]).copy()
    long_df = long_df[long_df["zhvi"] > 0].sort_values(["RegionID", "date"]).reset_index(drop=True)
    by_region = long_df.groupby("RegionID", sort=False)
    long_df["next_zhvi"] = by_region["zhvi"].shift(-1)
    long_df["prev_zhvi"] = by_region["zhvi"].shift(1)
    long_df["zhvi_12m_ago"] = by_region["zhvi"].shift(12)
    long_df["monthly_return"] = long_df["zhvi"] / long_df["prev_zhvi"] - 1.0
    long_df["annual_return"] = long_df["zhvi"] / long_df["zhvi_12m_ago"] - 1.0
    long_df = long_df.dropna(
        subset=["next_zhvi", "prev_zhvi", "zhvi_12m_ago", "monthly_return", "annual_return"]
    ).copy()
    long_df = long_df[long_df["next_zhvi"] > 0].copy()
    long_df = cap_contiguous_observations_per_group(
        long_df,
        "RegionID",
        "date",
        max_obs_per_group=max_obs_per_group,
        seed=SEED,
    )
    if len(long_df) > max_rows:
        long_df = long_df.sample(n=max_rows, random_state=SEED)
    y = long_df["next_zhvi"].astype(float).to_numpy()
    g = pd.factorize(long_df["RegionID"])[0].astype(np.int32)
    year = long_df["date"].dt.year.astype(float)
    month_angle = 2.0 * np.pi * (long_df["date"].dt.month.astype(float) - 1.0) / 12.0
    x = pd.concat(
        [
            pd.Series(1.0, index=long_df.index, name="intercept"),
            pd.Series(year, index=long_df.index, name="year"),
            pd.Series(np.sin(month_angle), index=long_df.index, name="month_sin"),
            pd.Series(np.cos(month_angle), index=long_df.index, name="month_cos"),
            pd.Series(np.log1p(long_df["SizeRank"].astype(float)), index=long_df.index, name="log_size_rank"),
            long_df[["monthly_return", "annual_return"]].astype(float),
            pd.get_dummies(long_df[["StateName"]].astype(str), drop_first=True),
        ],
        axis=1,
    ).astype(float).to_numpy()
    return x, y, g


def load_airbnb_nyc(base_dir, max_rows=120000, max_obs_per_group=AIRBNB_NYC_MAX_OBS_PER_GROUP):
    df = pd.read_csv(os.path.join(base_dir, "AB_NYC_2019.csv"))
    required = [
        "host_id", "neighbourhood_group", "neighbourhood", "latitude", "longitude",
        "room_type", "price", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count", "availability_365",
    ]
    df = df.dropna(subset=[c for c in required if c != "reviews_per_month"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[(df["price"] > 0) & (df["price"] <= 1000)].copy()
    df = cap_observations_per_group(
        df,
        "neighbourhood",
        max_obs_per_group=max_obs_per_group,
        seed=SEED,
    )
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    df["reviews_per_month"] = pd.to_numeric(df["reviews_per_month"], errors="coerce").fillna(0.0)
    y = df["price"].astype(float).to_numpy()
    g = pd.factorize(df["neighbourhood"].astype(str))[0].astype(np.int32)
    x = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            df[["latitude", "longitude"]].astype(float),
            np.log1p(df[["minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count"]].astype(float)).rename(
                columns={
                    "minimum_nights": "log_minimum_nights",
                    "number_of_reviews": "log_number_of_reviews",
                    "reviews_per_month": "log_reviews_per_month",
                    "calculated_host_listings_count": "log_host_listings_count",
                }
            ),
            pd.Series(df["availability_365"].astype(float) / 365.0, index=df.index, name="availability_share"),
            pd.get_dummies(df[["neighbourhood_group", "room_type"]].astype(str), drop_first=True),
        ],
        axis=1,
    ).astype(float).to_numpy()
    return x, y, g


DATASETS = {
    "us_fundamentals_assets": lambda base: load_us_fundamentals(os.path.join(base, "us-stocks-fundamentals")),
    "us_fundamentals_assets_balanced4_8": lambda base: load_us_fundamentals(
        os.path.join(base, "us-stocks-fundamentals"),
        min_obs_per_group=4,
        max_obs_per_group=8,
    ),
    "zillow": lambda base: load_zillow(os.path.join(base, "zillow")),
    "zillow_cap8": lambda base: load_zillow(os.path.join(base, "zillow"), max_obs_per_group=8),
    "airbnb_nyc": lambda base: load_airbnb_nyc(os.path.join(base, "New York City Airbnb Open Data")),
    "airbnb_nyc_cap8": lambda base: load_airbnb_nyc(
        os.path.join(base, "New York City Airbnb Open Data"),
        max_obs_per_group=8,
    ),
}

CONSECUTIVE_SPLIT_DATASETS = {
    "us_fundamentals_assets",
    "us_fundamentals_assets_balanced4_8",
    "zillow",
    "zillow_cap8",
}


def group_structure_record(dataset_name, x, y, g):
    group_sizes = pd.Series(g).value_counts(sort=False).to_numpy()
    return {
        "dataset": dataset_name,
        "n": int(len(y)),
        "p": int(x.shape[1]),
        "G": int(len(group_sizes)),
        "avg_group_size": float(np.mean(group_sizes)),
        "med_group_size": float(np.median(group_sizes)),
        "min_group_size": int(np.min(group_sizes)),
        "max_group_size": int(np.max(group_sizes)),
    }


def make_split(x, y, g, consecutive=False):
    train_idx, test_idx = split_within_groups(g, test_size=0.2, seed=SEED, consecutive=consecutive)
    x_train = x[train_idx]
    x_test = x[test_idx]
    x_train, x_test = impute_train_test_medians(x_train, x_test)
    x_train, x_test = standardize_train_test(x_train, x_test)
    return SplitData(x_train, y[train_idx], g[train_idx], x_test, y[test_idx], g[test_idx])


def run_models(split):
    rows = []

    boost_kwargs = {}
    if USE_OPTUNA and USE_OPTUNA_GG_BOOSTING:
        best_params, best_rounds = tune_gg_boosting(
            split.x_train,
            split.y_train,
            split.g_train,
        )
        boost_kwargs = {
            "num_boost_round": best_rounds,
            "learning_rate": float(best_params.get("learning_rate", 0.05)),
            "tree_params": {
                "num_leaves": int(best_params.get("num_leaves", 3)),
                "min_data_in_leaf": int(best_params.get("min_data_in_leaf", 10)),
                "max_depth": int(best_params.get("max_depth", 1)),
                "lambda_l2": float(best_params.get("lambda_l2", 0.0)),
                "max_bin": int(best_params.get("max_bin", 255)),
                "feature_fraction": float(best_params.get("feature_fraction", 1.0)),
                "line_search_step_length": bool(best_params.get("line_search_step_length", False)),
            },
        }

    gg_linear = fit_gg_linear(split.x_train, split.y_train, split.g_train)
    f_test_linear = split.x_test @ gg_linear["beta"]
    pred = np.asarray(
        gg_linear["gp_model"].predict(
            group_data_pred=split.g_test.astype(np.int32, copy=False),
            X_pred=split.x_test.astype(np.float64, copy=False),
            predict_response=True,
        )["mu"],
        dtype=float,
    ).reshape(-1)
    pred_metrics = summarize_prediction_metrics(split.y_test, pred, split.g_test)
    gg_linear_test_nll = grouped_gg_test_nll(
        split.y_test, split.g_test, f_test_linear, gg_linear["alpha"], gg_linear["delta"]
    )
    rows.append({
        "model": "gg_gpboost",
        "alpha_hat": gg_linear["alpha"],
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": gg_linear_test_nll,
        "test_nll_se": bootstrap_group_se(
            split.g_test,
            lambda idx, boot_group: grouped_gg_test_nll(
                split.y_test[idx], boot_group, f_test_linear[idx], gg_linear["alpha"], gg_linear["delta"]
            ),
        ),
    })

    gamma_normal = fit_gamma_normal_linear(split.x_train, split.y_train, split.g_train)
    pred_gamma_normal = np.asarray(
        gamma_normal["gp_model"].predict(
            group_data_pred=split.g_test.astype(np.int32, copy=False),
            X_pred=split.x_test.astype(np.float64, copy=False),
            predict_response=True,
        )["mu"],
        dtype=float,
    ).reshape(-1)
    latent_gamma_normal = gamma_normal["gp_model"].predict(
        group_data_pred=split.g_test.astype(np.int32, copy=False),
        X_pred=split.x_test.astype(np.float64, copy=False),
        predict_response=False,
        predict_var=True,
    )
    pred_metrics = summarize_prediction_metrics(split.y_test, pred_gamma_normal, split.g_test)
    gamma_normal_test_nll = gamma_normal_mc_test_nll(
        split.y_test,
        np.asarray(latent_gamma_normal["mu"], dtype=float).reshape(-1),
        np.asarray(latent_gamma_normal["var"], dtype=float).reshape(-1),
        gamma_normal["alpha"],
    )
    rows.append({
        "model": "gamma_normal_gpboost",
        "alpha_hat": gamma_normal["alpha"],
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": gamma_normal_test_nll,
        "test_nll_se": bootstrap_group_se(
            split.g_test,
            lambda idx, _: gamma_normal_mc_test_nll(
                split.y_test[idx],
                np.asarray(latent_gamma_normal["mu"], dtype=float).reshape(-1)[idx],
                np.asarray(latent_gamma_normal["var"], dtype=float).reshape(-1)[idx],
                gamma_normal["alpha"],
            ),
        ),
    })

    gg_boosted = fit_gg_boosted(split.x_train, split.y_train, split.g_train, **boost_kwargs)
    f_test = np.asarray(
        gg_boosted["booster"].predict(
            data=split.x_test.astype(np.float64, copy=False),
            group_data_pred=split.g_test.astype(np.int32, copy=False),
            pred_latent=True,
            ignore_gp_model=True,
        ),
        dtype=float,
    ).reshape(-1)
    pred_b = np.asarray(
        gg_boosted["booster"].predict(
            data=split.x_test.astype(np.float64, copy=False),
            group_data_pred=split.g_test.astype(np.int32, copy=False),
            pred_latent=False,
        )["response_mean"],
        dtype=float,
    ).reshape(-1)
    pred_metrics = summarize_prediction_metrics(split.y_test, pred_b, split.g_test)
    gg_boosted_test_nll = grouped_gg_test_nll(
        split.y_test, split.g_test, f_test, gg_boosted["alpha"], gg_boosted["delta"]
    )
    rows.append({
        "model": "gg_gpboost_boosted",
        "alpha_hat": gg_boosted["alpha"],
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": gg_boosted_test_nll,
        "test_nll_se": bootstrap_group_se(
            split.g_test,
            lambda idx, boot_group: grouped_gg_test_nll(
                split.y_test[idx], boot_group, f_test[idx], gg_boosted["alpha"], gg_boosted["delta"]
            ),
        ),
    })

    gamma_normal_boosted = fit_gamma_normal_boosted(
        split.x_train,
        split.y_train,
        split.g_train,
        **boost_kwargs,
    )
    pred_gamma_normal_b = np.asarray(
        gamma_normal_boosted["booster"].predict(
            data=split.x_test.astype(np.float64, copy=False),
            group_data_pred=split.g_test.astype(np.int32, copy=False),
            pred_latent=False,
        )["response_mean"],
        dtype=float,
    ).reshape(-1)
    latent_gamma_normal_b = gamma_normal_boosted["booster"].predict(
        data=split.x_test.astype(np.float64, copy=False),
        group_data_pred=split.g_test.astype(np.int32, copy=False),
        pred_latent=True,
        predict_var=True,
    )
    latent_gamma_normal_b_mean = (
        np.asarray(latent_gamma_normal_b["fixed_effect"], dtype=float).reshape(-1)
        + np.asarray(latent_gamma_normal_b["random_effect_mean"], dtype=float).reshape(-1)
    )
    pred_metrics = summarize_prediction_metrics(split.y_test, pred_gamma_normal_b, split.g_test)
    gamma_normal_boosted_test_nll = gamma_normal_mc_test_nll(
        split.y_test,
        latent_gamma_normal_b_mean,
        np.asarray(latent_gamma_normal_b["random_effect_cov"], dtype=float).reshape(-1),
        gamma_normal_boosted["alpha"],
    )
    rows.append({
        "model": "gamma_normal_gpboost_boosted",
        "alpha_hat": gamma_normal_boosted["alpha"],
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": gamma_normal_boosted_test_nll,
        "test_nll_se": bootstrap_group_se(
            split.g_test,
            lambda idx, _: gamma_normal_mc_test_nll(
                split.y_test[idx],
                latent_gamma_normal_b_mean[idx],
                np.asarray(latent_gamma_normal_b["random_effect_cov"], dtype=float).reshape(-1)[idx],
                gamma_normal_boosted["alpha"],
            ),
        ),
    })

    plain_boost_kwargs = {}
    if USE_OPTUNA:
        best_params_plain, best_rounds_plain = tune_plain_boosting(
            split.x_train,
            split.y_train,
            split.g_train,
        )
        plain_boost_kwargs = {
            "num_boost_round": best_rounds_plain,
            "learning_rate": float(best_params_plain.get("learning_rate", 0.05)),
            "tree_params": {
                "num_leaves": int(best_params_plain.get("num_leaves", 3)),
                "min_data_in_leaf": int(best_params_plain.get("min_data_in_leaf", 10)),
                "max_depth": int(best_params_plain.get("max_depth", 1)),
                "lambda_l2": float(best_params_plain.get("lambda_l2", 0.0)),
                "max_bin": int(best_params_plain.get("max_bin", 255)),
                "feature_fraction": float(best_params_plain.get("feature_fraction", 1.0)),
                "line_search_step_length": bool(best_params_plain.get("line_search_step_length", False)),
            },
        }

    plain_no_group = fit_plain_boosting_global(
        split.x_train,
        split.y_train,
        **plain_boost_kwargs,
    )
    pred_plain_no_group = np.asarray(
        plain_no_group.predict(data=split.x_test.astype(np.float64, copy=False)),
        dtype=float,
    ).reshape(-1)
    pred_metrics = summarize_prediction_metrics(split.y_test, pred_plain_no_group, split.g_test)
    rows.append({
        "model": "boosting_plain_no_group",
        "alpha_hat": np.nan,
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": np.nan,
        "test_nll_se": np.nan,
    })

    x_train_with_group = np.column_stack([split.x_train, split.g_train.astype(float)])
    x_test_with_group = np.column_stack([split.x_test, split.g_test.astype(float)])
    group_col_idx = x_train_with_group.shape[1] - 1
    plain_with_group = fit_plain_boosting_global(
        x_train_with_group,
        split.y_train,
        categorical_feature=[group_col_idx],
        **plain_boost_kwargs,
    )
    pred_plain_with_group = np.asarray(
        plain_with_group.predict(data=x_test_with_group.astype(np.float64, copy=False)),
        dtype=float,
    ).reshape(-1)
    pred_metrics = summarize_prediction_metrics(split.y_test, pred_plain_with_group, split.g_test)
    rows.append({
        "model": "boosting_plain_with_group",
        "alpha_hat": np.nan,
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": np.nan,
        "test_nll_se": np.nan,
    })

    global_fe = fit_global_gamma_fe(split.x_train, split.y_train, ALPHA_GRID)
    f_global = split.x_test @ global_fe["beta"]
    pred_global = global_fe["alpha"] * np.exp(-f_global)
    pred_metrics = summarize_prediction_metrics(split.y_test, pred_global, split.g_test)
    global_fe_test_nll = gamma_fe_test_nll(split.y_test, f_global, global_fe["alpha"])
    rows.append({
        "model": "gamma_fixed_global",
        "alpha_hat": global_fe["alpha"],
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": global_fe_test_nll,
        "test_nll_se": bootstrap_group_se(
            split.g_test,
            lambda idx, _: gamma_fe_test_nll(split.y_test[idx], f_global[idx], global_fe["alpha"]),
        ),
    })

    group_models = fit_groupwise_gamma_fe(
        split.x_train, split.y_train, split.g_train, global_fe["alpha"], global_fe["beta"]
    )
    pred_group = predict_groupwise_gamma_fe(
        split.x_test, split.g_test, global_fe["alpha"], group_models, global_fe["beta"]
    )
    f_group = np.empty(split.x_test.shape[0], dtype=float)
    for i, gid in enumerate(split.g_test):
        beta = group_models.get(int(gid), global_fe["beta"])
        f_group[i] = split.x_test[i] @ beta
    pred_metrics = summarize_prediction_metrics(split.y_test, pred_group, split.g_test)
    group_fe_test_nll = gamma_fe_test_nll(split.y_test, f_group, global_fe["alpha"])
    rows.append({
        "model": "gamma_fixed_groupwise",
        "alpha_hat": global_fe["alpha"],
        "rmse": pred_metrics["rmse"],
        "rmse_se": pred_metrics["rmse_se"],
        "mae": pred_metrics["mae"],
        "mae_se": pred_metrics["mae_se"],
        "corr": pred_metrics["corr"],
        "corr_se": pred_metrics["corr_se"],
        "test_nll": group_fe_test_nll,
        "test_nll_se": bootstrap_group_se(
            split.g_test,
            lambda idx, _: gamma_fe_test_nll(split.y_test[idx], f_group[idx], global_fe["alpha"]),
        ),
    })

    return rows


def run_dataset_models(dataset_name):
    base_dir = os.path.join(os.path.dirname(__file__), "datasets")
    x, y, g = DATASETS[dataset_name](base_dir)
    split = make_split(x, y, g, consecutive=dataset_name in CONSECUTIVE_SPLIT_DATASETS)
    return [{"dataset": dataset_name, **rec} for rec in run_models(split)]


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "datasets")
    out_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    loaded = []
    group_structure_rows = []
    for dataset_name, loader in DATASETS.items():
        print(dataset_name)
        try:
            x, y, g = loader(base_dir)
        except Exception as exc:
            print(f"[warn] skipping {dataset_name}: {exc}")
            continue
        loaded.append((dataset_name, make_split(x, y, g, consecutive=dataset_name in CONSECUTIVE_SPLIT_DATASETS)))
        group_structure_rows.append(group_structure_record(dataset_name, x, y, g))

    group_structure = pd.DataFrame(group_structure_rows)
    print("\nGroup structure before train-test split:")
    print(group_structure.to_string(index=False, formatters={"avg_group_size": "{:.2f}".format}))

    rows = []
    if RUN_DATASETS_IN_SUBPROCESS:
        ctx = mp.get_context("spawn")
        for dataset_name, _ in loaded:
            with ctx.Pool(1) as pool:
                rows.extend(pool.apply(run_dataset_models, (dataset_name,)))
    else:
        for dataset_name, split in loaded:
            for rec in run_models(split):
                rows.append({"dataset": dataset_name, **rec})

    summary = pd.DataFrame(rows)[[
        "model", "dataset",
        "rmse", "rmse_se",
        "mae", "mae_se",
        "corr", "corr_se",
        "test_nll", "test_nll_se",
        "alpha_hat",
    ]]
    summary = summary.sort_values(["dataset", "model"]).reset_index(drop=True)

    metric_specs = [
        ("rmse", "rmse_se", False),
        ("mae", "mae_se", False),
        ("corr", "corr_se", True),
        ("test_nll", "test_nll_se", False),
    ]
    for metric_col, se_col, higher_is_better in metric_specs:
        is_best, is_secondary = highlight_metric_flags(
            summary,
            metric_col=metric_col,
            se_col=se_col,
            higher_is_better=higher_is_better,
        )
        summary[f"{metric_col}_best"] = is_best
        summary[f"{metric_col}_secondary"] = is_secondary

    summary_path = os.path.join(out_dir, SUMMARY_CSV)
    summary.to_csv(summary_path, index=False)
    display = summary[["model", "dataset", "rmse", "mae", "corr", "test_nll", "alpha_hat"]].copy()
    for metric_col, se_col, _ in metric_specs:
        display[metric_col] = [
            apply_highlight_markup(
                format_estimate_with_se(v, se),
                is_best=bool(best),
                is_secondary=bool(secondary),
            )
            for v, se, best, secondary in zip(
                summary[metric_col],
                summary[se_col],
                summary[f"{metric_col}_best"],
                summary[f"{metric_col}_secondary"],
            )
        ]
    display["alpha_hat"] = [
        format_estimate_with_se(v, np.nan)
        for v in summary["alpha_hat"]
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
