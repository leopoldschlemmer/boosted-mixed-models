import os
import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.model_selection import GroupKFold


SEED = 42
ALPHA_GRID = np.linspace(0.5, 10, 30) 
GG_INIT_ALPHA = 1.0
RESULTS_DIR = "results"
SUMMARY_CSV = "real_data_gg_summary.csv"
USE_OPTUNA = True
ONLINE_RETAIL_MAX_GROUPS = 200
CREDIT_CARD_MAX_GROUPS = 200


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    g_test: np.ndarray


def import_gpboost():
    try:
        import gpboost as gpb
        return gpb
    except ModuleNotFoundError:
        fallback = os.path.join(os.path.dirname(__file__), "GPBoost_full backup", "python-package")
        if fallback not in sys.path:
            sys.path.insert(0, fallback)
        import gpboost as gpb
        return gpb


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred > 0)
    y_t = y_true[valid]
    y_p = y_pred[valid]
    return {
        "rmse": float(np.sqrt(np.mean((y_t - y_p) ** 2))),
        "mae": float(np.mean(np.abs(y_t - y_p))),
        "corr": float(np.corrcoef(y_t, y_p)[0, 1]) if len(y_t) > 2 else np.nan,
    }


def split_within_groups(group, test_size=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for gid in np.unique(group):
        idx = np.flatnonzero(group == gid)
        if len(idx) < 2:
            train_idx.extend(idx.tolist())
            continue
        n_test = max(1, int(np.floor(test_size * len(idx))))
        chosen = rng.choice(idx, size=n_test, replace=False)
        chosen_set = set(chosen.tolist())
        test_idx.extend(chosen.tolist())
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


def fit_gg_boosted(x_train, y_train, g_train, init_alpha=GG_INIT_ALPHA, num_boost_round=200, learning_rate=0.05, tree_params=None):
    gpb = import_gpboost()
    gp_model = gpb.GPModel(
        group_data=g_train.astype(np.int32, copy=False),
        likelihood="gamma_gamma",
        likelihood_additional_param=float(init_alpha),
        num_data=len(y_train),
        free_raw_data=False,
    )
    gp_model.set_optim_params({
        "init_aux_pars": np.array([float(init_alpha)], dtype=float),
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
    gp_model._user_likelihood = "poisson_gamma"
    gp_model.set_optim_params({
        "trace": False,
        "init_aux_pars": np.array([float(init_alpha)], dtype=float),
        "estimate_aux_pars": True,
    })
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
        gp_model=gp_model,
        max_num_boost_round=int(max_rounds),
        early_stopping_rounds=int(early_stopping),
        folds=folds,
        metric="gamma",
        cv_seed=4,
        tpe_seed=1,
        verbose_eval=1,
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


def fit_plain_boosting_global(x_train, y_train, num_boost_round=200, learning_rate=0.05, tree_params=None):
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
            free_raw_data=False,
        ),
        num_boost_round=int(num_boost_round),
    )
    return booster


def fit_plain_boosting_groupwise(x_train, y_train, g_train, num_boost_round=200, learning_rate=0.05, tree_params=None):
    group_models = {}
    global_booster = fit_plain_boosting_global(
        x_train=x_train,
        y_train=y_train,
        num_boost_round=num_boost_round,
        learning_rate=learning_rate,
        tree_params=tree_params,
    )
    for gid in np.unique(g_train):
        idx = np.flatnonzero(g_train == gid)
        try:
            group_models[int(gid)] = fit_plain_boosting_global(
                x_train=x_train[idx],
                y_train=y_train[idx],
                num_boost_round=num_boost_round,
                learning_rate=learning_rate,
                tree_params=tree_params,
            )
        except Exception:
            continue
    return group_models, global_booster


def predict_plain_boosting_groupwise(x, g, group_models, global_booster):
    pred_global = np.asarray(
        global_booster.predict(data=x.astype(np.float64, copy=False)),
        dtype=float,
    ).reshape(-1)
    pred = pred_global.copy()
    for gid, model in group_models.items():
        idx = np.flatnonzero(g == gid)
        if len(idx) == 0:
            continue
        pred[idx] = np.asarray(
            model.predict(data=x[idx].astype(np.float64, copy=False)),
            dtype=float,
        ).reshape(-1)
    return pred


def standardize_columns(x):
    x = x.copy().astype(float)
    for j in range(1, x.shape[1]):
        s = x[:, j].std()
        if s > 0:
            x[:, j] = (x[:, j] - x[:, j].mean()) / s
    return x


def sample_groups(df, group_col, max_groups, seed=SEED):
    if max_groups is None:
        return df
    group_ids = pd.Series(df[group_col].astype(str).unique())
    if len(group_ids) <= max_groups:
        return df
    keep = set(group_ids.sample(n=int(max_groups), random_state=seed).tolist())
    return df[df[group_col].astype(str).isin(keep)].copy()


def load_fremtpl(base_dir, max_rows=120000):
    freq = pd.read_csv(os.path.join(base_dir, "freMTPLfreq.csv"))
    sev = pd.read_csv(os.path.join(base_dir, "freMTPLsev.csv"))
    df = sev.merge(freq, on="PolicyID", how="left")
    df = df[df["ClaimAmount"] > 0].dropna(subset=["Exposure", "CarAge", "DriverAge", "Density", "Region", "Brand", "Gas", "Power"]).copy()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    y = df["ClaimAmount"].astype(float).to_numpy()
    g = pd.factorize(df["Region"])[0].astype(np.int32)
    x = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            df[["Exposure", "CarAge", "DriverAge", "Density"]].astype(float),
            pd.get_dummies(df[["Power", "Brand", "Gas"]].astype(str), drop_first=True),
        ],
        axis=1,
    ).astype(float).to_numpy()
    return standardize_columns(x), y, g


def load_sandp500(base_dir, max_rows=140000):
    df = pd.read_csv(os.path.join(base_dir, "all_stocks_5yr.csv"))
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "Name"])
    df = df[df["close"] > 0].copy()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    y = df["close"].astype(float).to_numpy()
    g = pd.factorize(df["Name"])[0].astype(np.int32)
    x = np.column_stack([
        np.ones(len(df), dtype=float),
        df["open"].astype(float).to_numpy(),
        df["high"].astype(float).to_numpy(),
        df["low"].astype(float).to_numpy(),
        np.log1p(df["volume"].astype(float).to_numpy()),
    ])
    return standardize_columns(x), y, g


def load_us_fundamentals(base_dir, max_rows=120000):
    df = pd.read_csv(os.path.join(base_dir, "assets_forecast_panel.csv"))
    df = df.dropna(subset=["company_name", "feature_year", "y_assets_next"]).copy()
    df = df[df["y_assets_next"] > 0].copy()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
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
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    year = pd.to_numeric(df["feature_year"], errors="coerce").astype(float)
    year_norm = (year - year.mean()) / max(year.std(), 1.0)
    x = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            pd.Series(year_norm, index=df.index, name="feature_year_norm"),
            x_df,
        ],
        axis=1,
    ).astype(float).to_numpy()
    return standardize_columns(x), y, g


def load_online_retail(base_dir, max_rows=120000, max_groups=ONLINE_RETAIL_MAX_GROUPS):
    df = pd.read_csv(os.path.join(base_dir, "online_retail.csv"))
    df = df.dropna(subset=["CustomerID", "Quantity", "UnitPrice", "InvoiceDate", "Country"]).copy()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["sales_amount"] = df["Quantity"] * df["UnitPrice"]
    df = df[df["sales_amount"] > 0].copy()
    df = sample_groups(df, "CustomerID", max_groups=max_groups, seed=SEED)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.loc[dt.notna()].copy()
    dt = dt.loc[df.index]
    y = df["sales_amount"].astype(float).to_numpy()
    g = pd.factorize(df["CustomerID"].astype(str))[0].astype(np.int32)
    x = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            np.log1p(df[["UnitPrice", "Quantity"]].astype(float)).rename(
                columns={"UnitPrice": "log_unitprice", "Quantity": "log_quantity"}
            ),
            pd.Series(dt.dt.hour.astype(float), index=df.index, name="invoice_hour"),
            pd.Series(dt.dt.dayofweek.astype(float), index=df.index, name="invoice_dow"),
            pd.Series(dt.dt.month.astype(float), index=df.index, name="invoice_month"),
            pd.get_dummies(df[["Country"]].astype(str), drop_first=True),
        ],
        axis=1,
    ).astype(float).to_numpy()
    return standardize_columns(x), y, g


def load_credit_card_transactions(base_dir, max_rows=120000, max_groups=CREDIT_CARD_MAX_GROUPS):
    df = pd.read_csv(os.path.join(base_dir, "credit_card_transactions.csv"))
    cols = [
        "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
        "state", "city_pop", "lat", "long", "merch_lat", "merch_long",
    ]
    df = df.dropna(subset=cols).copy()
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df["city_pop"] = pd.to_numeric(df["city_pop"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["merch_lat"] = pd.to_numeric(df["merch_lat"], errors="coerce")
    df["merch_long"] = pd.to_numeric(df["merch_long"], errors="coerce")
    df = df[df["amt"] > 0].copy()
    df = sample_groups(df, "cc_num", max_groups=max_groups, seed=SEED)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=SEED)
    dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df = df.loc[dt.notna()].copy()
    dt = dt.loc[df.index]
    y = df["amt"].astype(float).to_numpy()
    g = pd.factorize(df["cc_num"].astype(str))[0].astype(np.int32)
    x = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="intercept"),
            pd.Series(np.log1p(df["city_pop"].astype(float)), index=df.index, name="log_city_pop"),
            pd.Series(dt.dt.hour.astype(float), index=df.index, name="trans_hour"),
            pd.Series(dt.dt.dayofweek.astype(float), index=df.index, name="trans_dow"),
            pd.Series(dt.dt.month.astype(float), index=df.index, name="trans_month"),
            df[["lat", "long", "merch_lat", "merch_long"]].astype(float),
            pd.get_dummies(df[["category", "gender", "state"]].astype(str), drop_first=True),
        ],
        axis=1,
    ).astype(float).to_numpy()
    return standardize_columns(x), y, g


DATASETS = {
    "fremtpl": lambda base: load_fremtpl(os.path.join(base, "fremtpl-french-motor-tpl-insurance-claims")),
    "sandp500": lambda base: load_sandp500(os.path.join(base, "sandp500")),
    "us_fundamentals_assets": lambda base: load_us_fundamentals(os.path.join(base, "us-stocks-fundamentals")),
    "online_retail": lambda base: load_online_retail(os.path.join(base, "online-retail-dataset")),
    "credit_card_transactions": lambda base: load_credit_card_transactions(os.path.join(base, "credit-card-transactions-dataset")),
}


def make_split(x, y, g):
    train_idx, test_idx = split_within_groups(g, test_size=0.2, seed=SEED)
    return SplitData(x[train_idx], y[train_idx], g[train_idx], x[test_idx], y[test_idx], g[test_idx])


def run_models(split):
    rows = []

    gg_linear = fit_gg_linear(split.x_train, split.y_train, split.g_train)
    f_test_linear = split.x_test @ gg_linear["beta"]
    shape, rate, id_to_idx = posterior_group_stats(
        split.g_train,
        split.y_train,
        split.x_train @ gg_linear["beta"],
        gg_linear["alpha"],
        gg_linear["delta"],
    )
    pred = predict_grouped_from_beta(
        split.x_test, split.g_test, gg_linear["beta"], gg_linear["alpha"], shape, rate, id_to_idx, gg_linear["delta"]
    )
    rows.append({
        "model": "gg_gpboost",
        "alpha_hat": gg_linear["alpha"],
        "rmse": metrics(split.y_test, pred)["rmse"],
        "mae": metrics(split.y_test, pred)["mae"],
        "corr": metrics(split.y_test, pred)["corr"],
        "test_nll": grouped_gg_test_nll(split.y_test, split.g_test, f_test_linear, gg_linear["alpha"], gg_linear["delta"]),
    })

    boost_kwargs = {}
    if USE_OPTUNA:
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
    shape_b, rate_b, id_to_idx_b = posterior_group_stats(
        split.g_train, split.y_train, gg_boosted["f_train"], gg_boosted["alpha"], gg_boosted["delta"]
    )
    pred_b = predict_grouped_from_f(
        f_test, split.g_test, gg_boosted["alpha"], shape_b, rate_b, id_to_idx_b, gg_boosted["delta"]
    )
    rows.append({
        "model": "gg_gpboost_boosted",
        "alpha_hat": gg_boosted["alpha"],
        "rmse": metrics(split.y_test, pred_b)["rmse"],
        "mae": metrics(split.y_test, pred_b)["mae"],
        "corr": metrics(split.y_test, pred_b)["corr"],
        "test_nll": grouped_gg_test_nll(split.y_test, split.g_test, f_test, gg_boosted["alpha"], gg_boosted["delta"]),
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

    plain_global = fit_plain_boosting_global(
        split.x_train,
        split.y_train,
        **plain_boost_kwargs,
    )
    pred_plain_global = np.asarray(
        plain_global.predict(data=split.x_test.astype(np.float64, copy=False)),
        dtype=float,
    ).reshape(-1)
    rows.append({
        "model": "boosting_global_plain",
        "alpha_hat": np.nan,
        "rmse": metrics(split.y_test, pred_plain_global)["rmse"],
        "mae": metrics(split.y_test, pred_plain_global)["mae"],
        "corr": metrics(split.y_test, pred_plain_global)["corr"],
        "test_nll": np.nan,
    })

    plain_group_models, plain_group_global_fallback = fit_plain_boosting_groupwise(
        split.x_train,
        split.y_train,
        split.g_train,
        **plain_boost_kwargs,
    )
    pred_plain_group = predict_plain_boosting_groupwise(
        split.x_test,
        split.g_test,
        plain_group_models,
        plain_group_global_fallback,
    )
    rows.append({
        "model": "boosting_groupwise_plain",
        "alpha_hat": np.nan,
        "rmse": metrics(split.y_test, pred_plain_group)["rmse"],
        "mae": metrics(split.y_test, pred_plain_group)["mae"],
        "corr": metrics(split.y_test, pred_plain_group)["corr"],
        "test_nll": np.nan,
    })

    global_fe = fit_global_gamma_fe(split.x_train, split.y_train, ALPHA_GRID)
    f_global = split.x_test @ global_fe["beta"]
    pred_global = global_fe["alpha"] * np.exp(-f_global)
    rows.append({
        "model": "gamma_fixed_global",
        "alpha_hat": global_fe["alpha"],
        "rmse": metrics(split.y_test, pred_global)["rmse"],
        "mae": metrics(split.y_test, pred_global)["mae"],
        "corr": metrics(split.y_test, pred_global)["corr"],
        "test_nll": gamma_fe_test_nll(split.y_test, f_global, global_fe["alpha"]),
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
    rows.append({
        "model": "gamma_fixed_groupwise",
        "alpha_hat": global_fe["alpha"],
        "rmse": metrics(split.y_test, pred_group)["rmse"],
        "mae": metrics(split.y_test, pred_group)["mae"],
        "corr": metrics(split.y_test, pred_group)["corr"],
        "test_nll": gamma_fe_test_nll(split.y_test, f_group, global_fe["alpha"]),
    })

    return rows


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "datasets")
    out_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for dataset_name, loader in DATASETS.items():
        print(dataset_name)
        x, y, g = loader(base_dir)
        split = make_split(x, y, g)
        for rec in run_models(split):
            rows.append({"dataset": dataset_name, **rec})

    summary = pd.DataFrame(rows)[["model", "dataset", "rmse", "mae", "corr", "test_nll", "alpha_hat"]]
    summary = summary.sort_values(["dataset", "model"]).reset_index(drop=True)
    summary_path = os.path.join(out_dir, SUMMARY_CSV)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
