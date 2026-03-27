import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.model_selection import GroupKFold


SEED = 123
RESULTS_DIR = "results"
SUMMARY_CSV = "real_data_experiment_pg_summary.csv"
USE_OPTUNA = True


def import_gpboost():
    try:
        import gpboost as gpb
        return gpb
    except ModuleNotFoundError:
        fallback_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Gamma_Gamma", "GPBoost_full backup", "python-package")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "GPBoost_full backup", "python-package")),
        ]
        for path in fallback_paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        import gpboost as gpb
        return gpb


_GPB = None


def get_gpb():
    global _GPB
    if _GPB is None:
        start = time.time()
        _GPB = import_gpboost()
        print(f"gpboost import finished in {time.time() - start:.1f}s", flush=True)
    return _GPB


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    group_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    group_test: np.ndarray


def compute_rmspe(y_true, mu_pred):
    y_true = np.asarray(y_true, dtype=float)
    mu_pred = np.asarray(mu_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(mu_pred) & (mu_pred > 0)
    if not np.any(valid):
        return np.nan
    residual = ((y_true[valid] - mu_pred[valid]) ** 2) / mu_pred[valid]
    residual = residual[residual > 0]
    return float(np.sqrt(np.mean(residual))) if residual.size else np.nan


def prepend_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])


def extract_response_mean(pred):
    if isinstance(pred, dict):
        for key in ("response_mean", "mu"):
            if key in pred:
                return np.asarray(pred[key], dtype=float).reshape(-1)
        for value in pred.values():
            try:
                return np.asarray(value, dtype=float).reshape(-1)
            except Exception:
                pass
        raise ValueError("Could not extract response predictions from GPBoost output")
    return np.asarray(pred, dtype=float).reshape(-1)


def group_sums(values, groups, n_groups):
    return np.bincount(groups, weights=values, minlength=n_groups).astype(float)


def group_sum_exp(F, groups, n_groups):
    max_by_group = np.full(n_groups, -np.inf, dtype=float)
    np.maximum.at(max_by_group, groups, F)
    shifted = np.exp(F - max_by_group[groups])
    sum_shifted = np.bincount(groups, weights=shifted, minlength=n_groups).astype(float)
    return sum_shifted * np.exp(max_by_group)


def poisson_gamma_nll(y, F, groups, gamma):
    gamma = float(max(gamma, 1e-12))
    _, inv = np.unique(groups, return_inverse=True)
    n_groups = int(inv.max()) + 1
    sum_y = group_sums(y, inv, n_groups)
    sum_exp_f = group_sum_exp(F, inv, n_groups)
    sum_y_f = group_sums(y * F, inv, n_groups)
    sum_log_fact = group_sums(gammaln(y + 1.0), inv, n_groups)
    gamma_prime = gamma + sum_y
    lambda_prime = np.clip(gamma + sum_exp_f, 1e-12, None)
    return float(np.sum(
        -gamma * np.log(gamma)
        + gammaln(gamma)
        - gammaln(gamma_prime)
        + gamma_prime * np.log(lambda_prime)
        - sum_y_f
        + sum_log_fact
    ))


def poisson_normal_nll_mc(y, F, groups, sigma2, mc_samples=2000, seed=SEED):
    sigma2 = float(max(sigma2, 1e-12))
    rng = np.random.default_rng(seed)
    nll = 0.0
    for gid in np.unique(groups):
        mask = groups == gid
        y_i = np.asarray(y[mask], dtype=float)
        F_i = np.asarray(F[mask], dtype=float)
        theta = rng.normal(0.0, np.sqrt(sigma2), size=int(mc_samples))
        eta = theta[:, None] + F_i[None, :]
        log_likelihood = np.sum(
            -np.exp(eta) + eta * y_i[None, :] - gammaln(y_i[None, :] + 1.0),
            axis=1,
        )
        max_log_likelihood = float(np.max(log_likelihood))
        marginal = max_log_likelihood + np.log(np.mean(np.exp(log_likelihood - max_log_likelihood)))
        nll -= marginal
    return float(nll)


def tune_boosting(X, y, groups, likelihood, n_trials=8, n_splits=5, max_rounds=150, early_stopping=15):
    gpb = get_gpb()
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    gp_model = gpb.GPModel(
        group_data=groups.astype(np.int32, copy=False),
        likelihood=likelihood,
        num_data=len(y),
        free_raw_data=False,
    )
    gp_model.set_optim_params({"trace": False})
    n = X.shape[0]
    search_space = {
        "learning_rate": [0.03, 0.10],
        "min_data_in_leaf": [10, max(12, min(25, n // 20 if n >= 20 else 25))],
        "max_depth": [1, 2],
        "num_leaves": [2, 8],
        "lambda_l2": [0.0, 2.0],
        "max_bin": [63, 127],
        "feature_fraction": [0.8, 1.0],
        "line_search_step_length": [False, True],
    }
    started = time.time()
    opt = gpb.tune_pars_TPE_algorithm_optuna(
        search_space=search_space,
        n_trials=int(n_trials),
        X=X.astype(np.float64, copy=False),
        y=y.astype(np.float64, copy=False),
        gp_model=gp_model,
        max_num_boost_round=int(max_rounds),
        early_stopping_rounds=int(early_stopping),
        folds=folds,
        metric="poisson",
        cv_seed=4,
        tpe_seed=1,
        verbose_eval=1,
        use_gp_model_for_validation=True,
        train_gp_model_cov_pars=True,
    )
    best_params = dict(opt.get("best_params", {}))
    best_rounds = int(opt.get("best_num_boost_round", opt.get("best_iter", 100)))
    print(f"tuning finished in {time.time() - started:.1f}s")
    return best_params, best_rounds


def tune_plain_boosting(X, y, groups, n_trials=8, n_splits=5, max_rounds=150, early_stopping=15):
    gpb = get_gpb()
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    n = X.shape[0]
    search_space = {
        "learning_rate": [0.03, 0.10],
        "min_data_in_leaf": [10, max(12, min(25, n // 20 if n >= 20 else 25))],
        "max_depth": [1, 2],
        "num_leaves": [2, 8],
        "lambda_l2": [0.0, 2.0],
        "max_bin": [63, 127],
        "feature_fraction": [0.8, 1.0],
        "line_search_step_length": [False, True],
    }
    started = time.time()
    opt = gpb.tune_pars_TPE_algorithm_optuna(
        search_space=search_space,
        n_trials=int(n_trials),
        X=X.astype(np.float64, copy=False),
        y=y.astype(np.float64, copy=False),
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
    print(f"plain boosting tuning finished in {time.time() - started:.1f}s")
    return best_params, best_rounds


def fit_plain_boosting_global(split, num_estimators=200, n_trials=8):
    gpb = get_gpb()
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 3,
        "min_data_in_leaf": 10,
        "max_depth": 1,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "verbose": -1,
    }
    rounds = int(num_estimators)
    if USE_OPTUNA:
        best_params, rounds = tune_plain_boosting(split.X_train, split.y_train, split.group_train, n_trials=n_trials)
        params.update({
            "learning_rate": float(best_params.get("learning_rate", params["learning_rate"])),
            "num_leaves": int(best_params.get("num_leaves", params["num_leaves"])),
            "min_data_in_leaf": int(best_params.get("min_data_in_leaf", params["min_data_in_leaf"])),
            "max_depth": int(best_params.get("max_depth", params["max_depth"])),
            "lambda_l2": float(best_params.get("lambda_l2", params["lambda_l2"])),
            "max_bin": int(best_params.get("max_bin", params["max_bin"])),
            "feature_fraction": float(best_params.get("feature_fraction", params["feature_fraction"])),
            "line_search_step_length": bool(best_params.get("line_search_step_length", False)),
        })
    booster = gpb.train(
        params=params,
        train_set=gpb.Dataset(
            split.X_train.astype(np.float64, copy=False),
            label=split.y_train.astype(np.float64, copy=False),
        ),
        num_boost_round=rounds,
    )
    mu_pred = np.asarray(
        booster.predict(data=split.X_test.astype(np.float64, copy=False)),
        dtype=float,
    ).reshape(-1)
    return {
        "rmspe": compute_rmspe(split.y_test, mu_pred),
        "nll": np.nan,
    }


def fit_plain_boosting_groupwise(split, num_estimators=200, n_trials=8):
    gpb = get_gpb()
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 3,
        "min_data_in_leaf": 10,
        "max_depth": 1,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "verbose": -1,
    }
    rounds = int(num_estimators)
    if USE_OPTUNA:
        best_params, rounds = tune_plain_boosting(split.X_train, split.y_train, split.group_train, n_trials=n_trials)
        params.update({
            "learning_rate": float(best_params.get("learning_rate", params["learning_rate"])),
            "num_leaves": int(best_params.get("num_leaves", params["num_leaves"])),
            "min_data_in_leaf": int(best_params.get("min_data_in_leaf", params["min_data_in_leaf"])),
            "max_depth": int(best_params.get("max_depth", params["max_depth"])),
            "lambda_l2": float(best_params.get("lambda_l2", params["lambda_l2"])),
            "max_bin": int(best_params.get("max_bin", params["max_bin"])),
            "feature_fraction": float(best_params.get("feature_fraction", params["feature_fraction"])),
            "line_search_step_length": bool(best_params.get("line_search_step_length", False)),
        })
    global_booster = gpb.train(
        params=params,
        train_set=gpb.Dataset(
            split.X_train.astype(np.float64, copy=False),
            label=split.y_train.astype(np.float64, copy=False),
        ),
        num_boost_round=rounds,
    )
    pred = np.asarray(
        global_booster.predict(data=split.X_test.astype(np.float64, copy=False)),
        dtype=float,
    ).reshape(-1)
    for gid in np.unique(split.group_train):
        idx_train = np.flatnonzero(split.group_train == gid)
        idx_test = np.flatnonzero(split.group_test == gid)
        if len(idx_test) == 0:
            continue
        try:
            booster = gpb.train(
                params=params,
                train_set=gpb.Dataset(
                    split.X_train[idx_train].astype(np.float64, copy=False),
                    label=split.y_train[idx_train].astype(np.float64, copy=False),
                ),
                num_boost_round=rounds,
            )
            pred[idx_test] = np.asarray(
                booster.predict(data=split.X_test[idx_test].astype(np.float64, copy=False)),
                dtype=float,
            ).reshape(-1)
        except Exception:
            continue
    return {
        "rmspe": compute_rmspe(split.y_test, pred),
        "nll": np.nan,
    }


def fit_pg_linear(split):
    gpb = get_gpb()
    gp_model = gpb.GPModel(
        group_data=split.group_train.astype(np.int32, copy=False),
        likelihood="poisson_gamma",
        num_data=len(split.y_train),
        free_raw_data=False,
    )
    gp_model.set_optim_params({
        "optimizer_cov": "nelder_mead",
        "optimizer_coef": "nelder_mead",
        "estimate_aux_pars": True,
        "trace": False,
    })
    gp_model.fit(
        y=split.y_train.astype(np.float64, copy=False),
        X=split.X_train.astype(np.float64, copy=False),
    )
    gamma_hat = float(np.asarray(gp_model.get_cov_pars(format_pandas=False), dtype=float).reshape(-1)[0])
    beta_hat = np.asarray(gp_model.get_coef(format_pandas=False), dtype=float).reshape(-1)
    F_train = split.X_train @ beta_hat
    F_test = split.X_test @ beta_hat
    uniq, inv = np.unique(split.group_train, return_inverse=True)
    sum_y = group_sums(split.y_train, inv, len(uniq))
    sum_exp_f = group_sums(np.exp(F_train), inv, len(uniq))
    theta_mean = (sum_y + gamma_hat) / np.clip(sum_exp_f + gamma_hat, 1e-12, None)
    theta_map = {gid: theta_mean[i] for i, gid in enumerate(uniq)}
    mu_pred = np.exp(F_test) * np.array([theta_map.get(gid, 1.0) for gid in split.group_test], dtype=float)
    return {
        "rmspe": compute_rmspe(split.y_test, mu_pred),
        "nll": poisson_gamma_nll(split.y_test, F_test, split.group_test, gamma_hat),
    }


def fit_pn_linear(split):
    gpb = get_gpb()
    gp_model = gpb.GPModel(
        group_data=split.group_train.astype(np.int32, copy=False),
        likelihood="poisson",
        num_data=len(split.y_train),
        free_raw_data=False,
    )
    gp_model.fit(
        y=split.y_train.astype(np.float64, copy=False),
        X=split.X_train.astype(np.float64, copy=False),
    )
    sigma2_hat = float(np.asarray(gp_model.get_cov_pars(format_pandas=False), dtype=float).reshape(-1)[0])
    beta_hat = np.asarray(gp_model.get_coef(format_pandas=False), dtype=float).reshape(-1)
    pred = gp_model.predict(
        predict_response=True,
        y=split.y_train.astype(np.float64, copy=False),
        group_data_pred=split.group_test.astype(np.int32, copy=False),
        X_pred=split.X_test.astype(np.float64, copy=False),
    )
    mu_pred = np.asarray(pred["mu"], dtype=float).reshape(-1)
    F_train = split.X_train @ beta_hat
    F_test = split.X_test @ beta_hat
    return {
        "rmspe": compute_rmspe(split.y_test, mu_pred),
        "nll": poisson_normal_nll_mc(split.y_test, F_test, split.group_test, sigma2_hat),
    }


def train_boosted(split, likelihood, num_estimators=200, n_trials=8):
    gpb = get_gpb()
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 3,
        "min_data_in_leaf": 10,
        "max_depth": 1,
        "lambda_l2": 0.0,
        "max_bin": 255,
        "feature_fraction": 1.0,
        "verbose": -1,
    }
    rounds = int(num_estimators)
    if USE_OPTUNA:
        best_params, rounds = tune_boosting(split.X_train, split.y_train, split.group_train, likelihood, n_trials=n_trials)
        params.update({
            "learning_rate": float(best_params.get("learning_rate", params["learning_rate"])),
            "num_leaves": int(best_params.get("num_leaves", params["num_leaves"])),
            "min_data_in_leaf": int(best_params.get("min_data_in_leaf", params["min_data_in_leaf"])),
            "max_depth": int(best_params.get("max_depth", params["max_depth"])),
            "lambda_l2": float(best_params.get("lambda_l2", params["lambda_l2"])),
            "max_bin": int(best_params.get("max_bin", params["max_bin"])),
            "feature_fraction": float(best_params.get("feature_fraction", params["feature_fraction"])),
            "line_search_step_length": bool(best_params.get("line_search_step_length", False)),
        })
    gp_model = gpb.GPModel(
        group_data=split.group_train.astype(np.int32, copy=False),
        likelihood=likelihood,
        num_data=len(split.y_train),
        free_raw_data=False,
    )
    booster = gpb.train(
        params=params,
        train_set=gpb.Dataset(split.X_train.astype(np.float64, copy=False), label=split.y_train.astype(np.float64, copy=False)),
        gp_model=gp_model,
        use_gp_model_for_validation=False if likelihood == "poisson" else True,
        num_boost_round=rounds,
    )
    F_train = np.asarray(booster.predict(
        data=split.X_train.astype(np.float64, copy=False),
        group_data_pred=split.group_train.astype(np.int32, copy=False),
        pred_latent=True,
        ignore_gp_model=True,
    ), dtype=float).reshape(-1)
    if likelihood == "poisson_gamma":
        F_test = np.asarray(booster.predict(
            data=split.X_test.astype(np.float64, copy=False),
            group_data_pred=split.group_test.astype(np.int32, copy=False),
            pred_latent=True,
            ignore_gp_model=True,
        ), dtype=float).reshape(-1)
        gamma_hat = float(np.asarray(gp_model.get_cov_pars(format_pandas=False), dtype=float).reshape(-1)[0])
        uniq, inv = np.unique(split.group_train, return_inverse=True)
        sum_y = group_sums(split.y_train, inv, len(uniq))
        sum_exp_f = group_sums(np.exp(F_train), inv, len(uniq))
        theta_mean = (sum_y + gamma_hat) / np.clip(sum_exp_f + gamma_hat, 1e-12, None)
        theta_map = {gid: theta_mean[i] for i, gid in enumerate(uniq)}
        mu_pred = np.exp(F_test) * np.array([theta_map.get(gid, 1.0) for gid in split.group_test], dtype=float)
        nll = poisson_gamma_nll(split.y_test, F_test, split.group_test, gamma_hat)
    else:
        pred = booster.predict(
            data=split.X_test.astype(np.float64, copy=False),
            group_data_pred=split.group_test.astype(np.int32, copy=False),
            pred_latent=False,
        )
        mu_pred = extract_response_mean(pred)
        F_test = np.asarray(booster.predict(
            data=split.X_test.astype(np.float64, copy=False),
            group_data_pred=split.group_test.astype(np.int32, copy=False),
            pred_latent=True,
            ignore_gp_model=True,
        ), dtype=float).reshape(-1)
        sigma2_hat = float(np.asarray(gp_model.get_cov_pars(format_pandas=False), dtype=float).reshape(-1)[0])
        nll = poisson_normal_nll_mc(split.y_test, F_test, split.group_test, sigma2_hat)
    return {
        "rmspe": compute_rmspe(split.y_test, mu_pred),
        "nll": nll,
    }


def holdout_one_per_group(data, group_col, rng, min_random=0):
    groups = pd.Categorical(data[group_col]).codes.astype(np.int32)
    data = data.copy()
    data[group_col] = groups
    split = np.zeros(len(data), dtype=np.int32)
    for gid in np.unique(groups):
        idx = np.flatnonzero(groups == gid)
        if len(idx) <= 1:
            test_idx = idx[0]
        elif len(idx) == 2 and min_random == 1:
            test_idx = idx[1]
        else:
            test_idx = idx[int(rng.integers(len(idx)))]
        split[test_idx] = 1
    return data[split == 0].copy(), data[split == 1].copy()


def make_split(train, test, features, target, group):
    return SplitData(
        X_train=train[features].to_numpy(dtype=np.float32),
        y_train=train[target].to_numpy(dtype=np.float32),
        group_train=train[group].astype(np.int32).to_numpy(),
        X_test=test[features].to_numpy(dtype=np.float32),
        y_test=test[target].to_numpy(dtype=np.float32),
        group_test=test[group].astype(np.int32).to_numpy(),
    )


def load_epilepsy():
    data = pd.read_csv("data/epilepsy.csv")
    return make_split(data[data["time"] != 4], data[data["time"] == 4], ["time", "drug", "base", "age"], "y", "id")


def load_cd4():
    data = pd.read_csv("data/cd4.csv")
    data["id"] = pd.Categorical(data["id"]).codes.astype(np.int32)
    data["gender"] = pd.Categorical(data["gender"]).codes.astype(np.int32)
    data["y"] = np.rint(data["cd4"]).astype(np.int32)
    data["zA225z"] = (data["treatment"] == "zA225z").astype(np.int32)
    data["zA400d"] = (data["treatment"] == "zA400d").astype(np.int32)
    data["zX400d"] = (data["treatment"] == "zX400d").astype(np.int32)
    features = ["age", "gender", "week", "zA225z", "zA400d", "zX400d"]
    return make_split(data[data["last_visit"] != 1], data[data["last_visit"] == 1], features, "y", "id")


def load_bolus():
    data = pd.read_csv("data/bolus.csv")
    data["1mg"] = (data["group"] == "1mg").astype(np.int32)
    data["2mg"] = (data["group"] == "2mg").astype(np.int32)
    return make_split(data[data["time"] != 12], data[data["time"] == 12], ["time", "2mg", "1mg"], "y", "id")


def load_owls():
    data = pd.read_csv("data/owls.csv")
    data["id"] = pd.Categorical(data["id"]).codes.astype(np.int32)
    data["FT"] = (data["FoodTreatment"] == "Satiated").astype(np.int32)
    data["SP"] = (data["SexParent"] == "Male").astype(np.int32)
    data["y"] = data["SiblingNegotiation"].astype(np.int32)
    train, test = holdout_one_per_group(data, "id", np.random.default_rng(SEED), min_random=1)
    return make_split(train, test, ["ArrivalTime", "FT", "SP"], "y", "id")


def load_fruits():
    data = pd.read_csv("data/fruits.csv")
    data["amd"] = (data["amd"] == "clipped").astype(np.int32)
    data["rack"] = (data["rack"] == 2).astype(np.int32)
    data["normal"] = (data["status"] == "Normal").astype(np.int32)
    data["trans"] = (data["status"] == "Transplant").astype(np.int32)
    data["petri"] = (data["status"] == "Petri.Plate").astype(np.int32)
    data["y"] = data["total.fruits"]
    train, test = holdout_one_per_group(data, "id", np.random.default_rng(SEED))
    return make_split(train, test, ["nutrient", "amd", "rack", "normal", "trans", "petri"], "y", "id")


def load_claims():
    data = pd.read_csv("data/insurance.csv", sep=";")
    data["Date_last_renewal"] = pd.to_datetime(data["Date_last_renewal"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Date_last_renewal"]).copy()
    for col in ["Length", "Premium", "Power", "Cylinder_capacity", "Value_vehicle", "Weight"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Type_fuel"] = data["Type_fuel"].fillna("NA").astype(str)
    data = pd.get_dummies(data, columns=["Type_fuel"], prefix="Type_fuel")
    numeric_cols = [
        "Distribution_channel", "Seniority", "Policies_in_force", "Max_policies", "Max_products", "Lapse",
        "Payment", "Premium", "Type_risk", "Area", "Second_driver", "Year_matriculation", "Power",
        "Cylinder_capacity", "Value_vehicle", "N_doors", "Length", "Weight",
    ]
    fuel_cols = [c for c in data.columns if c.startswith("Type_fuel_")]
    features = numeric_cols + fuel_cols
    for col in features:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(pd.to_numeric(data[col], errors="coerce").median())
    data["N_claims_year"] = pd.to_numeric(data["N_claims_year"], errors="coerce")
    data = data.dropna(subset=["N_claims_year"]).copy()
    test_idx = data.groupby("ID")["Date_last_renewal"].idxmax()
    return make_split(data.drop(index=test_idx), data.loc[test_idx], features, "N_claims_year", "ID")


DATASETS = {
    "Epilepsy": load_epilepsy,
    "CD4": load_cd4,
    "Bolus": load_bolus,
    "Owls": load_owls,
    "Fruits": load_fruits,
    "Claims": load_claims,
}


MODELS = {
    "poisson_gamma_linear_mixed_effects": fit_pg_linear,
    "poisson_normal_linear_mixed_effects": fit_pn_linear,
    "poisson_gamma_tree_boosting_mixed_effects": lambda split: train_boosted(split, "poisson_gamma"),
    "poisson_normal_tree_boosting_mixed_effects": lambda split: train_boosted(split, "poisson"),
    "boosting_global_plain": fit_plain_boosting_global,
    "boosting_groupwise_plain": fit_plain_boosting_groupwise,
}


def run_all():
    print("run_all started", flush=True)
    rows = []
    for dataset_name, loader in DATASETS.items():
        print(f"dataset start: {dataset_name}", flush=True)
        try:
            split = loader()
            split = SplitData(
                X_train=prepend_intercept(split.X_train),
                y_train=split.y_train,
                group_train=split.group_train,
                X_test=prepend_intercept(split.X_test),
                y_test=split.y_test,
                group_test=split.group_test,
            )
            row = {"dataset": dataset_name}
            for model_name, fit_fn in MODELS.items():
                result = fit_fn(split)
                row[f"{model_name}_rmspe"] = result["rmspe"]
                row[f"{model_name}_nll"] = result["nll"]
                print(model_name, {k: round(v, 4) if np.isfinite(v) else v for k, v in result.items()}, flush=True)
            rows.append(row)
        except Exception as exc:
            print(f"{dataset_name} failed: {exc}", flush=True)
            row = {"dataset": dataset_name}
            for model_name in MODELS:
                row[f"{model_name}_rmspe"] = np.nan
                row[f"{model_name}_nll"] = np.nan
            rows.append(row)
    df = pd.DataFrame(rows).round(4)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, SUMMARY_CSV)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False), flush=True)
    print(out_path, flush=True)
    return df


if __name__ == "__main__":
    print("script entry", flush=True)
    run_all()
