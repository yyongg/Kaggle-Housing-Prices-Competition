# -*- coding: utf-8 -*-

"""Functions used for model validation"""

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb


def cross_validate_xgb_stratified(
    X, y, params, n_splits=5, n_estimators=5000, early_stopping_rounds=50
):
    """
    Train an XGBoost model using stratified k-fold cross validation with
    neighbourhood target encoding applied inside each fold.

    Stratification is based on price decile buckets to ensure each fold
    contains a representative spread of cheap, mid-range, and expensive
    houses. This prevents folds from being dominated by unusual price
    distributions which would produce misleading fold scores.

    Neighbourhood encoding is computed exclusively from each fold's training
    rows and applied to the validation rows. This prevents data leakage —
    if encoding were computed from all data before splitting, the validation
    set's encoding would contain information derived from those very houses.

    The following neighbourhood encodings are computed per fold:
        Nbr_MeanPrice       — Mean log sale price per neighbourhood.
        Nbr_StdPrice        — Std of log sale price per neighbourhood.
        Nbr_CoV             — Coefficient of variation (std / mean); captures
                              how unpredictable prices are in each area.
        QualStdInteraction  — OverallQual * Nbr_StdPrice; high quality house
                              in a high-variance neighbourhood commands premium.
        SFStdInteraction    — TotalSF * Nbr_StdPrice; large house in a
                              high-variance neighbourhood shows bigger spread.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame after engineer_features() and fill_missing()
        have been applied and categorical columns cast to category dtype.
        Must contain a "Neighborhood" column for target encoding.
    y : pd.Series
        Log-transformed target (np.log1p(SalePrice)).
    params : dict
        XGBoost hyperparameters to pass to XGBRegressor. Should NOT include
        n_estimators or early_stopping_rounds as these are passed separately.
    n_splits : int, optional
        Number of cross-validation folds. Default is 5.
    n_estimators : int, optional
        Maximum number of boosting rounds per fold. Early stopping will
        halt training before this if the validation score plateaus.
        Default is 5000.
    early_stopping_rounds : int, optional
        Number of rounds without improvement before early stopping triggers.
        Default is 50.

    Returns
    -------
    models : list of tuples
        Each tuple contains (model, nbr_mean, nbr_std, nbr_cov) for one fold.
        The encoding maps are stored alongside the model so predict_ensemble()
        can apply the same neighbourhood encoding at inference time.
    oof_predictions : np.ndarray of shape (len(X),)
        Out-of-fold predictions for every training house. Each house is
        predicted exactly once by a model that never saw it during training.
        The OOF RMSE computed from these is the most honest performance estimate.
    fold_scores : list of float
        Per-fold RMSE scores. High variance across folds indicates the model
        struggles with certain price ranges or neighbourhood types.

    Examples
    --------
    >>> models, oof_preds, fold_scores = cross_validate_xgb_stratified(
    ...     X_train, y_train, best_params, n_splits=5
    ... )
    >>> test_preds_log = predict_ensemble(models, X_final)
    >>> test_preds     = np.expm1(test_preds_log)
    """

    # compute price buckets internally — no global dependency
    price_buckets = pd.qcut(y, q=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, price_buckets)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")

        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # ── neighbourhood target encoding (fit on train fold only) ────────────
        nbr_mean = y_tr.groupby(X_tr["Neighborhood"].astype(str)).mean()
        nbr_std = y_tr.groupby(X_tr["Neighborhood"].astype(str)).std()
        nbr_cov = nbr_std / nbr_mean  # coefficient of variation

        # mean price encoding
        X_tr["Nbr_MeanPrice"] = X_tr["Neighborhood"].astype(str).map(nbr_mean)
        X_val["Nbr_MeanPrice"] = (
            X_val["Neighborhood"].astype(str).map(nbr_mean).fillna(y_tr.mean())
        )

        # std price encoding
        X_tr["Nbr_StdPrice"] = X_tr["Neighborhood"].astype(str).map(nbr_std)
        X_val["Nbr_StdPrice"] = (
            X_val["Neighborhood"].astype(str).map(nbr_std).fillna(y_tr.std())
        )

        # coefficient of variation — relative price uncertainty per neighbourhood
        X_tr["Nbr_CoV"] = X_tr["Neighborhood"].astype(str).map(nbr_cov)
        X_val["Nbr_CoV"] = (
            X_val["Neighborhood"].astype(str).map(nbr_cov).fillna(nbr_cov.mean())
        )

        # quality × std interaction — high qual in high-variance neighbourhood
        X_tr["QualStdInteraction"] = X_tr["OverallQual"] * X_tr["Nbr_StdPrice"]
        X_val["QualStdInteraction"] = X_val["OverallQual"] * X_val["Nbr_StdPrice"]

        # size × std interaction — large house in high-variance neighbourhood
        X_tr["SFStdInteraction"] = X_tr["TotalSF"] * X_tr["Nbr_StdPrice"]
        X_val["SFStdInteraction"] = X_val["TotalSF"] * X_val["Nbr_StdPrice"]

        model = xgb.XGBRegressor(
            **params,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            enable_categorical=True,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

        oof_predictions[val_idx] = model.predict(X_val)
        fold_rmse = root_mean_squared_error(y_val, oof_predictions[val_idx])
        fold_scores.append(fold_rmse)
        models.append((model, nbr_mean, nbr_std, nbr_cov))
        print(
            f"Fold {fold + 1} RMSE: {fold_rmse:.5f} | "
            f"Best iteration: {model.best_iteration}"
        )

    oof_rmse = root_mean_squared_error(y, oof_predictions)
    print(f"\n{'='*45}")
    print(f"Per-fold RMSE: {[round(s, 5) for s in fold_scores]}")
    print(f"Mean RMSE:     {np.mean(fold_scores):.5f}")
    print(f"Std RMSE:      {np.std(fold_scores):.5f}")
    print(f"OOF RMSE:      {oof_rmse:.5f}")
    print(f"{'='*45}")

    return models, oof_predictions, fold_scores


def cross_validate_lgb_stratified(
    X, y, params, n_splits=5, n_estimators=5000, early_stopping_rounds=50
):
    """
    Train an LGB model using stratified k-fold cross validation with
    neighbourhood target encoding applied inside each fold.

    Stratification is based on price decile buckets to ensure each fold
    contains a representative spread of cheap, mid-range, and expensive
    houses. This prevents folds from being dominated by unusual price
    distributions which would produce misleading fold scores.

    Neighbourhood encoding is computed exclusively from each fold's training
    rows and applied to the validation rows. This prevents data leakage —
    if encoding were computed from all data before splitting, the validation
    set's encoding would contain information derived from those very houses.

    The following neighbourhood encodings are computed per fold:
        Nbr_MeanPrice       — Mean log sale price per neighbourhood.
        Nbr_StdPrice        — Std of log sale price per neighbourhood.
        Nbr_CoV             — Coefficient of variation (std / mean); captures
                              how unpredictable prices are in each area.
        QualStdInteraction  — OverallQual * Nbr_StdPrice; high quality house
                              in a high-variance neighbourhood commands premium.
        SFStdInteraction    — TotalSF * Nbr_StdPrice; large house in a
                              high-variance neighbourhood shows bigger spread.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame after engineer_features() and fill_missing()
        have been applied and categorical columns cast to category dtype.
        Must contain a "Neighborhood" column for target encoding.
    y : pd.Series
        Log-transformed target (np.log1p(SalePrice)).
    params : dict
        XGBoost hyperparameters to pass to XGBRegressor. Should NOT include
        n_estimators or early_stopping_rounds as these are passed separately.
    n_splits : int, optional
        Number of cross-validation folds. Default is 5.
    n_estimators : int, optional
        Maximum number of boosting rounds per fold. Early stopping will
        halt training before this if the validation score plateaus.
        Default is 5000.
    early_stopping_rounds : int, optional
        Number of rounds without improvement before early stopping triggers.
        Default is 50.

    Returns
    -------
    models : list of tuples
        Each tuple contains (model, nbr_mean, nbr_std, nbr_cov) for one fold.
        The encoding maps are stored alongside the model so predict_ensemble()
        can apply the same neighbourhood encoding at inference time.
    oof_predictions : np.ndarray of shape (len(X),)
        Out-of-fold predictions for every training house. Each house is
        predicted exactly once by a model that never saw it during training.
        The OOF RMSE computed from these is the most honest performance estimate.
    fold_scores : list of float
        Per-fold RMSE scores. High variance across folds indicates the model
        struggles with certain price ranges or neighbourhood types.

    Examples
    --------
    >>> models, oof_preds, fold_scores = cross_validate_xgb_stratified(
    ...     X_train, y_train, best_params, n_splits=5
    ... )
    >>> test_preds_log = predict_ensemble(models, X_final)
    >>> test_preds     = np.expm1(test_preds_log)
    """

    price_buckets = pd.qcut(y, q=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, price_buckets)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")

        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # same neighbourhood encoding as XGBoost
        nbr_mean = y_tr.groupby(X_tr["Neighborhood"].astype(str)).mean()
        nbr_std = y_tr.groupby(X_tr["Neighborhood"].astype(str)).std()
        nbr_cov = nbr_std / nbr_mean

        X_tr["Nbr_MeanPrice"] = X_tr["Neighborhood"].astype(str).map(nbr_mean)
        X_val["Nbr_MeanPrice"] = (
            X_val["Neighborhood"].astype(str).map(nbr_mean).fillna(y_tr.mean())
        )
        X_tr["Nbr_StdPrice"] = X_tr["Neighborhood"].astype(str).map(nbr_std)
        X_val["Nbr_StdPrice"] = (
            X_val["Neighborhood"].astype(str).map(nbr_std).fillna(y_tr.std())
        )
        X_tr["Nbr_CoV"] = X_tr["Neighborhood"].astype(str).map(nbr_cov)
        X_val["Nbr_CoV"] = (
            X_val["Neighborhood"].astype(str).map(nbr_cov).fillna(nbr_cov.mean())
        )
        X_tr["QualStdInteraction"] = X_tr["OverallQual"] * X_tr["Nbr_StdPrice"]
        X_val["QualStdInteraction"] = X_val["OverallQual"] * X_val["Nbr_StdPrice"]
        X_tr["SFStdInteraction"] = X_tr["TotalSF"] * X_tr["Nbr_StdPrice"]
        X_val["SFStdInteraction"] = X_val["TotalSF"] * X_val["Nbr_StdPrice"]

        # convert category cols to int for LightGBM
        X_tr_lgb = X_tr.copy()
        X_val_lgb = X_val.copy()
        cat_cols = X_tr_lgb.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            X_tr_lgb[col] = X_tr_lgb[col].cat.codes
            X_val_lgb[col] = X_val_lgb[col].cat.codes

        model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, verbose=-1)
        model.fit(
            X_tr_lgb,
            y_tr,
            eval_set=[(X_val_lgb, y_val)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        oof_predictions[val_idx] = model.predict(X_val_lgb)
        fold_rmse = root_mean_squared_error(y_val, oof_predictions[val_idx])
        fold_scores.append(fold_rmse)
        models.append((model, nbr_mean, nbr_std, nbr_cov))
        print(f"Fold {fold + 1} RMSE: {fold_rmse:.5f}")

    oof_rmse = root_mean_squared_error(y, oof_predictions)
    print(f"\n{'='*45}")
    print(f"Per-fold RMSE: {[round(s, 5) for s in fold_scores]}")
    print(f"Mean RMSE:     {np.mean(fold_scores):.5f}")
    print(f"Std RMSE:      {np.std(fold_scores):.5f}")
    print(f"OOF RMSE:      {oof_rmse:.5f}")
    print(f"{'='*45}")

    return models, oof_predictions, fold_scores


def predict_ensemble(models, X_test):
    """
    Generate predictions by averaging across all fold models.

    Applies the same neighbourhood target encoding that was used during
    training — each fold model uses its own encoding maps (fitted only on
    that fold's training data) to encode the test set. Predictions from
    all folds are then averaged, which reduces variance compared to using
    any single fold model.

    Parameters
    ----------
    models : list of tuples
        Output from cross_validate_xgb_stratified(). Each tuple contains
        (model, nbr_mean, nbr_std, nbr_cov).
    X_test : pd.DataFrame
        Feature DataFrame for the test set, processed through the same
        engineer_features(), fill_missing(), drop_cols, and category
        alignment steps as the training data. Must contain a
        "Neighborhood" column and "OverallQual" and "TotalSF" columns
        for the interaction encodings.

    Returns
    -------
    np.ndarray of shape (len(X_test),)
        Averaged predictions in log(1 + SalePrice) space. Apply
        np.expm1() to convert back to dollar values before submitting.

    Examples
    --------
    >>> test_preds_log = predict_ensemble(models, X_final)
    >>> test_preds     = np.expm1(test_preds_log)
    >>> submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds})
    >>> submission.to_csv("submission.csv", index=False)
    """
    preds = []
    for model, nbr_mean, nbr_std, nbr_cov in models:
        X_t = X_test.copy()

        X_t["Nbr_MeanPrice"] = (
            X_t["Neighborhood"].astype(str).map(nbr_mean).fillna(nbr_mean.mean())
        )
        X_t["Nbr_StdPrice"] = (
            X_t["Neighborhood"].astype(str).map(nbr_std).fillna(nbr_std.mean())
        )
        X_t["Nbr_CoV"] = (
            X_t["Neighborhood"].astype(str).map(nbr_cov).fillna(nbr_cov.mean())
        )
        X_t["QualStdInteraction"] = X_t["OverallQual"] * X_t["Nbr_StdPrice"]
        X_t["SFStdInteraction"] = X_t["TotalSF"] * X_t["Nbr_StdPrice"]

        preds.append(model.predict(X_t))
    return np.column_stack(preds).mean(axis=1)


def predict_lgb_ensemble(models, X_test):
    """
    Generate LightGBM predictions by averaging across all fold models.

    Applies the same neighbourhood target encoding used during training,
    converts category columns to integer codes (required by LightGBM),
    and averages predictions across all fold models to reduce variance.

    Parameters
    ----------
    models : list of tuples
        Output from cross_validate_lgb_stratified(). Each tuple contains
        (model, nbr_mean, nbr_std, nbr_cov).
    X_test : pd.DataFrame
        Feature DataFrame processed through engineer_features(),
        fill_missing(), drop_cols, and category alignment. Must contain
        "Neighborhood", "OverallQual", and "TotalSF" columns.

    Returns
    -------
    np.ndarray of shape (len(X_test),)
        Averaged predictions in log(1 + SalePrice) space. Apply
        np.expm1() to convert back to dollar values.
    """
    preds = []
    for model, nbr_mean, nbr_std, nbr_cov in models:
        X_t = X_test.copy()

        # ── neighbourhood encoding ────────────────────────────────────────────
        X_t["Nbr_MeanPrice"] = (
            X_t["Neighborhood"].astype(str).map(nbr_mean).fillna(nbr_mean.mean())
        )
        X_t["Nbr_StdPrice"] = (
            X_t["Neighborhood"].astype(str).map(nbr_std).fillna(nbr_std.mean())
        )
        X_t["Nbr_CoV"] = (
            X_t["Neighborhood"].astype(str).map(nbr_cov).fillna(nbr_cov.mean())
        )
        X_t["QualStdInteraction"] = X_t["OverallQual"] * X_t["Nbr_StdPrice"]
        X_t["SFStdInteraction"] = X_t["TotalSF"] * X_t["Nbr_StdPrice"]

        # ── convert category cols to int codes for LightGBM ──────────────────
        cat_cols = X_t.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            X_t[col] = X_t[col].cat.codes

        preds.append(model.predict(X_t))

    return np.column_stack(preds).mean(axis=1)
