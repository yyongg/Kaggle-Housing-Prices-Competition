# -*- coding: utf-8 -*-

"""Functions used for feature engineering process"""

import numpy as np


def engineer_features(df):
    """
    Apply feature engineering transformations to the Ames Housing dataset.

    Creates new features that XGBoost cannot derive on its own, including
    arithmetic combinations across columns, ordinal encodings with domain
    knowledge, interaction terms, and flag features for specific house
    characteristics. All engineered columns are cast to float at the end
    to ensure compatibility with XGBoost's enable_categorical=True setting.

    Note: This function should be called BEFORE fill_missing(), as some
    engineered features (e.g. GarageAge, TotalSF) depend on raw columns
    that may contain nulls. Call fill_missing() afterward to handle any
    remaining nulls in both raw and engineered columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature DataFrame. Must contain the standard Ames Housing
        columns as provided by the Kaggle competition dataset. Should
        NOT contain the SalePrice target column.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with all original columns retained plus
        the following engineered feature groups added:

        Neighbourhood features:
            NbrTier         — Manual price tier (1-5) based on median sale price
                              per neighbourhood, derived from EDA.

        Area features:
            TotalSF         — Total indoor square footage (basement + floors),
                              log-transformed.
            TotalPorchSF    — Sum of all porch area types.
            LivLotRatio     — Ratio of above-grade living area to lot size.

        Bathroom features:
            TotalBathrooms  — Weighted bathroom count (full=1, half=0.5).

        Age features:
            HouseAge        — Years from build to sale.
            HouseAge2       — Squared house age, captures non-linear depreciation.
            RemodAge        — Years since last remodel to sale.
            RemodAge2       — Squared remodel age.
            IsRemodeled     — 1 if house was ever remodelled, else 0.
            GarageAge       — Years from garage build to sale.

        Quality interactions:
            OverallScore    — OverallQual * OverallCond.
            ScoreSF         — OverallScore * TotalSF.
            QualSF          — OverallQual * TotalSF (log-transformed).
            QualGarageArea  — OverallQual * GarageArea.
            QualTotalBath   — OverallQual * TotalBathrooms.
            CondSF          — OverallCond * TotalSF.
            QualCondGap     — OverallQual - OverallCond; large gap indicates a
                              house that appears better than its condition.
            LowCondHighQual — Flag for high quality but poor condition houses.

        Neighbourhood interactions:
            QualNbrTier     — OverallQual * NbrTier.
            CondNbrTier     — OverallCond * NbrTier.
            SFNbrTier       — TotalSF * NbrTier (log-transformed).
            ExterCondNbr    — ExterCondScore * NbrTier; run-down house in good area.

        Ordinal quality mappings (Ex=5, Gd=4, TA=3, Fa=2, Po=1):
            KitchenQual     — Kitchen quality as ordered numeric.
            BsmtQual        — Basement quality as ordered numeric.
            FireplaceQu     — Fireplace quality as ordered numeric.
            ExterCondScore  — Exterior condition as ordered numeric.
            BsmtCondScore   — Basement condition as ordered numeric.

        Distressed property flags:
            IsOldLowQual    — Old house (>50yr) with low quality (<=5).
            IsOldLowCond    — Old house (>50yr) with poor condition (<=4).
            LowQualOldNbr   — IsOldLowQual in a tier-1 neighbourhood.
            OldStory1pt5    — Old 1.5-story house (historically cheap type).

        Functional impairment:
            FunctionalScore        — Typ=7 down to Sal=0; impaired houses sell
                                     at steep discounts.
            IsFunctionallyImpaired — Flag for FunctionalScore <= 5.
            FunctionalQualCombo    — FunctionalScore * OverallQual.

        Sale condition flags:
            IsAbnormal          — Abnormal sale condition.
            IsPartial           — Partial sale (new construction incomplete).
            IsFamily            — Family sale (below-market by definition).
            IsAdjLand           — Adjacent land sale.
            NonMarketSale       — Any of Abnorml/Family/AdjLand.
            AbnormalQual        — IsAbnormal * OverallQual.
            AbnormalSF          — IsAbnormal * TotalSF.
            AbnormalNbr         — IsAbnormal * NbrTier.
            FamilySaleNbr       — IsFamily * NbrTier.
            FamilySaleQual      — IsFamily * OverallQual.
            FamilySaleDiscount  — IsFamily * OverallQual * NbrTier; captures
                                  that family discounts are larger on better houses
                                  in better neighbourhoods.
            FamilyHighQual      — Family sale on a quality>=6 house.
            NonMarketSaleNbr    — NonMarketSale * NbrTier.
            PartialLowCond      — Partial sale with condition<=5 (new build problems).
            NewBuildLowCond     — Partial sale with condition<7.
            NewBuildCondGap     — IsPartial * (10 - OverallCond).

        Non-standard sale types:
            IsNonStandardSale   — SaleType in [Alloca, COD, ConLw, ConLD, Oth].
            NonStandardNbr      — IsNonStandardSale * NbrTier.
            NonStandardQual     — IsNonStandardSale * OverallQual.

        Garage features:
            NoGarage        — Flag for houses with no garage (GarageCars==0).
            NoGarageNbr     — NoGarage * NbrTier; penalty larger in nicer areas.
            NoGarageQual    — NoGarage * OverallQual.

        Bedroom features:
            LowBedroom      — Flag for houses with <=1 bedroom above grade.
            BedroomPerRoom  — Bedrooms as a fraction of total rooms above grade.

        MSSubClass type features:
            IsOldSubClass   — Old/cheap house types [30, 45, 50, 190].
            OldSubClassNbr  — IsOldSubClass * NbrTier.
            OldSubClassCond — IsOldSubClass * OverallCond.

        Low-tier neighbourhood interactions:
            LowNbrOldStyle  — Tier-1 neighbourhood with old house style.
            LowNbrLowCond   — Tier-1 neighbourhood with condition<=5.
            LowNbrOldSub    — Tier-1 neighbourhood with old MSSubClass.

        Multifamily features:
            MultiFamily     — Flag for non-single-family building types.
            MultiNbrTier    — MultiFamily * NbrTier.

        Lot features:
            LotShapeScore   — Lot regularity (Reg=4 down to IR3=1).

        Luxury tiers:
            IsLuxury        — OverallQual >= 9.
            IsUltraLuxury   — OverallQual == 10.
            LuxurySF        — IsLuxury * TotalSF.
            UltraLuxurySF   — IsUltraLuxury * TotalSF.
            LuxuryNbr       — IsLuxury * NbrTier.
            UltraLuxuryNbr  — IsUltraLuxury * NbrTier.
            LuxuryGarage    — IsLuxury * GarageArea.

        OldTown-specific features:
            IsOldTown           — Flag for OldTown neighbourhood.
            OldTownAge          — IsOldTown * HouseAge.
            OldTownQual         — IsOldTown * OverallQual.
            OldTownRemodeled    — IsOldTown * IsRemodeled.
            OldTownRecentRemod  — OldTown house remodelled within last 10 years.

    Examples
    --------
    >>> X_train = engineer_features(X_train)
    >>> X_test  = engineer_features(X_test)
    >>> X_train = fill_missing(X_train)
    >>> X_test  = fill_missing(X_test)
    """
    df = df.copy()

    # ── Rare neighbourhood grouping ───────────────────────────────────────────
    rare_nbr = ["Blueste", "NPkVill"]
    df["Neighborhood"] = df["Neighborhood"].replace(rare_nbr, "Other")

    # ── Neighbourhood tier ────────────────────────────────────────────────────
    top_nbr = ["NridgHt", "NoRidge", "StoneBr"]
    midhigh_nbr = ["Timber", "Veenker", "Somerst", "ClearCr", "Crawfor"]
    mid_nbr = ["CollgCr", "Blmngtn", "Gilbert", "SawyerW"]
    lowmid_nbr = ["NAmes", "Mitchel", "SWISU", "NWAmes"]
    low_nbr = ["IDOTRR", "BrkSide", "Edwards", "OldTown", "BrDale", "Sawyer", "MeadowV"]

    df["NbrTier"] = 3
    df.loc[df["Neighborhood"].isin(top_nbr), "NbrTier"] = 5
    df.loc[df["Neighborhood"].isin(midhigh_nbr), "NbrTier"] = 4
    df.loc[df["Neighborhood"].isin(mid_nbr), "NbrTier"] = 3
    df.loc[df["Neighborhood"].isin(lowmid_nbr), "NbrTier"] = 2
    df.loc[df["Neighborhood"].isin(low_nbr), "NbrTier"] = 1

    # ── Area features ─────────────────────────────────────────────────────────
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )
    df["LivLotRatio"] = df["GrLivArea"] / df["LotArea"].replace(0, np.nan)

    # ── Bathroom features ─────────────────────────────────────────────────────
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # ── Age & remodel features ────────────────────────────────────────────────
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["HouseAge2"] = df["HouseAge"] ** 2
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["RemodAge2"] = df["RemodAge"] ** 2
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"].replace(0, np.nan)

    # ── Ordinal quality mappings ──────────────────────────────────────────────
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    df["KitchenQual"] = df["KitchenQual"].map(qual_map).fillna(0)
    df["BsmtQual"] = df["BsmtQual"].map(qual_map).fillna(0)
    df["FireplaceQu"] = df["FireplaceQu"].map(qual_map).fillna(0)

    # ── Condition mappings ────────────────────────────────────────────────────
    cond_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    df["ExterCondScore"] = df["ExterCond"].map(cond_map).fillna(3)
    df["BsmtCondScore"] = df["BsmtCond"].map(cond_map).fillna(3)

    # ── Overall score ─────────────────────────────────────────────────────────
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    df["ScoreSF"] = df["OverallScore"] * df["TotalSF"]

    # ── Quality * size interactions ───────────────────────────────────────────
    df["QualSF"] = df["OverallQual"] * df["TotalSF"]
    df["QualGarageArea"] = df["OverallQual"] * df["GarageArea"]
    df["QualTotalBath"] = df["OverallQual"] * df["TotalBathrooms"]
    df["CondSF"] = df["OverallCond"] * df["TotalSF"]

    # ── Neighbourhood * quality interactions ──────────────────────────────────
    df["QualNbrTier"] = df["OverallQual"] * df["NbrTier"]
    df["CondNbrTier"] = df["OverallCond"] * df["NbrTier"]
    df["SFNbrTier"] = df["TotalSF"] * df["NbrTier"]
    df["ExterCondNbr"] = df["ExterCondScore"] * df["NbrTier"]

    # ── Old/distressed property flags ─────────────────────────────────────────
    df["IsOldLowQual"] = ((df["HouseAge"] > 50) & (df["OverallQual"] <= 5)).astype(int)
    df["IsOldLowCond"] = ((df["HouseAge"] > 50) & (df["OverallCond"] <= 4)).astype(int)
    df["LowQualOldNbr"] = df["IsOldLowQual"] * (df["NbrTier"] == 1).astype(int)
    df["OldStory1pt5"] = (
        df["HouseStyle"].isin(["1.5Fin", "1.5Unf"]) & (df["HouseAge"] > 50)
    ).astype(int)

    # ── Condition gap features ────────────────────────────────────────────────
    df["QualCondGap"] = df["OverallQual"] - df["OverallCond"]
    df["LowCondHighQual"] = (
        (df["OverallCond"] <= 4) & (df["OverallQual"] >= 6)
    ).astype(int)

    # ── Functional impairment ─────────────────────────────────────────────────
    functional_map = {
        "Typ": 7,
        "Min1": 6,
        "Min2": 5,
        "Mod": 4,
        "Maj1": 3,
        "Maj2": 2,
        "Sev": 1,
        "Sal": 0,
    }
    df["FunctionalScore"] = df["Functional"].map(functional_map).fillna(7)
    df["IsFunctionallyImpaired"] = (df["FunctionalScore"] <= 5).astype(int)
    df["FunctionalQualCombo"] = df["FunctionalScore"] * df["OverallQual"]

    # ── Sale condition flags ──────────────────────────────────────────────────
    df["IsAbnormal"] = (df["SaleCondition"] == "Abnorml").astype(int)
    df["IsPartial"] = (df["SaleCondition"] == "Partial").astype(int)
    df["IsFamily"] = (df["SaleCondition"] == "Family").astype(int)
    df["IsAdjLand"] = (df["SaleCondition"] == "AdjLand").astype(int)
    df["NonMarketSale"] = (
        df["SaleCondition"].isin(["Abnorml", "Family", "AdjLand"]).astype(int)
    )

    df["AbnormalQual"] = df["IsAbnormal"] * df["OverallQual"]
    df["AbnormalSF"] = df["IsAbnormal"] * df["TotalSF"]
    df["AbnormalNbr"] = df["IsAbnormal"] * df["NbrTier"]
    df["FamilySaleNbr"] = df["IsFamily"] * df["NbrTier"]
    df["FamilySaleQual"] = df["IsFamily"] * df["OverallQual"]
    df["NonMarketSaleNbr"] = df["NonMarketSale"] * df["NbrTier"]
    df["PartialLowCond"] = df["IsPartial"] * (df["OverallCond"] <= 5).astype(int)
    df["FamilySaleDiscount"] = df["IsFamily"] * df["OverallQual"] * df["NbrTier"]
    df["FamilyHighQual"] = df["IsFamily"] * (df["OverallQual"] >= 6).astype(int)

    # ── Non-standard sale types ───────────────────────────────────────────────
    df["IsNonStandardSale"] = (
        df["SaleType"].isin(["Alloca", "COD", "ConLw", "ConLD", "Oth"]).astype(int)
    )
    df["NonStandardNbr"] = df["IsNonStandardSale"] * df["NbrTier"]
    df["NonStandardQual"] = df["IsNonStandardSale"] * df["OverallQual"]

    # ── New build with problems ───────────────────────────────────────────────
    df["NewBuildLowCond"] = (df["IsPartial"] & (df["OverallCond"] < 7)).astype(int)
    df["NewBuildCondGap"] = df["IsPartial"] * (10 - df["OverallCond"])

    # ── Garage absence penalty ────────────────────────────────────────────────
    df["NoGarage"] = (df["GarageCars"] == 0).astype(int)
    df["NoGarageNbr"] = df["NoGarage"] * df["NbrTier"]
    df["NoGarageQual"] = df["NoGarage"] * df["OverallQual"]

    # ── Bedroom features ──────────────────────────────────────────────────────
    df["LowBedroom"] = (df["BedroomAbvGr"] <= 1).astype(int)
    df["BedroomPerRoom"] = df["BedroomAbvGr"] / df["TotRmsAbvGrd"].replace(0, np.nan)

    # ── MSSubClass type features ──────────────────────────────────────────────
    df["IsOldSubClass"] = df["MSSubClass"].isin([30, 45, 50, 190]).astype(int)
    df["OldSubClassNbr"] = df["IsOldSubClass"] * df["NbrTier"]
    df["OldSubClassCond"] = df["IsOldSubClass"] * df["OverallCond"]

    # ── Low tier neighbourhood interactions ───────────────────────────────────
    df["LowNbrOldStyle"] = (
        (df["NbrTier"] == 1)
        & df["HouseStyle"].isin(["1.5Fin", "1.5Unf", "2.5Fin", "2.5Unf"])
    ).astype(int)
    df["LowNbrLowCond"] = ((df["NbrTier"] == 1) & (df["OverallCond"] <= 5)).astype(int)
    df["LowNbrOldSub"] = (
        (df["NbrTier"] == 1) & df["MSSubClass"].isin([30, 45, 50, 190])
    ).astype(int)

    # ── Multifamily features ──────────────────────────────────────────────────
    df["MultiFamily"] = (
        df["BldgType"].isin(["2fmCon", "Duplex", "Twnhs", "TwnhsE"]).astype(int)
    )
    df["MultiNbrTier"] = df["MultiFamily"] * df["NbrTier"]

    # ── Lot shape ─────────────────────────────────────────────────────────────
    df["LotShapeScore"] = (
        df["LotShape"].map({"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}).fillna(3)
    )

    # ── Luxury tiers ──────────────────────────────────────────────────────────
    df["IsLuxury"] = (df["OverallQual"] >= 9).astype(int)
    df["IsUltraLuxury"] = (df["OverallQual"] == 10).astype(int)
    df["LuxurySF"] = df["IsLuxury"] * df["TotalSF"]
    df["UltraLuxurySF"] = df["IsUltraLuxury"] * df["TotalSF"]
    df["LuxuryNbr"] = df["IsLuxury"] * df["NbrTier"]
    df["UltraLuxuryNbr"] = df["IsUltraLuxury"] * df["NbrTier"]
    df["LuxuryGarage"] = df["IsLuxury"] * df["GarageArea"]

    # ── OldTown specific features ─────────────────────────────────────────────
    df["IsOldTown"] = (df["Neighborhood"] == "OldTown").astype(int)
    df["OldTownAge"] = df["IsOldTown"] * df["HouseAge"]
    df["OldTownQual"] = df["IsOldTown"] * df["OverallQual"]
    df["OldTownRemodeled"] = df["IsOldTown"] * df["IsRemodeled"]
    df["OldTownRecentRemod"] = df["IsOldTown"] * (df["RemodAge"] < 10).astype(int)

    # ── Log transform skewed numeric features ─────────────────────────────────
    skewed_cols = [
        "LotArea",
        "TotalSF",
        "GrLivArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "LivLotRatio",
        "QualSF",
        "SFNbrTier",
    ]
    for col in skewed_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ── Cast all engineered cols to float ─────────────────────────────────────
    new_numeric_cols = [
        "NbrTier",
        "TotalSF",
        "TotalPorchSF",
        "LivLotRatio",
        "TotalBathrooms",
        "HouseAge",
        "HouseAge2",
        "RemodAge",
        "RemodAge2",
        "IsRemodeled",
        "GarageAge",
        "KitchenQual",
        "BsmtQual",
        "FireplaceQu",
        "ExterCondScore",
        "BsmtCondScore",
        "OverallScore",
        "ScoreSF",
        "QualSF",
        "QualGarageArea",
        "QualTotalBath",
        "CondSF",
        "QualNbrTier",
        "CondNbrTier",
        "SFNbrTier",
        "ExterCondNbr",
        "IsOldLowQual",
        "IsOldLowCond",
        "LowQualOldNbr",
        "OldStory1pt5",
        "QualCondGap",
        "LowCondHighQual",
        "FunctionalScore",
        "IsFunctionallyImpaired",
        "FunctionalQualCombo",
        "IsAbnormal",
        "IsPartial",
        "IsFamily",
        "IsAdjLand",
        "NonMarketSale",
        "AbnormalQual",
        "AbnormalSF",
        "AbnormalNbr",
        "FamilySaleNbr",
        "FamilySaleQual",
        "NonMarketSaleNbr",
        "PartialLowCond",
        "FamilySaleDiscount",
        "FamilyHighQual",
        "IsNonStandardSale",
        "NonStandardNbr",
        "NonStandardQual",
        "NewBuildLowCond",
        "NewBuildCondGap",
        "NoGarage",
        "NoGarageNbr",
        "NoGarageQual",
        "LowBedroom",
        "BedroomPerRoom",
        "IsOldSubClass",
        "OldSubClassNbr",
        "OldSubClassCond",
        "LowNbrOldStyle",
        "LowNbrLowCond",
        "LowNbrOldSub",
        "MultiFamily",
        "MultiNbrTier",
        "LotShapeScore",
        "IsLuxury",
        "IsUltraLuxury",
        "LuxurySF",
        "UltraLuxurySF",
        "LuxuryNbr",
        "UltraLuxuryNbr",
        "LuxuryGarage",
        "IsOldTown",
        "OldTownAge",
        "OldTownQual",
        "OldTownRemodeled",
        "OldTownRecentRemod",
    ]
    for col in new_numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def fill_missing(df):
    """
    Impute missing values in the Ames Housing dataset using domain knowledge.

    Handles two distinct types of missingness that require different treatment:

    1. Structural missingness — the feature is null because the property
       physically lacks that feature (e.g. no garage, no basement, no pool).
       Categorical columns of this type are filled with the string "None",
       which becomes its own learnable category. Numeric columns are filled
       with 0, which correctly represents the absence of that feature.

    2. Data entry errors — a small number of columns have nulls in the test
       set that never appear in training (e.g. MSZoning has 4 test nulls,
       KitchenQual has 1). XGBoost never learns a default split direction for
       these columns, so without imputation it falls back to an arbitrary
       default for those houses, producing unreliable predictions. Filling
       with "None" or the most reasonable default prevents this.

    This function must be called AFTER engineer_features(), because several
    engineered features depend on raw columns that are imputed here. Calling
    fill_missing() before engineer_features() would cause those features to
    be computed from already-imputed values, which is fine, but calling it
    after ensures that columns like TotalSF and GarageAge are computed from
    the original nulls first (which produce NaN in engineered features) and
    then the fill step cleans those up.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame after engineer_features() has been applied.
        May contain nulls in both raw and engineered columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with all nulls in the specified columns filled. Other
        columns are not modified. Columns not present in the DataFrame
        are silently skipped, making the function safe to call on both
        training and test splits regardless of which columns have been
        dropped.

    Notes
    -----
    The following imputation strategy is applied per column group:

    Categorical — filled with "None" (structural absence):
        MSZoning, Utilities, Exterior1st, Exterior2nd, MasVnrType,
        KitchenQual, SaleType, GarageType, GarageFinish, GarageQual,
        GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1,
        BsmtFinType2, Electrical, FireplaceQu, PoolQC, Fence,
        MiscFeature, Alley, ExterCond, BsmtCond

    Functional — filled with "Typ" (standard assumption for missing):
        Functional

    Numeric — filled with 0 (structural absence):
        GarageArea, GarageCars, GarageYrBlt, TotalBsmtSF, BsmtFinSF1,
        BsmtFinSF2, BsmtUnfSF, BsmtFullBath, BsmtHalfBath, MasVnrArea,
        LotFrontage, BedroomAbvGr, TotRmsAbvGrd

    Examples
    --------
    >>> X_train = engineer_features(X_train)
    >>> X_test  = engineer_features(X_test)
    >>> X_train = fill_missing(X_train)
    >>> X_test  = fill_missing(X_test)
    >>> remaining = X_test.isnull().sum()
    >>> print(remaining[remaining > 0])
    """
    df = df.copy()

    # ── categorical — "None" means feature doesn't exist ─────────────────────
    cat_fill_none = [
        "MSZoning",
        "Utilities",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "KitchenQual",
        "SaleType",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Electrical",
        "FireplaceQu",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "Alley",
        "ExterCond",
        "BsmtCond",
    ]
    for col in cat_fill_none:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # ── Functional defaults to Typ when missing ───────────────────────────────
    if "Functional" in df.columns:
        df["Functional"] = df["Functional"].fillna("Typ")

    # ── numeric — 0 means absence of feature ─────────────────────────────────
    num_fill_zero = [
        "GarageArea",
        "GarageCars",
        "GarageYrBlt",
        "TotalBsmtSF",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
        "LotFrontage",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
    ]
    for col in num_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
