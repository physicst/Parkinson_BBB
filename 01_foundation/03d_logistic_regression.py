"""Fit logistic regression: PD vs HC ~ BBB_score, evaluate AUC.
Mirrors the v1 analysis from Manuscript.md."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

from lib.config import load_config


def main() -> None:
    cfg = load_config()
    cache = Path(cfg["results"]["cache_dir"])
    v1_dir = Path(cfg["results"]["v1_baseline_dir"])

    canonical = pd.read_parquet(cache / "canonical_clinical.parquet")
    scores = pd.read_parquet(v1_dir / "ppmi_baseline_v1_scores.parquet")

    df = scores.merge(
        canonical[["RNA_HudAlphaID", "PATNO", "Cohort_Current",
                   "Age_at_Baseline", "Sex"]].drop_duplicates("RNA_HudAlphaID"),
        on="RNA_HudAlphaID", how="left",
    )
    df = df[df["Cohort_Current"].isin(["PD", "HC"])].copy()
    df["y"] = (df["Cohort_Current"] == "PD").astype(int)
    df["sex_male"] = (df["Sex"] == "Male").astype(int)
    df = df.dropna(subset=["BBB_score_v1", "Age_at_Baseline", "sex_male"])
    print(f"Modeling cohort: {len(df)} subjects "
          f"({(df['y']==1).sum()} PD, {(df['y']==0).sum()} HC)")

    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260501)

    # Model 1: BBB score alone (matches Manuscript.md headline)
    X1 = df[["BBB_score_v1"]].values
    y = df["y"].values
    aucs1 = cross_val_score(clf, X1, y, scoring="roc_auc", cv=cv)
    proba1 = cross_val_predict(clf, X1, y, cv=cv, method="predict_proba")[:, 1]
    auc1_overall = roc_auc_score(y, proba1)

    # Model 2: Age + Sex only (Manuscript.md reports AUC ~ 0.510)
    X2 = df[["Age_at_Baseline", "sex_male"]].values
    aucs2 = cross_val_score(clf, X2, y, scoring="roc_auc", cv=cv)
    proba2 = cross_val_predict(clf, X2, y, cv=cv, method="predict_proba")[:, 1]
    auc2_overall = roc_auc_score(y, proba2)

    # Model 3: BBB + Age + Sex
    X3 = df[["BBB_score_v1", "Age_at_Baseline", "sex_male"]].values
    aucs3 = cross_val_score(clf, X3, y, scoring="roc_auc", cv=cv)
    proba3 = cross_val_predict(clf, X3, y, cv=cv, method="predict_proba")[:, 1]
    auc3_overall = roc_auc_score(y, proba3)

    out = pd.DataFrame({
        "model": ["BBB_score_only", "Age+Sex_only", "BBB+Age+Sex"],
        "auc_5fold_mean": [aucs1.mean(), aucs2.mean(), aucs3.mean()],
        "auc_5fold_sd": [aucs1.std(), aucs2.std(), aucs3.std()],
        "auc_overall_pooled": [auc1_overall, auc2_overall, auc3_overall],
        "manuscript_target": [0.811, 0.510, None],
    })
    out["concordance_within_0.05"] = (
        out.apply(
            lambda r: abs(r["auc_overall_pooled"] - r["manuscript_target"]) <= 0.05
            if pd.notna(r["manuscript_target"]) else None,
            axis=1,
        )
    )
    out.to_csv(v1_dir / "v1_auc_results.tsv", sep="\t", index=False)
    print("\n", out.to_string(index=False))

    # Save ROC data for the report
    fpr1, tpr1, _ = roc_curve(y, proba1)
    pd.DataFrame({"fpr": fpr1, "tpr": tpr1}).to_csv(v1_dir / "v1_roc_curve.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
