
\"""
Synthetic framework to generate Figures 1–5 for:

- PCA before and after batch correction ("ComBat-like")
- Covariate balance before and after matching
- Prognostic signature: ROC curve
- Calibration plot with isotonic recalibration
- Epidemiology metrics: VE and Rt trajectories

All data are synthetic, generated on the fly.
No external datasets are required.

Requirements:
  python >= 3.9
  numpy
  pandas
  matplotlib
  scikit-learn
  statsmodels (optional, only for logistic regression summary if you want)

Run:
  python generate_all_figures.py
\"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# -------------------------------------------------------------------
# Global configuration
# -------------------------------------------------------------------

np.random.seed(42)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def standardised_mean_difference(x_treated, x_control):
    \"""
    Standardised mean difference for one covariate.
    \"""
    mean_t = np.mean(x_treated)
    mean_c = np.mean(x_control)
    sd = np.sqrt((np.var(x_treated, ddof=1) + np.var(x_control, ddof=1)) / 2.0)
    return (mean_t - mean_c) / sd if sd > 0 else 0.0


# -------------------------------------------------------------------
# FIGURE 1 – PCA before and after batch correction (ComBat-like)
# -------------------------------------------------------------------

def generate_figure1_pca():
    \"""
    Synthetic gene expression matrix with:
      - 2 batches (batch 0 / batch 1)
      - 2 groups (e.g. case vs control)
      - batch effects and group effects

    We apply a very simplified "ComBat-like" correction:
      X_corrected = X - batch_mean_per_gene

    Then we show PCA before and after correction.
    \"""
    n_samples = 200
    n_genes = 500

    # Two batches of equal size
    batch = np.random.choice([0, 1], size=n_samples)
    # Two groups (e.g. case vs control)
    group = np.random.choice([0, 1], size=n_samples)

    # Base expression: N(0, 1)
    X = np.random.normal(0, 1, (n_samples, n_genes))

    # Add batch effect: each batch gets a shift
    batch_effect = (batch.reshape(-1, 1) * 0.8)
    X += batch_effect

    # Add a modest group effect on a subset of genes
    de_genes = np.random.choice(n_genes, size=80, replace=False)
    group_effect = np.zeros_like(X)
    group_effect[:, de_genes] = group.reshape(-1, 1) * 0.5
    X += group_effect

    # Standardise per gene
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # PCA before correction
    pca = PCA(n_components=2)
    pc_before = pca.fit_transform(X_std)

    # Very simple ComBat-like batch mean removal per gene
    X_corrected = X_std.copy()
    for b in [0, 1]:
        idx = np.where(batch == b)[0]
        batch_mean = X_std[idx].mean(axis=0)
        X_corrected[idx] = X_std[idx] - batch_mean

    # PCA after correction
    pc_after = pca.fit_transform(X_corrected)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    # Before
    sc1 = axes[0].scatter(
        pc_before[:, 0],
        pc_before[:, 1],
        c=batch,
        cmap="coolwarm",
        alpha=0.8
    )
    axes[0].set_title(
        "Figure 1A: PCA before batch correction\\n"
        "(synthetic; colour = batch)"
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # After
    sc2 = axes[1].scatter(
        pc_after[:, 0],
        pc_after[:, 1],
        c=batch,
        cmap="coolwarm",
        alpha=0.8
    )
    axes[1].set_title(
        "Figure 1B: PCA after ComBat-like correction\\n"
        "(batch separation reduced)"
    )
    axes[1].set_xlabel("PC1")

    fig.colorbar(sc2, ax=axes.ravel().tolist(), label="Batch")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure1_pca_before_after.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {out_path}")


# -------------------------------------------------------------------
# FIGURE 2 – Covariate balance before and after matching
# -------------------------------------------------------------------

def generate_figure2_covariate_balance():
    \"""
    Synthetic propensity score + covariates:

    - 1000 subjects
    - 10 covariates
    - treatment depends on covariates
    - we simulate "matching" by subtracting most of the
      systematic difference in means so that SMD goes from
      ~0.21 to ~0.06 (approx, not exact guarantee).
    \"""
    n = 1000
    p = 10

    # Covariates
    Z = np.random.normal(0, 1, size=(n, p))

    # True linear predictor for treatment
    beta_t = np.linspace(0.5, 1.2, p)
    lp_treat = Z @ beta_t
    p_treat = 1 / (1 + np.exp(-lp_treat))

    # Treatment assignment
    treated = np.random.binomial(1, p_treat)

    # Standardised mean difference per covariate BEFORE matching
    smd_before = []
    for j in range(p):
        x_t = Z[treated == 1, j]
        x_c = Z[treated == 0, j]
        smd_before.append(standardised_mean_difference(x_t, x_c))
    smd_before = np.array(smd_before)

    # Simulate matching effect: move treated covariates
    # closer to controls to reduce SMD.
    # This is a fake "perfect matching", just to illustrate.
    Z_matched = Z.copy()
    for j in range(p):
        x_t = Z_matched[treated == 1, j]
        x_c = Z_matched[treated == 0, j]
        mean_diff = x_t.mean() - x_c.mean()
        # shrink the difference by ~70%
        Z_matched[treated == 1, j] -= 0.7 * mean_diff

    smd_after = []
    for j in range(p):
        x_t = Z_matched[treated == 1, j]
        x_c = Z_matched[treated == 0, j]
        smd_after.append(standardised_mean_difference(x_t, x_c))
    smd_after = np.array(smd_after)

    # Plot
    covariate_index = np.arange(1, p + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(
        covariate_index,
        np.abs(smd_before),
        marker="o",
        label="Before matching"
    )
    plt.plot(
        covariate_index,
        np.abs(smd_after),
        marker="o",
        label="After matching"
    )
    plt.axhline(0.1, color="grey", linestyle="--", alpha=0.5)
    plt.xlabel("Covariate")
    plt.ylabel("Absolute SMD")
    plt.title(
        "Figure 2: Covariate balance before and after matching\\n"
        "(synthetic; median SMD reduced)"
    )
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure2_covariate_balance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {out_path}")


# -------------------------------------------------------------------
# FIGURE 3 – ROC under nested cross validation (synthetic)
# -------------------------------------------------------------------

def generate_figure3_roc():
    \"""
    Synthetic "LASSO-like" signature ROC:

    - We generate features X and outcome y with a true signal
      in a subset of features.
    - We train a simple logistic regression (L2 penalty) in
      outer CV and collect predicted probabilities.
    - Then we plot a ROC curve with the mean AUC.
    \"""
    n = 800
    p = 50

    # True signal on first 8 features
    X = np.random.normal(0, 1, size=(n, p))
    beta_true = np.zeros(p)
    beta_true[:8] = np.linspace(0.6, 1.2, 8)
    lp = X @ beta_true
    p_y = 1 / (1 + np.exp(-lp))
    y = np.random.binomial(1, p_y)

    # Nested CV simplified: outer CV only, no inner hyperparam search
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=500
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]

        y_all.append(y_test)
        y_pred_all.append(y_proba)

    y_all = np.concatenate(y_all)
    y_pred_all = np.concatenate(y_pred_all)

    auc = roc_auc_score(y_all, y_pred_all)
    fpr, tpr, _ = roc_curve(y_all, y_pred_all)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Signature (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Figure 3: ROC under cross-validation (synthetic)\\n"
        "Dashed line = chance"
    )
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure3_roc.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {out_path}")


# -------------------------------------------------------------------
# FIGURE 4 – Calibration plot
# -------------------------------------------------------------------

def generate_figure4_calibration():
    \"""
    Calibration plot (synthetic):

    - Use the same X, y generation as for ROC
      (or regenerate a fresh dataset).
    - We fit a logistic model, compute predicted probabilities,
      calibrate with isotonic regression, and plot:
        - mean predicted vs observed (binned)
        - diagonal line for perfect calibration
    \"""
    n = 800
    p = 50

    # Generate new synthetic data
    X = np.random.normal(0, 1, size=(n, p))
    beta_true = np.zeros(p)
    beta_true[:8] = np.linspace(0.6, 1.2, 8)
    lp = X @ beta_true
    p_y = 1 / (1 + np.exp(-lp))
    y = np.random.binomial(1, p_y)

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=500
    )
    clf.fit(X, y)
    prob_raw = clf.predict_proba(X)[:, 1]

    # Brier score
    brier = brier_score_loss(y, prob_raw)

    # Isotonic regression for recalibration
    ir = IsotonicRegression(out_of_bounds="clip")
    prob_iso = ir.fit_transform(prob_raw, y)

    # Binning for smooth calibration curve
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(prob_raw, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    obs_rate = np.array([
        y[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan
        for i in range(n_bins)
    ])

    plt.figure(figsize=(6, 5))
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    # Observed vs predicted
    plt.plot(bin_centers, obs_rate, "o-", label="Raw model")

    # Isotonic-smoothed curve (sorted)
    sort_idx = np.argsort(prob_raw)
    plt.plot(
        prob_raw[sort_idx],
        prob_iso[sort_idx],
        alpha=0.5,
        label="Isotonic recalibration"
    )

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(
        f"Figure 4: Calibration plot (synthetic)\\n"
        f"Brier score: {brier:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure4_calibration.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {out_path}")


# -------------------------------------------------------------------
# FIGURE 5 – VE and Rt trajectories
# -------------------------------------------------------------------

def generate_figure5_ve_rt():
    \"""
    Synthetic VE and Rt trajectories:

    - Weeks 1 to 10
    - VE rising from ~0.63 to ~0.79
    - Rt declining from ~1.14 to ~0.89
    - Simple symmetric uncertainty bands around each curve
      (e.g. +/- 0.05 for VE, +/- 0.07 for Rt)
    \"""
    weeks = np.arange(1, 11)

    ve_start, ve_end = 0.63, 0.79
    rt_start, rt_end = 1.14, 0.89

    ve = ve_start + (weeks - 1) * (ve_end - ve_start) / (len(weeks) - 1)
    rt = rt_start + (weeks - 1) * (rt_end - rt_start) / (len(weeks) - 1)

    # Simple uncertainty bands
    ve_lower = ve - 0.05
    ve_upper = ve + 0.05
    rt_lower = rt - 0.07
    rt_upper = rt + 0.07

    fig, ax1 = plt.subplots(figsize=(7, 5))

    color_ve = "tab:blue"
    color_rt = "tab:red"

    ax1.set_xlabel("Week")
    ax1.set_ylabel("Vaccine effectiveness", color=color_ve)
    ax1.plot(weeks, ve, marker="o", color=color_ve, label="VE")
    ax1.fill_between(
        weeks, ve_lower, ve_upper,
        alpha=0.2,
        color=color_ve
    )
    ax1.tick_params(axis="y", labelcolor=color_ve)
    ax1.set_ylim(0.4, 1.0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Effective reproduction number Rt", color=color_rt)
    ax2.plot(weeks, rt, marker="s", color=color_rt, label="Rt")
    ax2.fill_between(
        weeks, rt_lower, rt_upper,
        alpha=0.2,
        color=color_rt
    )
    ax2.tick_params(axis="y", labelcolor=color_rt)
    ax2.set_ylim(0.6, 1.4)

    plt.title(
        "Figure 5: VE rising and Rt declining trajectories (synthetic)\\n"
        "With illustrative uncertainty bands"
    )
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure5_ve_rt.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 to {out_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    generate_figure1_pca()
    generate_figure2_covariate_balance()
    generate_figure3_roc()
    generate_figure4_calibration()
    generate_figure5_ve_rt()
    print("\\nAll figures generated in the 'figures' folder.")
