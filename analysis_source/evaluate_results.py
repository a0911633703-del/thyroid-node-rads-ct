"""Evaluate fixed nested-CV out-of-fold predictions using shared cluster bootstrap.

The analysis unit is the selected lymph node. Cluster identifiers are source-record
patient groups and have not been independently verified against hospital identifiers.
Intervals condition on the supplied OOF predictions; the bootstrap does not refit
models, retune hyperparameters, recreate folds, or reselect classification thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score


MODELS = ("NodeRADS", "Imaging", "Combined")
METRICS = (
    "roc_auc", "average_precision", "brier_score", "log_loss", "accuracy",
    "sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy",
    "calibration_in_the_large", "calibration_slope", "calibration_intercept_free",
)
PATIENT_METRICS = (
    "roc_auc", "average_precision", "accuracy", "sensitivity", "specificity",
    "ppv", "npv", "f1", "balanced_accuracy",
)
EPSILON = 1e-8


def safe_divide(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def calibration_parameters(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Return CITL, free calibration slope, and free calibration intercept.

    CITL solves sum[w*(y-expit(a+logit(p)))]=0, i.e. offset slope fixed to 1.
    The two-parameter fit is unpenalized. Centering/scaling improves numerics and
    its coefficients are transformed back to the original logit-probability scale.
    Nonconvergence, singular Hessians, and boundary solutions return NaN.
    """
    take = w > 0
    y, p, w = y[take], p[take], w[take]
    total = float(w.sum())
    events = float(np.dot(w, y))
    if total <= 0 or not 0 < events < total:
        return (float("nan"),) * 3
    x = logit(np.clip(p, EPSILON, 1 - EPSILON))
    citl = brentq(lambda a: float(np.dot(w, expit(x + a)) - events), -100, 100)
    mu = float(np.average(x, weights=w))
    sd = float(np.sqrt(np.average((x - mu) ** 2, weights=w)))
    if sd < 1e-10:
        return float(citl), float("nan"), float("nan")
    z = (x - mu) / sd
    design = np.column_stack((np.ones(y.size), z))
    weight = w / total

    def loss_gradient(coef: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ coef
        loss = float(np.dot(weight, np.logaddexp(0, linear) - y * linear))
        gradient = design.T @ (weight * (expit(linear) - y))
        return loss, gradient

    initial = np.array([float(logit(events / total)), 0.0])
    fit = minimize(loss_gradient, initial, jac=True, method="L-BFGS-B",
                   bounds=((-100, 100), (-100, 100)),
                   options={"maxiter": 150, "ftol": 1e-12, "gtol": 1e-9})
    fitted = expit(design @ fit.x)
    hessian = (design.T * (weight * fitted * (1 - fitted))) @ design
    well_defined = (
        np.all(np.isfinite(fit.x))
        and (fit.success or np.max(np.abs(fit.jac)) < 1e-6)
        and np.max(np.abs(fit.x)) < 99.9
        and np.linalg.cond(hessian) < 1e10
    )
    if not well_defined:
        return float(citl), float("nan"), float("nan")
    slope = float(fit.x[1] / sd)
    intercept = float(fit.x[0] - slope * mu)
    return float(citl), slope, intercept


def metric_values(y: np.ndarray, p: np.ndarray, prediction: np.ndarray,
                  w: np.ndarray | None = None, calibration: bool = True) -> dict[str, float]:
    if w is None:
        w = np.ones(y.size, dtype=float)
    w = np.asarray(w, dtype=float)
    total = float(w.sum())
    positive = float(np.dot(w, y))
    negative = total - positive
    tp = float(np.dot(w, (prediction == 1) & (y == 1)))
    fp = float(np.dot(w, (prediction == 1) & (y == 0)))
    tn = float(np.dot(w, (prediction == 0) & (y == 0)))
    fn = float(np.dot(w, (prediction == 0) & (y == 1)))
    sens, spec = safe_divide(tp, tp + fn), safe_divide(tn, tn + fp)
    pc = np.clip(p, EPSILON, 1 - EPSILON)
    results = {
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if positive > 0 and negative > 0 else float("nan"),
        "average_precision": float(average_precision_score(y, p, sample_weight=w)) if positive > 0 else float("nan"),
        "accuracy": safe_divide(tp + tn, total),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": float((sens + spec) / 2),
    }
    if calibration:
        citl, slope, intercept = calibration_parameters(y, p, w)
        results.update({
            "brier_score": float(np.dot(w, (p - y) ** 2) / total),
            "log_loss": float(-np.dot(w, y * np.log(pc) + (1 - y) * np.log1p(-pc)) / total),
            "calibration_in_the_large": citl,
            "calibration_slope": slope,
            "calibration_intercept_free": intercept,
        })
    return results


def interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return float(lo), float(hi), int(finite.size)


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    result = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p[index])
        result[index] = min(1.0, running)
    return result.tolist()


def calibration_bin_specification(p: np.ndarray, y: np.ndarray, model: str) -> tuple[np.ndarray, list[dict]]:
    # pd.qcut merges coincident quantile boundaries. Bins are fixed thereafter.
    codes, edges = pd.qcut(pd.Series(p), q=5, labels=False, retbins=True, duplicates="drop")
    if len(edges) < 2 or codes.isna().all():
        code = np.zeros(len(p), dtype=int)
        edges = np.array([float(p.min()), float(p.max())])
    else:
        code = codes.to_numpy(dtype=int)
    info = []
    for b in sorted(np.unique(code)):
        take = code == b
        info.append({
            "model": model, "bin": int(b + 1),
            "lower_probability_boundary": float(edges[b]),
            "upper_probability_boundary": float(edges[b + 1]),
            "mean_predicted_probability": float(p[take].mean()),
            "observed_event_fraction": float(y[take].mean()),
            "n_nodes": int(take.sum()), "events": int(y[take].sum()),
        })
    return code, info


def load_and_validate(prediction_file: Path, cohort_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(prediction_file, dtype={"node_id": str, "group_id": str})
    cohort = pd.read_csv(cohort_file, dtype={"node_id": str, "group_id": str})
    required = {"node_id", "group_id", "fold", "y_true", "model", "probability", "prediction", "threshold"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(df.columns))}")
    if df[list(required)].isna().any().any():
        raise ValueError("Prediction input contains missing required values")
    if set(df["model"]) != set(MODELS):
        raise ValueError(f"Exactly these models are required: {MODELS}")
    if df.duplicated(["model", "node_id"]).any():
        raise ValueError("More than one OOF prediction per node and model")
    if cohort["node_id"].duplicated().any():
        raise ValueError("Duplicate node_id in cohort")
    for field in ("y_true", "prediction"):
        if not set(df[field].unique()).issubset({0, 1}):
            raise ValueError(f"{field} must be binary 0/1")
    for field in ("probability", "threshold"):
        if not np.isfinite(df[field]).all() or not df[field].between(0, 1).all():
            raise ValueError(f"{field} must be finite and in [0,1]")
    base_nodes = set(cohort["node_id"])
    for model in MODELS:
        if set(df.loc[df["model"] == model, "node_id"]) != base_nodes:
            raise ValueError(f"Prediction/cohort node sets differ for {model}")
    for column in ("group_id", "fold", "y_true"):
        if (df.groupby("node_id")[column].nunique() != 1).any():
            raise ValueError(f"{column} differs between models for some nodes")
    if (df.groupby("group_id")["fold"].nunique() != 1).any():
        raise ValueError("Source group appears in multiple evaluation folds")
    if (df.groupby(["model", "fold"])["threshold"].nunique() != 1).any():
        raise ValueError("Classification threshold is not constant within model/fold")
    # Retain author-supplied thresholds and classifications rather than optimize them.
    expected = (df["probability"] >= df["threshold"]).astype(int)
    if not np.array_equal(df["prediction"].to_numpy(dtype=int), expected.to_numpy()):
        raise ValueError("Class predictions disagree with probability >= threshold")
    joined = df.loc[df["model"] == MODELS[0], ["node_id", "group_id", "y_true"]].merge(
        cohort[["node_id", "group_id", "outcome"]], on="node_id", validate="one_to_one", suffixes=("_pred", "_cohort"))
    if (joined["group_id_pred"] != joined["group_id_cohort"]).any() or (joined["y_true"] != joined["outcome"]).any():
        raise ValueError("Cohort group or outcome disagrees with OOF file")
    return df, cohort


def evaluate(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    analysis = args.analysis_dir.resolve()
    prediction_file = (args.predictions or analysis / "primary_predictions.csv").resolve()
    cohort_file = (args.cohort or analysis / "cohort_primary.csv").resolve()
    output_dir = (args.output_dir or analysis).resolve()
    analysis_role = args.analysis_role or ("primary" if prediction_file.name == "primary_predictions.csv" else "sensitivity")
    if any(c in args.prefix for c in ("/", "\\")):
        raise ValueError("Output prefix must be a filename prefix, not a path")
    output_dir.mkdir(parents=True, exist_ok=True)
    df, cohort = load_and_validate(prediction_file, cohort_file)
    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap replicates are required")
    node_ids = cohort["node_id"].tolist()
    indexed = {m: df.loc[df["model"] == m].set_index("node_id").loc[node_ids] for m in MODELS}
    first = indexed[MODELS[0]]
    y = first["y_true"].to_numpy(dtype=int)
    group_ids = first["group_id"].astype(str).to_numpy()
    groups = np.array(sorted(set(group_ids)), dtype=str)
    group_lookup = {g: i for i, g in enumerate(groups)}
    group_codes = np.array([group_lookup[g] for g in group_ids], dtype=int)
    n_groups, n_nodes = len(groups), len(y)
    p_by_model = {m: indexed[m]["probability"].to_numpy(dtype=float) for m in MODELS}
    pred_by_model = {m: indexed[m]["prediction"].to_numpy(dtype=int) for m in MODELS}
    thresholds = np.round(np.arange(0.05, 0.501, 0.01), 2)
    odds = thresholds / (1 - thresholds)
    point = {m: metric_values(y, p_by_model[m], pred_by_model[m]) for m in MODELS}
    boot = {m: {k: np.full(args.bootstrap, np.nan) for k in METRICS} for m in MODELS}
    bin_info, bin_codes, bin_boot = {}, {}, {}
    for m in MODELS:
        bin_codes[m], bin_info[m] = calibration_bin_specification(p_by_model[m], y, m)
        for row in bin_info[m]:
            b = row["bin"] - 1
            row["n_source_groups"] = int(len(set(group_ids[bin_codes[m] == b])))
        bin_boot[m] = np.full((args.bootstrap, len(bin_info[m])), np.nan)

    dca_point, dca_boot, dca_contribution = {}, {}, {}
    for m in MODELS:
        positive = p_by_model[m][:, None] >= thresholds[None, :]
        dca_contribution[m] = positive * (y[:, None] - (1 - y[:, None]) * odds[None, :])
        dca_point[m] = dca_contribution[m].mean(axis=0)
        dca_boot[m] = np.full((args.bootstrap, len(thresholds)), np.nan)
    dca_point["TreatAll"] = float(y.mean()) - (1 - float(y.mean())) * odds
    dca_point["TreatNone"] = np.zeros(len(thresholds))
    dca_boot["TreatAll"] = np.full((args.bootstrap, len(thresholds)), np.nan)
    dca_boot["TreatNone"] = np.zeros((args.bootstrap, len(thresholds)))

    patient_rows = []
    patient_y = np.zeros(n_groups, dtype=int)
    patient_p, patient_pred = {}, {}
    for m in MODELS:
        patient_p[m] = np.zeros(n_groups)
        patient_pred[m] = np.zeros(n_groups, dtype=int)
        for i, group in enumerate(groups):
            take = group_codes == i
            patient_y[i] = int(y[take].max())
            patient_p[m][i] = float(p_by_model[m][take].max())
            patient_pred[m][i] = int(pred_by_model[m][take].max())
            patient_rows.append({
                "group_id": group, "fold": int(first.loc[np.array(node_ids)[take], "fold"].iloc[0]),
                "model": m, "y_true_selected_nodes": patient_y[i],
                "max_node_score": patient_p[m][i], "any_node_positive": patient_pred[m][i],
                "n_selected_nodes": int(take.sum()), "n_malignant_selected_nodes": int(y[take].sum()),
                "node_threshold": float(indexed[m].loc[np.array(node_ids)[take], "threshold"].iloc[0]),
            })
    patient_point = {m: metric_values(patient_y, patient_p[m], patient_pred[m], calibration=False) for m in MODELS}
    patient_boot = {m: {k: np.full(args.bootstrap, np.nan) for k in PATIENT_METRICS} for m in MODELS}

    rng = np.random.default_rng(args.seed)
    for b in range(args.bootstrap):
        sampled_groups = rng.integers(0, n_groups, size=n_groups)
        group_weights = np.bincount(sampled_groups, minlength=n_groups).astype(float)
        node_weights = group_weights[group_codes]
        sampled_n = float(node_weights.sum())
        prevalence = float(np.dot(node_weights, y) / sampled_n)
        for m in MODELS:
            result = metric_values(y, p_by_model[m], pred_by_model[m], node_weights)
            for k in METRICS:
                boot[m][k][b] = result[k]
            patient_result = metric_values(patient_y, patient_p[m], patient_pred[m], group_weights, calibration=False)
            for k in PATIENT_METRICS:
                patient_boot[m][k][b] = patient_result[k]
            dca_boot[m][b, :] = node_weights @ dca_contribution[m] / sampled_n
            for bin_index in range(len(bin_info[m])):
                take = bin_codes[m] == bin_index
                bin_boot[m][b, bin_index] = safe_divide(float(np.dot(node_weights[take], y[take])), float(node_weights[take].sum()))
        dca_boot["TreatAll"][b, :] = prevalence - (1 - prevalence) * odds
        if (b + 1) % 250 == 0:
            print(f"Shared source-group bootstrap {b + 1}/{args.bootstrap}", flush=True)

    metric_rows, patient_metric_rows = [], []
    for m in MODELS:
        for k in METRICS:
            low, high, valid = interval(boot[m][k])
            metric_rows.append({"model": m, "metric": k, "estimate": point[m][k],
                                "ci_lower": low, "ci_upper": high, "bootstrap_valid": valid,
                                "n_nodes": n_nodes, "n_source_groups": n_groups,
                                "malignant_nodes": int(y.sum()), "interval_method": "fixed_oof_source_group_percentile_bootstrap"})
        for k in PATIENT_METRICS:
            low, high, valid = interval(patient_boot[m][k])
            patient_metric_rows.append({"model": m, "metric": k, "estimate": patient_point[m][k],
                                       "ci_lower": low, "ci_upper": high, "bootstrap_valid": valid,
                                       "n_source_groups": n_groups,
                                       "groups_with_malignancy_in_selected_nodes": int(patient_y.sum()),
                                       "scope": "selected_nodes_only_max_node_score_is_not_patient_probability"})
        for b, row in enumerate(bin_info[m]):
            low, high, valid = interval(bin_boot[m][:, b])
            row.update({"observed_ci_lower": low, "observed_ci_upper": high,
                        "bootstrap_valid": valid, "bin_definition": "fixed_full_oof_probability_quantile_bins"})

    comparisons = []
    for metric in ("roc_auc", "brier_score"):
        for comparator in ("NodeRADS", "Imaging"):
            estimate = float(point["Combined"][metric] - point[comparator][metric])
            differences = boot["Combined"][metric] - boot[comparator][metric]
            low, high, valid = interval(differences)
            finite = differences[np.isfinite(differences)]
            pvalue = float((1 + np.sum(np.abs(finite - estimate) >= abs(estimate))) / (1 + len(finite))) if len(finite) else float("nan")
            comparisons.append({
                "comparison": f"Combined - {comparator}", "metric": metric,
                "difference": estimate, "ci_lower": low, "ci_upper": high,
                "bootstrap_valid": valid, "p_centered_bootstrap": pvalue,
                "p_holm_primary_family": float("nan"),
                "comparison_status": ("primary_two_comparison_family" if analysis_role == "primary" else "sensitivity_within_scenario_two_comparison_family") if metric == "roc_auc" else "secondary_unadjusted_descriptive",
                "favors_combined_when": "positive" if metric == "roc_auc" else "negative",
            })
    for row, adjusted in zip(comparisons[:2], holm_adjust([r["p_centered_bootstrap"] for r in comparisons[:2]])):
        row["p_holm_primary_family"] = adjusted

    dca_rows = []
    for m in (*MODELS, "TreatAll", "TreatNone"):
        for i, threshold in enumerate(thresholds):
            low, high, valid = interval(dca_boot[m][:, i])
            dca_rows.append({"model": m, "threshold_probability": float(threshold),
                             "net_benefit": float(dca_point[m][i]), "ci_lower": low,
                             "ci_upper": high, "bootstrap_valid": valid,
                             "scope": "exploratory_node_level_no_established_clinical_action_threshold"})

    outputs = {
        "metrics_primary.csv": pd.DataFrame(metric_rows),
        "comparisons_primary.csv": pd.DataFrame(comparisons),
        "calibration_bins.csv": pd.DataFrame([row for m in MODELS for row in bin_info[m]]),
        "dca_source.csv": pd.DataFrame(dca_rows),
        "patient_group_metrics.csv": pd.DataFrame(patient_metric_rows),
        "patient_group_predictions.csv": pd.DataFrame(patient_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / (args.prefix + name), index=False, encoding="utf-8-sig", float_format="%.12g")
    notes = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "input_predictions": str(prediction_file), "input_cohort": str(cohort_file),
        "analysis_role": analysis_role,
        "output_prefix": args.prefix,
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed,
        "cluster_variable": "group_id from source-record patient group identifier; hospital identity not independently verified",
        "sampling": "Sample G source groups with replacement from G observed groups; keep all nodes of each draw; use exactly the same draws for all three models, node metrics, patient metrics, calibration bins, and DCA.",
        "interval_type": "Two-sided 95% percentile bootstrap intervals using finite replicates only; valid replicate counts supplied per estimate.",
        "conditioning": "Conditional on fixed OOF predictions, fixed fold-specific thresholds, and fixed calibration bins. Does not include uncertainty from refitting, tuning, feature or algorithm selection, or resplitting folds.",
        "primary_comparisons": "Combined-NodeRADS and Combined-Imaging ROC-AUC differences; two centered-bootstrap P values adjusted together using Holm.",
        "multiplicity_scope": "Holm adjustment applies only to the two AUC contrasts within this analysis scenario. Sensitivity analyses are exploratory and are not jointly adjusted across scenarios.",
        "centered_bootstrap_p_formula": "(1 + number(abs(delta_boot - delta_hat) >= abs(delta_hat))) / (1 + number_of_finite_paired_replicates)",
        "brier_comparisons": "Secondary descriptive differences with unadjusted P values; negative difference favors Combined.",
        "calibration_in_the_large": "Intercept-only binomial logistic model with offset logit(p), fixed slope 1.",
        "free_calibration": "Separate unpenalized binomial logistic regression y~1+logit(p); free intercept is not calibration-in-the-large.",
        "probability_clipping_for_logloss_and_calibration": EPSILON,
        "average_precision": "sklearn average_precision_score, stepwise noninterpolated precision-recall summary; not trapezoidal PR area.",
        "node_classification": "Supplied probability >= fold-specific development threshold; thresholds are not reselected from OOF outcomes or bootstrap data.",
        "patient_scope": "Source groups among selected nodes only. Truth is any malignant selected node; score is maximum node probability; classification is any selected node classified positive. Max node score is not a calibrated probability of patient-level nodal disease. No patient Brier, calibration, or DCA computed.",
        "dca": "Node-level net benefit = TP/N - FP/N * pt/(1-pt); thresholds 0.05 to 0.50 by 0.01. Exploratory selected-sample analysis without an established clinical action or validated clinical threshold. TreatAll and TreatNone supplied.",
        "calibration_bins": "Five full-OOF predicted probability quantile bins with duplicate edges merged. Boundaries fixed across bootstrap replicates; CIs describe observed fractions in those fixed bins.",
        "n_nodes": n_nodes, "n_source_groups": n_groups, "malignant_nodes": int(y.sum()),
        "groups_with_malignancy_in_selected_nodes": int(patient_y.sum()),
        "invalid_node_bootstrap_estimates": {m: {k: int((~np.isfinite(boot[m][k])).sum()) for k in METRICS} for m in MODELS},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "elapsed_seconds": float(time.perf_counter() - started),
        "outputs": [args.prefix + name for name in outputs],
    }
    (output_dir / (args.prefix + "bootstrap_notes.json")).write_text(json.dumps(notes, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"outputs": list(outputs), "elapsed_seconds": notes["elapsed_seconds"],
                      "n_nodes": n_nodes, "n_source_groups": n_groups}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parents[1] / "analysis")
    parser.add_argument("--predictions", type=Path, help="Long-format OOF prediction CSV; defaults to analysis/primary_predictions.csv")
    parser.add_argument("--cohort", type=Path, help="Matching cohort CSV with node_id, group_id and outcome; defaults to analysis/cohort_primary.csv")
    parser.add_argument("--output-dir", type=Path, help="Destination directory; defaults to analysis directory")
    parser.add_argument("--prefix", default="", help="Output filename prefix, e.g. all254_")
    parser.add_argument("--analysis-role", choices=("primary", "sensitivity"), help="Reporting role, inferred from prediction filename if omitted; does not change calculations")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
