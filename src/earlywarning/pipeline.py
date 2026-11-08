"""End to end: profile, build features, compare models, evaluate, pick a threshold, explain.

    PYTHONPATH=src python -m earlywarning --data-dir data --out outputs
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config, explain, models, plots, profiling
from . import evaluation as ev
from .data import load_extract
from .features import build_features, feature_group

CHOSEN = "Logistic regression"
EFFECTIVENESS_LEVELS = [1.0, 0.75, 0.5, 0.25]
CAPACITY_SHARES = [0.05, 0.10, 0.20]


def split_customers(labelled: pd.DataFrame, seed: int = config.SEED,
                    test_share: float = config.TEST_SHARE):
    # Stratify on label x hardship plan so the subgroup comparison has stable group sizes.
    strata = labelled[config.TARGET].astype(str) + "_" + labelled["hardship_plan"].astype(str)
    dev, test = train_test_split(labelled.index, test_size=test_share, random_state=seed,
                                 stratify=strata)
    return pd.Index(np.sort(dev)), pd.Index(np.sort(test))


def compare_models(feature_sets: dict, y_dev: np.ndarray, n_folds: int, seed: int, n_boot: int):
    """Out-of-fold scores on the development split; the test set is not touched here."""
    runs = {
        CHOSEN: ("core", models.logistic),
        "Logistic regression + recent-cycle and extreme-value features": ("extended", models.logistic),
        "Gradient boosting (extended features)": ("extended", lambda: models.boosted_trees(seed)),
        "Logistic regression on payments as recorded (not point-in-time)": ("as_recorded", models.logistic),
    }
    oof = {name: models.out_of_fold(make(), feature_sets[fs], y_dev, n_folds, seed)
           for name, (fs, make) in runs.items()}
    # What a support team could do today with a spreadsheet: rank by how many bills were late.
    oof["Rule of thumb: number of late bills"] = feature_sets["core"]["overdue_share"].to_numpy()

    rows = []
    for name, p in oof.items():
        is_probability = not name.startswith("Rule")
        row = {
            "model": name,
            "auc": roc_auc_score(y_dev, p),
            "auprc": average_precision_score(y_dev, p),
            "brier": brier_score_loss(y_dev, p) if is_probability else np.nan,
            "log_loss": log_loss(y_dev, p) if is_probability else np.nan,
        }
        if name != CHOSEN:
            diff, lo, hi = ev.paired_auc_difference(y_dev, p, oof[CHOSEN], n_boot, seed)
            row.update(auc_minus_chosen=diff, auc_minus_chosen_lo=lo, auc_minus_chosen_hi=hi)
        rows.append(row)

    # How much AUC moves from one fold to the next: a feel for how far a single
    # test sample can land from the cross-validated number by luck alone.
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_auc = [roc_auc_score(y_dev[va], oof[CHOSEN][va])
                for _, va in folds.split(np.zeros(len(y_dev)), y_dev)]
    return pd.DataFrame(rows), oof, fold_auc


def threshold_analysis(y_dev, p_dev, y_test, p_test, late_count_dev, late_count_test,
                       n_customers, n_boot, seed, tables, figures):
    t_star = ev.break_even_threshold()
    grid = np.round(np.arange(0.01, 0.4001, 0.0025), 4)
    dev_curve = ev.cost_curve(y_dev, p_dev, grid)
    test_curve = ev.cost_curve(y_test, p_test, grid)
    dev_curve.to_csv(tables / "threshold_cost_curve_dev.csv", index=False)
    test_curve.to_csv(tables / "threshold_cost_curve_test.csv", index=False)
    dev_best = dev_curve.loc[dev_curve["cost_per_customer"].idxmin()]
    band = ev.flat_band(dev_curve, 0.02)

    # The rule of thumb gets its cut-off chosen on development data too, to be fair to it.
    cuts = range(1, int(late_count_dev.max()) + 1)
    rule_k = min(cuts, key=lambda k: ev.policy_cost(y_dev, late_count_dev >= k))

    called = p_test >= t_star
    policies = {
        "Call nobody": np.zeros(len(y_test), dtype=bool),
        "Call everyone": np.ones(len(y_test), dtype=bool),
        f"Rule of thumb: {rule_k}+ late bills": late_count_test >= rule_k,
        f"Model, threshold {t_star:.4f} (recommended)": called,
        f"Model, threshold {dev_best['threshold']:.4f} (development optimum)":
            p_test >= dev_best["threshold"],
    }
    table = pd.DataFrame([{"policy": name, **ev.operating_point(y_test, flags)}
                          for name, flags in policies.items()])
    table["projected_cost_all_customers"] = table["cost_per_customer"] * n_customers
    table.to_csv(tables / "threshold_policies_test.csv", index=False)

    boot = ev.bootstrap(y_test, p_test, {
        "cost_per_customer": ev.at_threshold(t_star, "cost_per_customer"),
        "recall": ev.at_threshold(t_star, "recall"),
        "precision": ev.at_threshold(t_star, "precision"),
        "called_share": ev.at_threshold(t_star, "called_share"),
    }, n_boot, seed)
    boot.to_csv(tables / "threshold_recommended_ci_test.csv")

    # Is the model actually better than the rule, or is that within noise? Resample both together.
    both = np.column_stack([p_test, late_count_test])

    def saving_vs_rule(yy, pp):
        rule_cost = ev.policy_cost(yy, pp[:, 1] >= rule_k)
        model_cost = ev.policy_cost(yy, pp[:, 0] >= t_star)
        return (rule_cost - model_cost) / len(yy)

    saving = ev.bootstrap(y_test, both, {"saving_per_customer_vs_rule": saving_vs_rule}, n_boot, seed)
    saving.to_csv(tables / "threshold_saving_vs_rule_test.csv")

    # If the probabilities are honest, the cost the model expects should match the realised one.
    expected = np.where(called, config.CALL_COST, config.AVOIDABLE_LOSS * p_test).mean()

    rows = []
    for e in EFFECTIVENESS_LEVELS:
        t = ev.break_even_threshold(e)
        op = ev.operating_point(y_test, p_test >= t, e)
        nobody = ev.operating_point(y_test, np.zeros(len(y_test), dtype=bool), e)
        rows.append({"effectiveness": e, "threshold": t, "called_share": op["called_share"],
                     "recall": op["recall"], "precision": op["precision"],
                     "cost_per_customer": op["cost_per_customer"],
                     "call_nobody_cost_per_customer": nobody["cost_per_customer"],
                     "saving_vs_call_nobody": 1 - op["cost_total"] / nobody["cost_total"]})
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(tables / "threshold_effectiveness_sensitivity_test.csv", index=False)

    order = np.argsort(-p_test, kind="stable")
    rows = []
    for share in CAPACITY_SHARES + [float(called.mean())]:
        k = int(round(share * len(p_test)))
        top = np.zeros(len(p_test), dtype=bool)
        top[order[:k]] = True
        op = ev.operating_point(y_test, top)
        rows.append({"call_top_share": share, "calls": k, "recall": op["recall"],
                     "precision": op["precision"], "lowest_score_called": p_test[order[k - 1]],
                     "cost_per_customer": op["cost_per_customer"]})
    capacity = pd.DataFrame(rows)
    capacity.to_csv(tables / "threshold_capacity_test.csv", index=False)

    plots.cost_curve(dev_curve, test_curve, t_star, band,
                     nobody=config.AVOIDABLE_LOSS * y_dev.mean(), everyone=config.CALL_COST,
                     path=figures / "threshold_cost_curve.png")

    return {
        "break_even_threshold": t_star,
        "dev_optimal_threshold": float(dev_best["threshold"]),
        "dev_cost_per_customer_at_optimum": float(dev_best["cost_per_customer"]),
        "dev_cost_per_customer_at_break_even": float(
            ev.operating_point(y_dev, p_dev >= t_star)["cost_per_customer"]),
        "dev_flat_band_2pct": band,
        "rule_of_thumb_late_bills": int(rule_k),
        "test_saving_per_customer_vs_rule": saving.loc["saving_per_customer_vs_rule"].to_dict(),
        "test_policies": table.set_index("policy").to_dict(orient="index"),
        "test_recommended": {
            **ev.operating_point(y_test, called),
            "model_expected_cost_per_customer": float(expected),
            "ci": boot.to_dict(orient="index"),
        },
        "effectiveness": sensitivity.to_dict(orient="records"),
        "capacity": capacity.to_dict(orient="records"),
    }


def subgroup_analysis(labelled, dev, test, y_dev, p_dev, y_test, p_test, n_boot, seed, tables):
    t_star = ev.break_even_threshold()
    g_dev = labelled.loc[dev, "hardship_plan"].to_numpy()
    g_test = labelled.loc[test, "hardship_plan"].to_numpy()
    on_test = ev.subgroup_table(y_test, p_test, g_test, t_star, n_boot, seed).assign(sample="test")
    on_dev = ev.subgroup_table(y_dev, p_dev, g_dev, t_star, n_boot, seed).assign(
        sample="development, out-of-fold")
    table = pd.concat([on_test, on_dev], ignore_index=True)
    table["group"] = table["group"].map({0: "not on plan", 1: "on hardship plan"})
    table.to_csv(tables / "subgroup_hardship_plan.csv", index=False)

    called = p_test >= t_star
    members_share_of_calls = float(g_test[called].mean())
    cal = {name: ev.calibration_table(y_test[g_test == g], p_test[g_test == g], n_bins=5)
           for g, name in [(0, "not on plan"), (1, "on hardship plan")]}
    for name, t in cal.items():
        t.to_csv(tables / f"subgroup_calibration_{name.replace(' ', '_')}_test.csv", index=False)
    return table, cal, members_share_of_calls


def shap_analysis(model, customers, X_dev, X_test, p_test, y_test, tables, figures):
    values, base = explain.linear_shap(model, X_dev, X_test)
    by_concept = explain.grouped(values)
    imp = explain.importance(by_concept)
    imp.to_csv(tables / "shap_importance_grouped_test.csv")
    values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_csv(
        tables / "shap_importance_columns_test.csv")

    shown = explain.display_values(customers, X_test)
    top3 = list(imp.index[:3])
    levels = pd.concat([explain.level_table(n, by_concept[n], shown[n], p_test, y_test)
                        for n in top3], ignore_index=True)
    levels.to_csv(tables / "shap_top3_levels_test.csv", index=False)

    plots.shap_importance(imp, figures / "shap_importance.png")
    plots.shap_beeswarm(values, X_test, figures / "shap_beeswarm.png")
    plots.shap_top_features(by_concept, shown, top3, figures / "shap_top3.png")

    additivity = float(np.abs(values.sum(axis=1).to_numpy() + base
                              - model.decision_function(X_test)).max())
    return {"base_value_log_odds": base, "importance": imp.round(4).to_dict(), "top3": top3,
            "max_additivity_error": additivity}


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-boot", type=int, default=config.N_BOOT)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)

    started = time.time()
    tables, figures = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    extract = load_extract(args.data_dir)
    labelled = extract.labelled_customers()
    y = labelled[config.TARGET]
    dev, test = split_customers(labelled, args.seed)
    y_dev, y_test = y.loc[dev].to_numpy(), y.loc[test].to_numpy()
    print(f"{len(labelled):,} customers -> {len(dev):,} development / {len(test):,} test")

    summary = {"split": {"development": len(dev), "test": len(test),
                         "dev_positives": int(y_dev.sum()), "test_positives": int(y_test.sum())}}
    summary["profile"] = profiling.run(extract, dev, tables, figures)
    print("profiling done")

    X = build_features(extract.customers, extract.billing)
    X_ext = build_features(extract.customers, extract.billing, extended=True)
    X_raw = build_features(extract.customers, extract.billing, point_in_time=False)
    summary["features"] = list(X.columns)

    feature_sets = {"core": X.loc[dev], "extended": X_ext.loc[dev], "as_recorded": X_raw.loc[dev]}
    comparison, oof, fold_auc = compare_models(feature_sets, y_dev, config.N_FOLDS, args.seed,
                                               args.n_boot)
    comparison.to_csv(tables / "model_comparison_dev.csv", index=False)
    summary["model_comparison_dev"] = comparison.set_index("model").to_dict(orient="index")
    summary["chosen_model_fold_auc"] = fold_auc
    p_dev = oof[CHOSEN]
    # The same summary on 20,000 out-of-fold predictions: a steadier read than one test sample.
    summary["dev_oof"] = ev.score_summary(y_dev, p_dev)
    print("model comparison done")

    model = models.logistic().fit(X.loc[dev], y_dev)
    p_test = model.predict_proba(X.loc[test])[:, 1]
    pd.DataFrame({"customer_id": test, "p_shutoff": p_test}).to_csv(
        tables / "test_predictions.csv", index=False)

    coefs = pd.DataFrame({"feature": X.columns, "concept": [feature_group(c) for c in X.columns],
                          "coef_per_sd": model[-1].coef_[0]})
    coefs["odds_ratio_per_sd"] = np.exp(coefs["coef_per_sd"])
    # Per unit of the raw feature: for 0/1 columns this is "has it vs doesn't".
    coefs["odds_ratio_per_unit"] = np.exp(coefs["coef_per_sd"] / model[0].scale_)
    coefs.sort_values("coef_per_sd", key=np.abs, ascending=False).to_csv(
        tables / "model_coefficients.csv", index=False)

    scores = ev.bootstrap(y_test, p_test, ev.CORE_METRICS, args.n_boot, args.seed)
    scores.to_csv(tables / "test_metrics_ci.csv")
    summary["test"] = {**ev.score_summary(y_test, p_test), "ci": scores.to_dict(orient="index")}
    calibration = ev.calibration_table(y_test, p_test, n_bins=10)
    calibration.to_csv(tables / "test_calibration.csv", index=False)
    plots.roc_pr(y_test, p_test, figures / "test_roc_pr.png")
    print("test evaluation done")

    late_count = np.rint(X["overdue_share"] * (config.CUTOFF_CYCLE - 1)).astype(int)
    summary["threshold"] = threshold_analysis(
        y_dev, p_dev, y_test, p_test, late_count.loc[dev].to_numpy(),
        late_count.loc[test].to_numpy(), len(labelled), args.n_boot, args.seed, tables, figures)
    print("threshold analysis done")

    sub, sub_cal, members_share = subgroup_analysis(
        labelled, dev, test, y_dev, p_dev, y_test, p_test, args.n_boot, args.seed, tables)
    summary["subgroups"] = {"table": sub.to_dict(orient="records"),
                            "members_share_of_customers_test": float(labelled.loc[test, "hardship_plan"].mean()),
                            "members_share_of_calls_test": members_share}
    plots.calibration(calibration, sub_cal, summary["test"]["brier"], figures / "test_calibration.png")

    summary["shap"] = shap_analysis(model, extract.customers, X.loc[dev], X.loc[test],
                                    p_test, y_test, tables, figures)
    summary["runtime_seconds"] = round(time.time() - started, 1)

    with open(args.out / "summary.json", "w") as fh:
        json.dump(to_jsonable(summary), fh, indent=2)

    t = summary["test"]
    rec = summary["threshold"]["test_recommended"]
    print(f"\ntest AUC {t['auc']:.3f} [{t['ci']['auc']['ci_low']:.3f}, {t['ci']['auc']['ci_high']:.3f}]"
          f"  AUPRC {t['auprc']:.3f} [{t['ci']['auprc']['ci_low']:.3f}, {t['ci']['auprc']['ci_high']:.3f}]"
          f"  Brier {t['brier']:.4f}")
    print(f"threshold {summary['threshold']['break_even_threshold']:.4f}: call {rec['called_share']:.1%}, "
          f"reach {rec['recall']:.1%}, ${rec['cost_per_customer']:.2f} per customer")
    print(f"top SHAP concepts: {', '.join(summary['shap']['top3'])}")
    print(f"done in {summary['runtime_seconds']}s -> {args.out}")
    return 0
