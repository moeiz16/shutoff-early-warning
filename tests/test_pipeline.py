import json

from earlywarning.data import load_extract
from earlywarning.pipeline import main, split_customers


def _run(data_dir, out):
    assert main(["--data-dir", str(data_dir), "--out", str(out), "--n-boot", "30"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    summary.pop("runtime_seconds")
    return summary


def test_end_to_end_on_a_small_simulation(small_data_dir, tmp_path):
    summary = _run(small_data_dir, tmp_path)

    assert 0.5 < summary["test"]["auc"] <= 1
    ci = summary["test"]["ci"]["auc"]
    assert ci["ci_low"] <= summary["test"]["auc"] <= ci["ci_high"]
    assert summary["threshold"]["break_even_threshold"] == 0.08
    assert summary["shap"]["max_additivity_error"] < 1e-8
    assert len(summary["shap"]["top3"]) == 3
    assert len(summary["chosen_model_fold_auc"]) == 5

    for name in ["profile_overview", "test_calibration", "test_roc_pr", "threshold_cost_curve",
                 "shap_importance", "shap_beeswarm", "shap_top3"]:
        assert (tmp_path / "figures" / f"{name}.png").exists(), name
    for name in ["model_comparison_dev", "test_metrics_ci", "threshold_policies_test",
                 "subgroup_hardship_plan", "shap_importance_grouped_test"]:
        assert (tmp_path / "tables" / f"{name}.csv").exists(), name


def test_same_inputs_give_the_same_numbers(small_data_dir, tmp_path):
    assert _run(small_data_dir, tmp_path / "a") == _run(small_data_dir, tmp_path / "b")


def test_split_is_disjoint_and_keeps_group_rates(small_data_dir):
    labelled = load_extract(small_data_dir).labelled_customers()
    dev, test = split_customers(labelled)
    assert len(dev.intersection(test)) == 0
    assert len(dev) + len(test) == len(labelled)
    for col in ["shutoff_next_6", "hardship_plan"]:
        assert abs(labelled.loc[dev, col].mean() - labelled.loc[test, col].mean()) < 0.01
