import pandas as pd
import pytest

from earlywarning.data import Extract, check_extract, harmonize_tariffs, load_extract


def test_clean_simulated_extract_passes(small_extract):
    assert check_extract(Extract(*small_extract)) == []


def test_duplicate_customer_cycle_is_caught(small_extract):
    customers, billing, labels = small_extract
    billing = pd.concat([billing, billing.iloc[[0]]], ignore_index=True)
    problems = check_extract(Extract(customers, billing, labels))
    assert any("duplicate customer-cycle" in p for p in problems)


def test_missing_cycle_is_caught(small_extract):
    customers, billing, labels = small_extract
    first = billing["customer_id"].iloc[0]
    billing = billing.drop(index=billing.index[(billing.customer_id == first) & (billing.cycle == 3)])
    problems = check_extract(Extract(customers, billing, labels))
    assert any("cycles 1-12" in p for p in problems)


def test_mismatched_ids_are_caught(small_extract):
    customers, billing, labels = small_extract
    problems = check_extract(Extract(customers, billing, labels.iloc[1:]))
    assert any("customer_id differs" in p for p in problems)


def test_load_extract_refuses_a_broken_file(tmp_path, small_extract):
    customers, billing, labels = small_extract
    customers.to_csv(tmp_path / "customers.csv", index=False)
    billing.assign(amount_paid=-1).to_csv(tmp_path / "billing.csv", index=False)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(ValueError, match="amount_paid"):
        load_extract(tmp_path)


def test_tariff_codes_are_mapped_to_the_new_names(small_extract):
    _, billing, _ = small_extract
    harmonized = harmonize_tariffs(billing)
    assert set(harmonized.tariff) == {"STD", "FLEX", "CARE"}
    # After mapping, the recode no longer looks like every customer changed tariff.
    wide = harmonized.pivot(index="customer_id", columns="cycle", values="tariff")
    assert ((wide[8] == "CARE") == (wide[9] == "CARE")).all()
