"""Agreement with the 30 labelled reference tickets, per field.

Three versions are scored:
  * a plain keyword search on the label vocabulary (the obvious first attempt),
    so the failure modes in the notes are measured rather than asserted;
  * the extractor as written;
  * the extractor with its phrase lists cut down to phrases that occur in the
    90 unlabelled tickets. That's the fair test of whether it copes with
    phrasing it has never seen.

    python -m tickets.score --data-dir data
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .extract import CAUSES, PRIORITIES, extract, read_jsonl

FIELDS = ["asset", "priority", "cause"]

# Keyword search on the label names and their obvious synonyms; earliest hit in the text wins.
NAIVE_KEYWORDS = {
    "asset": {"meter": ["meter"], "transformer": ["transformer"], "valve": ["valve"],
              "pump": ["pump"], "streetlight": ["street light"], "hydrant": ["hydrant"],
              "panel": ["panel"], "service_line": ["service connection"]},
    "priority": {"routine": ["routine", "minor"], "soon": ["soon", "moderate"],
                 "urgent": ["urgent", "significant"], "emergency": ["emergency"]},
    "cause": {"wear": ["wear", "worn"], "workmanship": ["workmanship", "crew"],
              "weather": ["weather", "storm", "lightning", "flood"],
              "third_party": ["third party", "contractor", "vandal"],
              "manufacturer_defect": ["manufacturer", "factory", "defect"],
              "undetermined": ["undetermined", "no cause identified", "could not be confirmed"]},
}


def naive_extract(text: str) -> dict:
    out = {}
    for field, labels in NAIVE_KEYWORDS.items():
        best = (len(text) + 1, None)
        for label, words in labels.items():
            for word in words:
                hit = re.search(rf"\b{re.escape(word)}\b", text, re.I)
                if hit and hit.start() < best[0]:
                    best = (hit.start(), label)
        out[field] = best[1]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def agreement(truth: pd.DataFrame, predicted: pd.DataFrame, method: str) -> list[dict]:
    rows = []
    for field in FIELDS:
        t = truth[field].astype(str)
        p = predicted[field].fillna("<blank>").astype(str)
        k, n = int((t == p).sum()), len(t)
        lo, hi = wilson(k, n)
        rows.append({"method": method, "field": field, "n": n, "agree": k, "accuracy": k / n,
                     "ci_low": lo, "ci_high": hi, "cohen_kappa": cohen_kappa_score(t, p)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score extraction against the reference tickets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "agreement.csv")
    args = parser.parse_args(argv)

    tickets = read_jsonl(args.data_dir / "tickets.jsonl")
    truth = pd.DataFrame(read_jsonl(args.data_dir / "tickets_reference.jsonl")).set_index("ticket_id")
    ids = truth.index
    unlabelled = [t for t in tickets if t["ticket_id"] not in set(ids)]

    # Phrase lists limited to what the 90 unlabelled tickets actually contain.
    seen = [extract(t)["evidence"] for t in unlabelled]
    strict_causes = {k: v for k, v in CAUSES.items() if k in {e["cause"] for e in seen}}
    strict_priorities = {k: v for k, v in PRIORITIES.items() if k in {e["priority"] for e in seen}}
    dropped = sorted(set(CAUSES) - set(strict_causes)) + sorted(set(PRIORITIES) - set(strict_priorities))

    labelled_tickets = [t for t in tickets if t["ticket_id"] in set(ids)]
    versions = {
        "keyword search (baseline)": pd.DataFrame(
            [{"ticket_id": t["ticket_id"], **naive_extract(t["text"])} for t in labelled_tickets]),
        "sentence roles + phrase lists": pd.DataFrame([extract(t) for t in labelled_tickets]),
        "same, lists from the 90 unlabelled only": pd.DataFrame(
            [extract(t, strict_causes, strict_priorities) for t in labelled_tickets]),
    }
    versions = {name: df.set_index("ticket_id").loc[ids] for name, df in versions.items()}

    table = pd.DataFrame([row for name, df in versions.items() for row in agreement(truth, df, name)])
    table.to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))

    misses = []
    for name, df in versions.items():
        for field in FIELDS:
            wrong = df[field].fillna("<blank>") != truth[field]
            misses += [{"method": name, "ticket_id": tid, "field": field,
                        "reference": truth.loc[tid, field], "extracted": df.loc[tid, field]}
                       for tid in ids[wrong]]
    misses = pd.DataFrame(misses, columns=["method", "ticket_id", "field", "reference", "extracted"])
    misses.to_csv(args.out.with_name("disagreements.csv"), index=False)

    strict = versions["same, lists from the 90 unlabelled only"]
    print(f"\nphrases only found in the reference tickets (dropped for the strict run): {dropped}")
    print(f"strict run: {sum(bool(f) for f in strict['keyword_fallback'])} of 30 causes came from "
          f"the keyword fallback, {sum(bool(f) for f in strict['needs_review'])} sent to review")
    print(f"disagreements -> {args.out.with_name('disagreements.csv')} ({len(misses)} rows)")
    if not misses.empty:
        print(misses.groupby(["method", "field"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
