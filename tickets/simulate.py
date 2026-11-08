"""Simulated field-crew tickets with known labels.

120 short tickets written the way crews write them: an opening line naming the
asset, a line on the cause, a line on priority, and sometimes a remark about
something else nearby. 30 are set aside as a labelled reference sample. One
cause phrasing only ever appears in the reference sample, so a phrase list
built from the other 90 can't simply look it up.

    python -m tickets.simulate --out data
"""
import argparse
import json
from pathlib import Path

import numpy as np

N_TICKETS = 120
N_REFERENCE = 30
SEED = 7

ASSETS = {
    "meter": "smart meter",
    "transformer": "pole-mounted transformer",
    "valve": "isolation valve",
    "pump": "storm drain pump",
    "streetlight": "street light",
    "hydrant": "fire hydrant",
    "panel": "distribution panel",
    "service_line": "service connection",
}
CAUSES = {
    "wear": ["corroded terminals", "a perished gasket", "worn bearings", "fatigued mounting bolts",
             "age-related cracking"],
    "workmanship": ["a loose connection left by a previous crew", "supply left part-isolated after works",
                    "the wrong fuse fitted at the last visit"],
    "weather": ["a lightning strike nearby", "storm debris", "frost heave", "flood water in the chamber"],
    "third_party": ["an excavator strike by a contractor", "vehicle impact", "vandalism to the housing"],
    "manufacturer_defect": ["a factory defect in the new unit",
                            "a firmware bug in a recently fitted controller"],
}
REFERENCE_ONLY = "a firmware bug in a recently fitted controller"
PRIORITIES = {
    "routine": ["cosmetic only", "no customer impact", "minor", "can wait for planned work"],
    "soon": ["customers inconvenienced", "partial supply affected", "moderate"],
    "urgent": ["significant", "supply lost to one street", "safety risk flagged"],
    "emergency": ["emergency", "public safety hazard", "area cordoned off",
                  "supply lost across the district"],
}

OPENINGS = [
    "Crew attended {loc} {day} for the {asset}.",
    "Customer reported a problem with the {asset} at {loc} {day}.",
    "Ticket raised {day} for the {asset} at {loc}.",
]
CAUSE_LINES = ["Fault traced to {cause}.", "On arrival the crew found {cause}.", "Engineer confirmed {cause}."]
HEDGED_LINE = "Probable cause: {cause}, pending manufacturer report."
UNKNOWN_LINES = ["No cause identified yet; follow-up booked.", "Cause could not be confirmed on site."]
PRIORITY_LINES = ["Priority: {p}.", "Rated: {p}.", "Impact: {p}."]
SIDE_LINE = "The neighbouring {asset} was inspected and is working normally."
NEGATION_LINE = "No storm damage was observed nearby."
LOCATIONS = ["Harbour Road", "the north reservoir", "Station Street", "the depot", "Mill Lane",
             "the business park"]
DAYS = ["on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday", "overnight",
        "at the weekend"]


def _pick(rng, options):
    return str(options[rng.integers(len(options))])


def simulate(n_tickets: int = N_TICKETS, n_reference: int = N_REFERENCE, seed: int = SEED):
    rng = np.random.default_rng(seed)
    ids = [f"FT-{2201 + i}" for i in range(n_tickets)]
    reference = sorted(str(t) for t in rng.choice(ids, n_reference, replace=False))
    special = {str(t) for t in rng.choice(reference, 4, replace=False)}

    tickets = []
    for tid in ids:
        asset = _pick(rng, list(ASSETS))
        priority = _pick(rng, list(PRIORITIES))
        hedged = False
        if tid in special:
            cause, phrase = "manufacturer_defect", REFERENCE_ONLY
        elif rng.random() < 0.12:
            cause, phrase = "undetermined", None
        else:
            cause = _pick(rng, list(CAUSES))
            phrase = _pick(rng, [p for p in CAUSES[cause] if p != REFERENCE_ONLY])
            hedged = rng.random() < 0.12

        lines = [_pick(rng, OPENINGS).format(loc=_pick(rng, LOCATIONS), day=_pick(rng, DAYS),
                                             asset=ASSETS[asset])]
        if cause != "weather" and rng.random() < 0.12:
            lines.append(NEGATION_LINE)
        if cause == "undetermined":
            lines.append(_pick(rng, UNKNOWN_LINES))
        elif hedged:
            lines.append(HEDGED_LINE.format(cause=phrase))
        else:
            lines.append(_pick(rng, CAUSE_LINES).format(cause=phrase))
        lines.append(_pick(rng, PRIORITY_LINES).format(p=_pick(rng, PRIORITIES[priority])))
        if rng.random() < 0.15:
            other = _pick(rng, [a for a in ASSETS if a != asset])
            lines.append(SIDE_LINE.format(asset=ASSETS[other]))

        tickets.append({"ticket_id": tid, "text": " ".join(lines), "asset": asset,
                        "priority": priority, "cause": cause})
    return tickets, set(reference)


def write(out_dir, seed: int = SEED) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickets, reference = simulate(seed=seed)
    with open(out_dir / "tickets.jsonl", "w") as fh:
        for t in tickets:
            fh.write(json.dumps({"ticket_id": t["ticket_id"], "text": t["text"]}) + "\n")
    with open(out_dir / "tickets_reference.jsonl", "w") as fh:
        for t in tickets:
            if t["ticket_id"] in reference:
                fh.write(json.dumps(t) + "\n")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the simulated field-crew tickets.")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    write(args.out, args.seed)
    print(f"{N_TICKETS} tickets ({N_REFERENCE} labelled for reference) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
