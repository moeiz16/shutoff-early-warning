"""Pull asset, priority and cause out of field-crew tickets.

Tickets follow a handful of sentence patterns, so this works sentence by
sentence: first decide what a sentence is doing (opening the ticket and naming
the asset, giving a cause, setting a priority, or just mentioning something
nearby), then map its key phrase to a label. A phrase that isn't in the lists
below is left blank and the ticket is flagged for review rather than guessed at.

    python -m tickets.extract --data-dir data --out tickets/extracted.jsonl
"""
import argparse
import json
import re
from pathlib import Path

ASSETS = {
    "smart meter": "meter",
    "pole-mounted transformer": "transformer",
    "isolation valve": "valve",
    "storm drain pump": "pump",
    "street light": "streetlight",
    "fire hydrant": "hydrant",
    "distribution panel": "panel",
    "service connection": "service_line",
}

CAUSES = {
    "corroded terminals": "wear",
    "a perished gasket": "wear",
    "worn bearings": "wear",
    "fatigued mounting bolts": "wear",
    "age-related cracking": "wear",
    "a loose connection left by a previous crew": "workmanship",
    "supply left part-isolated after works": "workmanship",
    "the wrong fuse fitted at the last visit": "workmanship",
    "a lightning strike nearby": "weather",
    "storm debris": "weather",
    "frost heave": "weather",
    "flood water in the chamber": "weather",
    "an excavator strike by a contractor": "third_party",
    "vehicle impact": "third_party",
    "vandalism to the housing": "third_party",
    "a factory defect in the new unit": "manufacturer_defect",
    # Judgement call: new kit failing on its firmware is the supplier's problem, not wear.
    "a firmware bug in a recently fitted controller": "manufacturer_defect",
}

# Tickets never state a priority scale, so this ordering is my reading of the phrases.
PRIORITIES = {
    "cosmetic only": "routine",
    "no customer impact": "routine",
    "minor": "routine",
    "can wait for planned work": "routine",
    "moderate": "soon",
    "customers inconvenienced": "soon",
    "partial supply affected": "soon",
    "significant": "urgent",
    "supply lost to one street": "urgent",
    "safety risk flagged": "urgent",
    "emergency": "emergency",
    "public safety hazard": "emergency",
    "area cordoned off": "emergency",
    "supply lost across the district": "emergency",
}

# Second chance for a cause phrase that isn't listed above. Generic words for each
# label, applied only inside the cause sentence and never to the whole ticket:
# across the whole text "storm" also matches "storm drain pump" and "No storm damage".
CAUSE_KEYWORDS = {
    "wear": ["corrod", "perish", "worn", "wear", "fatigue", "age-related", "crack"],
    "workmanship": ["left by", "left part", "wrong", "previous crew", "workmanship", "not reset"],
    "weather": ["lightning", "storm", "frost", "flood", "high wind", "icing"],
    "third_party": ["excavator", "contractor", "vehicle", "vandal", "third party"],
    "manufacturer_defect": ["factory", "defect", "firmware", "manufactur", "recently fitted", "new unit"],
}

OPENING = re.compile(r"^(crew attended|customer reported a problem with|ticket raised)\b", re.I)
SIDE_MENTION = re.compile(r"\bneighbouring\b.*\binspected\b", re.I)  # "The neighbouring meter was inspected..."
NEGATED = re.compile(r"^no\b.*\b(observed|found)\b", re.I)            # "No storm damage was observed..."
NO_CAUSE = re.compile(r"no cause identified|could not be confirmed", re.I)
CAUSE = re.compile(r"^(fault traced to|on arrival the crew found|engineer confirmed|probable cause:)\s*(.+?)\.?$", re.I)
HEDGE = re.compile(r",?\s*pending manufacturer report$", re.I)
GRADE = re.compile(r"^(priority:|rated:|impact:)\s*(.+?)\.?$", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text.strip()) if s.strip()]


def find_asset(sents: list[str]):
    candidates = [s for s in sents if OPENING.match(s)] or [s for s in sents if not SIDE_MENTION.search(s)]
    for sent in candidates:
        lowered = sent.lower()
        # Longest phrase first, so a longer asset name is never read as a shorter one.
        for phrase in sorted(ASSETS, key=len, reverse=True):
            if phrase in lowered:
                return ASSETS[phrase], sent
    return None, None


def cause_from_keywords(phrase: str):
    """Label from generic words, but only if exactly one label's words appear."""
    hits = {label for label, words in CAUSE_KEYWORDS.items() if any(w in phrase for w in words)}
    return hits.pop() if len(hits) == 1 else None


def find_cause(sents: list[str], phrases: dict = CAUSES):
    """Returns (label, confirmed, evidence, used_fallback).

    confirmed is False for a hedged 'Probable cause: ..., pending manufacturer report'
    and None when no cause was found.
    """
    for sent in sents:
        if NEGATED.match(sent) or SIDE_MENTION.search(sent):
            continue
        if NO_CAUSE.search(sent):
            return "undetermined", None, sent, False
        match = CAUSE.match(sent)
        if match:
            lead, phrase = match.group(1).lower(), match.group(2)
            hedged = lead == "probable cause:" or bool(HEDGE.search(phrase))
            phrase = HEDGE.sub("", phrase).strip().lower()
            if phrase in phrases:
                return phrases[phrase], not hedged, phrase, False
            label = cause_from_keywords(phrase)
            return label, (not hedged) if label else None, phrase, label is not None
    return None, None, None, False


def find_priority(sents: list[str], phrases: dict = PRIORITIES):
    for sent in sents:
        match = GRADE.match(sent)
        if match:
            phrase = match.group(2).strip().lower()
            return phrases.get(phrase), phrase
    return None, None


def extract(ticket: dict, cause_phrases: dict = CAUSES, priority_phrases: dict = PRIORITIES) -> dict:
    sents = sentences(ticket["text"])
    asset, asset_evidence = find_asset(sents)
    cause, confirmed, cause_evidence, fallback = find_cause(sents, cause_phrases)
    priority, priority_evidence = find_priority(sents, priority_phrases)
    fields = {"asset": asset, "priority": priority, "cause": cause}
    return {
        "ticket_id": ticket["ticket_id"],
        **fields,
        "cause_confirmed": confirmed,
        "needs_review": [name for name, value in fields.items() if value is None],
        "keyword_fallback": ["cause"] if fallback else [],
        "evidence": {"asset": asset_evidence, "cause": cause_evidence, "priority": priority_evidence},
    }


def read_jsonl(path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract fields from field-crew tickets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "extracted.jsonl")
    args = parser.parse_args(argv)

    tickets = read_jsonl(args.data_dir / "tickets.jsonl")
    rows = [extract(t) for t in tickets]
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    flagged = [r["ticket_id"] for r in rows if r["needs_review"]]
    fallback = [r["ticket_id"] for r in rows if r["keyword_fallback"]]
    hedged = sum(r["cause_confirmed"] is False for r in rows)
    print(f"{len(rows)} tickets -> {args.out}")
    print(f"needs review: {len(flagged)} {flagged if flagged else ''}")
    print(f"cause labelled by keyword fallback: {len(fallback)} {fallback if fallback else ''}")
    print(f"hedged causes (cause_confirmed=false): {hedged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
