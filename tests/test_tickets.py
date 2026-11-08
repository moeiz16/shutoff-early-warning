import json

import pytest

from tickets import simulate as ticket_sim
from tickets.extract import extract
from tickets.score import naive_extract


def run(text):
    return extract({"ticket_id": "T", "text": text})


def test_neighbouring_asset_is_not_the_subject():
    r = run("Crew attended Mill Lane on Monday for the distribution panel. Fault traced to worn bearings. "
            "Priority: minor. The neighbouring storm drain pump was inspected and is working normally.")
    assert r["asset"] == "panel"
    assert r["cause"] == "wear"


def test_negated_storm_mention_is_not_a_cause():
    r = run("Ticket raised on Friday for the street light at the depot. No storm damage was observed nearby. "
            "No cause identified yet; follow-up booked. Impact: significant.")
    assert r["cause"] == "undetermined"
    assert r["cause_confirmed"] is None


def test_hedged_cause_keeps_its_cause_and_is_marked_unconfirmed():
    r = run("Customer reported a problem with the smart meter at Station Street on Monday. "
            "Probable cause: corroded terminals, pending manufacturer report. Priority: emergency.")
    # "manufacturer" here is about who will report, not what broke
    assert r["cause"] == "wear"
    assert r["cause_confirmed"] is False
    assert r["priority"] == "emergency"


@pytest.mark.parametrize("phrase, label", [
    ("Priority: customers inconvenienced.", "soon"),
    ("Rated: safety risk flagged.", "urgent"),
    ("Impact: area cordoned off.", "emergency"),
    ("Impact: no customer impact.", "routine"),
])
def test_priority_phrases(phrase, label):
    r = run(f"Ticket raised on Monday for the isolation valve at Mill Lane. Fault traced to frost heave. {phrase}")
    assert r["priority"] == label


def test_unlisted_cause_falls_back_to_keywords_inside_the_cause_sentence_only():
    r = run("Crew attended the depot on Monday for the storm drain pump. "
            "Fault traced to a contractor's digger hitting the main. Priority: minor.")
    # "storm" appears in the asset name and "crew" in the opening, but neither may decide the cause
    assert r["cause"] == "third_party"
    assert r["keyword_fallback"] == ["cause"]


def test_ambiguous_or_unknown_phrases_go_to_review_instead_of_a_guess():
    r = run("Ticket raised on Monday for the fire hydrant at the depot. "
            "Fault traced to worn seal damaged by flood water. Priority: somewhat bad.")
    assert r["cause"] is None and r["priority"] is None
    assert set(r["needs_review"]) == {"cause", "priority"}


def test_keyword_search_baseline_shows_the_traps():
    text = ("Crew attended Harbour Road on Monday for the smart meter. "
            "Engineer confirmed vehicle impact. Impact: significant.")
    assert naive_extract(text)["cause"] == "workmanship"  # "crew" in the opening; wrong, and that's the point
    assert run(text)["cause"] == "third_party"


def test_simulated_tickets_round_trip_through_the_extractor():
    tickets, reference = ticket_sim.simulate()
    assert len(tickets) == 120 and len(reference) == 30
    for t in tickets:
        got = extract(t)
        for field in ["asset", "priority", "cause"]:
            assert got[field] == t[field], (t["ticket_id"], field, got[field], t[field])
        json.dumps(got)


def test_one_cause_phrasing_only_appears_in_the_reference_sample():
    tickets, reference = ticket_sim.simulate()
    where = {t["ticket_id"] in reference for t in tickets if ticket_sim.REFERENCE_ONLY in t["text"]}
    assert where == {True}
