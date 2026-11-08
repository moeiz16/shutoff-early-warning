# Field-crew tickets: asset, priority and cause

**The task.** Crews write short tickets such as:

> "Crew attended Mill Lane on Thursday for the distribution panel. No storm damage was observed nearby. Engineer confirmed vehicle impact. Impact: area cordoned off. The neighbouring storm drain pump was inspected and is working normally."
>
> Answer: asset = panel, cause = third_party, priority = emergency.

`simulate.py` generates 120 of these, and 30 are set aside with labels as a reference sample. One cause phrasing only ever appears in those 30, which matters for the strict check below.

**Approach.** `extract.py` goes sentence by sentence. It first decides what a sentence is doing:
- the opening line, which names the asset;
- a cause line;
- a priority line;
- a side remark ("the neighbouring ... was inspected");
- a negation ("No storm damage was observed").

Only then does it map the key phrase through a short list. A cause phrase that isn't listed gets a second try from generic keywords, applied *inside that sentence only*. Anything still unresolved is left blank and flagged `needs_review` instead of guessed. Hedged causes ("Probable cause: X, pending manufacturer report") keep their cause and get `cause_confirmed: false`. All 120 tickets are in `extracted.jsonl`, with the evidence phrase for each field.

**Agreement with the 30 reference tickets** (`agreement.csv`):

| | asset | priority | cause |
|---|---|---|---|
| Keyword search on label names (first attempt) | 30/30 | 12/30 | 11/30 |
| Sentence roles + phrase lists | 30/30 | 30/30 | 30/30 |
| Same, lists rebuilt from the 90 unlabelled tickets only | 30/30 | 30/30 | 30/30 |

With 30 tickets, 30/30 has a 95% Wilson interval of 89–100%. Because I wrote the generator, the middle row is not a blind test. The last row is the fairer one: restricted to phrases seen in the other 90 tickets, the phrase "a firmware bug in a recently fitted controller" (4 reference tickets) drops out of the list, and the in-sentence fallback still labels all four correctly.

**Where the keyword search fails** (`disagreements.csv`):
- **Words in the wrong role**:
  - "Crew attended..." pulls the cause to workmanship;
  - "storm drain pump", whether it's the asset or a neighbouring one, pulls it to weather;
  - "pending manufacturer report" pulls it to manufacturer_defect.
- **Negation:** "No storm damage was observed" reads as weather.
- **A vocabulary gap:** priorities are written as "cosmetic only" or "area cordoned off", almost never as routine or urgent, so 18 of 30 come back blank.

The extractor's own weak spots are judgement calls and brittleness. Nothing states a priority scale, so "safety risk flagged" → urgent rather than emergency is my reading. So is "a firmware bug in a recently fitted controller" → manufacturer_defect rather than wear. Real crews' phrasing would go to review far more often than these generated tickets do. The next step would be a small classifier or language model for the `needs_review` tickets only, with a second person labelling a fresh sample to see how much people disagree with each other.
