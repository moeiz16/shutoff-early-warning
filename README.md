# Shut-off early warning

A small end-to-end project on a question every utility faces: **which households are about to lose service for non-payment, and who should the support team call first?**

I built it to work through the parts of this kind of problem that are easy to get wrong in practice. Most of them aren't about the model. They're about using only what you would really know on the day you score, and turning a probability into a decision that has a cost attached. The data is simulated (`src/earlywarning/simulate.py`). The simulator deliberately bakes in the sort of problems a real billing extract has, so the pipeline has something real to catch, and it has to find them from the data alone.

## The problem, as I framed it

- A utility bills about 25,000 households once per billing cycle (monthly). If bills go unpaid long enough, service is shut off.
- A shut-off is bad for everyone. The household loses its supply, and the utility loses revenue and pays for crew visits and reconnection.
- A support call (a payment plan, help joining the hardship plan) can prevent some shut-offs, but calls cost staff time.
- **The question:** standing at the end of cycle 12 with a year of billing history, score every household's risk of a non-payment shut-off in cycles 13–18, and decide who gets a call.
- **My cost assumptions:** a call costs $40 of staff time whether or not it was needed. A shut-off that a call would have prevented costs $500 (lost revenue, two crew visits, reconnection). The numbers are mine; section 4 shows how the answer moves if they're wrong.

## The data (simulated)

| File | Rows | Contents |
|---|---|---|
| `customers.csv` | 25,000 | district, home type, occupants (recorded at sign-up), tenure in cycles and hardship-plan flag (both as of cycle 12), an estimated income tier (1–5, sometimes missing), e-billing flag, plus two fields stamped on the day the extract was pulled (end of cycle 18): `shutoff_notices` and `status_at_pull` |
| `billing.csv` | 300,000 | one row per customer per cycle: amount due, amount paid, days overdue when paid, tariff code, payment channel |
| `labels.csv` | 25,000 | `shutoff_next_6`: shut off for non-payment in cycles 13–18 (9.3% of customers) |

Behind the scenes each household has a hidden "struggling to pay" score. Lateness in every cycle and the chance of a shut-off both depend on it, so twelve cycles of bills are a noisy window onto it.

## 1. What profiling turned up, and what I left out

`src/earlywarning/profiling.py` doesn't know the simulator's settings; everything below is found from the files. Checks that look at the label use the development split only.

- **Two fields give the answer away.** `shutoff_notices` and `status_at_pull` are stamped at the end of cycle 18, after the outcome window. With zero notices nobody was shut off, and with two or more everybody was. A status of "shut_off" is the label restated. On their own they reach AUC 0.996 and 0.94. Both are out, but dated warning letters sent before cycle 12 would be fair game.
- **Some payments arrived after the scoring date.** `days_overdue` records how late a bill was *when it was paid*, and the extract was pulled six cycles later. If bills fall due at the end of their cycle, the cycle-12 bill had only just fallen due on the scoring date (7,140 of them were paid late, so after it), and 2,282 bills from cycles 10–11 were also paid after it. I rebuild the history as it stood on the scoring date: cycle-12 payment details are dropped, and payments that hadn't arrived count as unpaid. Taking the extract at face value would add 0.002 AUC (CI 0.001–0.003). It's small, but it's credit for information nobody had yet.
- **Tariff codes were renamed at cycle 9** (A1/A2/A9 became STD/FLEX/CARE, matched by volume), so a "changed tariff" feature would fire for everyone. Once mapped, CARE is exactly the 3,399 hardship-plan customers. STD vs FLEX repeats from one cycle to the next 54.4% of the time, against 54.5% if it were drawn at random. Real tariffs don't behave like that, so the field is dropped.
- **Income tier is missing for 8.9% of customers, and not at random.** It's missing for 22% of hardship-plan members against 7% of others, and those customers are shut off more often (13.1% vs 8.9%). Missing gets its own category instead of an imputed value.
- **Flagged, not fixed:**
  - 868 customers whose tenure says they joined mid-year still have twelve cycles of bills (3,960 dated before "sign-up"). The early bills look like the same household (37.9% late before vs 36.1% after), so I suspect the tenure field.
  - Days overdue piles up at exactly 90 (1,154 bills vs 57 at 89), which looks like an export cap.
  - 133 bills show nothing paid but a lateness value.
  - Occupants is as of sign-up.
- **Behaviour is flat across the year.** Customers later shut off were late on about 59% of bills in every cycle, the rest on about 25%, while amounts due swing with the seasons for both (`outputs/figures/profile_overview.png`). So averages carry the signal, and recent-cycle features add nothing (tested below).

## 2. The model

An 80/20 split, stratified on label × hardship plan (20,000 / 5,000). Every choice was made with 5-fold cross-validation on the development set, and the test set was scored once.

The 24 features are:
- **payment behaviour:** share of the 11 observable bills paid late, average days overdue, share of the amount due left unpaid at the cutoff, and the average bill;
- **how bills are paid:** the channel mix;
- **customer fields:** income tier with a missing category, home type, district, occupants, log tenure, hardship plan and e-billing.

| Development, out-of-fold | AUC | AUPRC | Brier | AUC vs chosen (95% CI) |
|---|---|---|---|---|
| **Logistic regression (chosen)** | 0.847 | 0.413 | 0.0669 | |
| + recent-cycle and extreme-value features | 0.847 | 0.412 | 0.0670 | −0.0001 (−0.0007, +0.0005) |
| Gradient boosting, same + extra features | 0.838 | 0.389 | 0.0680 | −0.009 (−0.012, −0.006) |
| Logistic, payments as recorded (leaky) | 0.849 | 0.419 | 0.0665 | +0.002 (+0.001, +0.003) |
| Rule of thumb: number of late bills | 0.835 | 0.358 | | −0.012 (−0.015, −0.008) |

Boosting did worse, which fits risk that adds up from a few rates. The logistic model is at least as accurate, is calibrated without help (no class weights, so the probabilities stay honest), and every coefficient can be read.

## 3. Performance

**Test set** (5,000 customers, 464 shut off):

| Metric | Value (95% CI) | Baseline |
|---|---|---|
| AUC | **0.822** (0.803–0.841) | |
| AUPRC | **0.377** (0.332–0.425) | 0.093 (the base rate) |
| Brier | **0.0701** (0.0651–0.0756) | 0.0842 (predicting the base rate) |

Intervals come from 2,000 bootstrap resamples.

**Why the test AUC sits below the cross-validated 0.847:** across the five development folds of 4,000 customers, AUC ranged from 0.836 to 0.857, so 0.822 on 5,000 is an unlucky draw rather than overfitting (24 coefficients, 20,000 rows). I kept the split I'd fixed in advance rather than re-rolling it.

**Calibration:**
- The same sample makes the test calibration slope 0.88: the top tenth is over-predicted (44% predicted vs 39% observed).
- On the 20,000 out-of-fold predictions, the slope is 0.99, and the mean prediction is 9.28% vs 9.29% observed (`outputs/figures/test_calibration.png`).

## 4. Who to call

Call when the expected avoidable loss beats the cost of the call: p × $500 > $40, so **p ≥ 0.08**. That only works because the probabilities are calibrated.

As a check, the development cost curve bottoms out at 0.0875 and stays within 2% of its minimum from 0.055 to 0.108; at 0.08 it costs 0.3% more. I keep 0.08 because it comes from the costs, not from noise in one sample.

| Test set | Called | Shut-offs reached | Cost per customer |
|---|---|---|---|
| Call nobody | 0% | 0% | $46.40 |
| Call everyone | 100% | 100% | $40.00 |
| Rule of thumb: 5+ late bills (cut-off picked on development data) | 27.5% | 69.2% | $25.28 |
| **Model, p ≥ 0.08** | **30.4%** | **73.7%** (69.8–77.7) | **$24.36** (22.28–26.53) |

That comes to $121,800 for the 5,000 test customers, or about $609k per half-year across all 25,000, against $1.16M for doing nothing.

**The model vs a simple rule.** On this test sample the model beats the "5+ late bills" rule by $0.92 per customer, but the interval (−$0.18 to $2.14) includes zero. The model ranks clearly better (+0.012 AUC on the development folds) and gives a probability you can plan capacity around, but in money terms the rule is a serious baseline, and I'd say so. The model's own expected cost on test was $21.97, below the realised $24.36, which matches the weaker test sample.

**Two assumptions matter more than the model:**
- **Do calls work?** I assumed every call to a would-be shut-off prevents it. If calls work half the time, the break-even threshold doubles to 0.16 and the list shrinks to 17% of customers. That effectiveness rate is the number I'd most want measured.
- **Capacity.** 30% of customers is about 7,600 calls per half-year. With fewer staff, work down the ranked list:

| Call the top | Shut-offs reached | Calls that reach a would-be shut-off |
|---|---|---|
| 5% | 27% | 50% |
| 10% | 42% | |
| 20% | 61% | |

## 5. Hardship-plan customers

| Test set | Customers (shut off) | Observed | Mean predicted | AUC | Called | Reached | Precision |
|---|---|---|---|---|---|---|---|
| Not on plan | 4,320 (351) | 8.1% | 8.2% | 0.816 (0.792–0.839) | 27% | 69% | 21% |
| On hardship plan | 680 (113) | 16.6% | 17.1% | 0.801 (0.760–0.840) | 55% | 89% | 27% |

Calibration holds in both groups. The ranking is weaker for members, and on the larger development folds the gap is clear: 0.813 (0.792–0.833) vs 0.846 (0.835–0.856). The likely reason is that income tier, one of the strongest inputs, barely varies inside the plan: 76% of members are in the bottom two tiers or have no estimate, against 32% of everyone else.

What I'd tell the support team:
- **More than half of members will be on the list**, and members will be a quarter of all calls while being 14% of customers. That isn't the model singling them out. Their risk really is double (16.6% vs 8.1%), and a score means the same thing for members as for everyone else.
- **Within the plan, treat the ranking as rough.** It sorts members less sharply than other customers, so caseworkers who know these households should have the final say on order.
- **Calls to members land more often:** 27% reach someone who would otherwise be shut off, against 21% for other customers.
- **Compared with otherwise similar households** (same income tier, same payment record), members have about a third lower odds of a shut-off (odds ratio 0.68). That's consistent with the plan working, but it's an observation, not proof.

## 6. What drives the score

The SHAP values are exact for a linear model. I've summed them over each concept's one-hot columns (`outputs/figures/shap_importance.png`, `shap_top3.png`). The top three by mean |SHAP|:

1. **How many bills were paid late.** By far the strongest. Each extra late bill out of 11 adds about 0.23 to the log-odds, roughly a quarter more odds, with everything else held fixed. Average predicted risk is 1% with no late bills, 5% with three, 18% with six and 46% with nine.
2. **Income tier.** Tier 1 averages 29% predicted risk, tier 3 6% and tier 5 1%. Customers with no estimate average 13%, between tiers 2 and 3, which is why imputing "average income" would have been wrong.
3. **How late, not just how often.** Customers in the lowest fifth for average days overdue (under a day) average 1% predicted risk; the highest fifth (12+ days) averages 31%.

Just behind these: paying by post or at a counter roughly doubles the odds compared with direct debit, all else equal. I wouldn't read that as "move people to direct debit". Households on tight budgets often avoid it so they control when money leaves.

## 7. Before using it for real, and what's next

- **Confirm the billing calendar**, which the point-in-time rules depend on, and get dated warning letters.
- **Learn how shut-off decisions are made** (protections, payment plans). The label partly measures policy: 224 development customers shut off in the window were connected again by cycle 18.
- **Validate on a later cohort**, and watch for drift. The tariff recode is exactly the kind of change that breaks a model quietly.
- **Hold a random slice of flagged customers out of outreach.** It's the only way to measure whether calls work, and without it future labels get contaminated by the calls themselves.
- **Use the score for supportive outreach only.** It leans on estimated income; reused for collections, it would fall hardest on the poorest households.

**Next steps:**
- an out-of-time backtest;
- a time-to-event model to decide who to call first;
- an uplift model once call outcomes exist;
- a deliberate look at interactions (income × channel);
- intervals that also bootstrap the training.

## Side project: reading field-crew tickets

The same utility's field crews write short free-text tickets. `tickets/` pulls three fields out of 120 simulated tickets (asset, priority, cause) by working out what each sentence is doing before mapping any phrase. That way it doesn't fall for "No storm damage was observed", a neighbouring asset that was checked, or "pending manufacturer report". It agrees with all 30 labelled reference tickets on every field, against 12/30 and 11/30 for a plain keyword search. The details and failure modes are in [tickets/NOTE.md](tickets/NOTE.md).

## Running it

Python 3.11. `./reproduce.sh` (or `make all`) builds `.venv` from the pinned `requirements.txt`, then:
1. simulates the data;
2. runs the 52 tests;
3. rebuilds `outputs/` and the ticket results.

It takes about 2 minutes. Seeds are fixed, so two runs give identical tables.

```
src/earlywarning/   simulate.py, data checks, profiling, point-in-time features, models,
                    evaluation, SHAP, plots, pipeline.py (python -m earlywarning)
tickets/            simulate.py, extract.py, score.py, NOTE.md, extracted.jsonl, agreement.csv
tests/              unit tests, end-to-end run, reproducibility, checks on the simulator itself
outputs/            summary.json (every number above), tables/*.csv, figures/*.png
```
