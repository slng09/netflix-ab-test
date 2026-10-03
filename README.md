# Personalized Artwork A/B Test (Netflix-style, simulated data)

**[Open the interactive dashboard](https://slng09.github.io/netflix-ab-test/dashboard.html)**: filter by tenure, device and region, switch between CUPED and unadjusted estimates, explore lift over time, and try the sample-size calculator.

An end-to-end A/B test analysis for a streaming homepage. The question: does showing each member a
**personalized thumbnail** for each title, instead of a single default image, increase viewing?

> **Data note:** the data is **simulated** (`generate_data.py`, seed 42). Netflix doesn't publish
> experiment data. The scenario is inspired by Netflix's public writing on artwork personalisation, but
> every number here comes from my own simulation with assumed effect sizes. Because I know the true
> effects, I can check whether the analysis recovers them.

## Result at a glance

| | Result |
|---|---|
| **Decision** | Ship |
| Streaming hours / member (CUPED) | **+2.76%** (95% CI +1.94% to +3.58%), p < 0.0001 |
| Steady-state estimate (week 4) | **+1.99%**, lower than the 28-day average because of a novelty effect |
| Play-start rate | +1.59 percentage points (62.0% → 63.6%) |
| 28-day cancellation | −5.5%, **inconclusive** (CI crosses zero) |
| Guardrails | Load time +15 ms (within the 50 ms margin); playback errors within the 10% margin |

## What this project demonstrates

- **Experiment design before data:** hypothesis, primary/secondary/guardrail metrics, a decision rule set in advance, and a power analysis (MDE ≈ 2.1% at 40k members per arm)
- **Trust checks:** sample ratio mismatch (chi-square) and pre-period balance
- **Inference:** Welch t-test, delta-method CIs for relative lift, and a bootstrap cross-check
- **CUPED variance reduction:** 65% variance reduction, a 41% narrower CI (cross-checked with regression adjustment)
- **Ratio metrics:** delta method for play-start rate (member-level randomisation, session-level denominator)
- **Multiple testing:** Holm correction across secondary metrics, with segment cuts labelled as exploratory
- **Non-inferiority guardrails:** including a real *false positive* (playback errors, p ≈ 0.04, true effect zero) and why margins handle it
- **Novelty effects:** daily lift by tenure; tenured lift decays from +4.4% (week 1) to +1.2% (week 4)
- **Heterogeneous effects:** new members gain the most (interaction test)
- **Stakeholder communication:** an interactive dashboard (segment filters, CUPED toggle, daily vs cumulative lift, tooltips, sample-size calculator) with a recommendation and limitations

## Repo structure

```
generate_data.py      # reproducible synthetic data (documents the "true" effects)
build_notebook.py     # builds analysis.ipynb from source cells
analysis.ipynb        # full analysis, executed with outputs
results.json          # metrics exported from the notebook (feeds the dashboard)
dashboard.html        # interactive stakeholder dashboard (built by build_dashboard.py)
build_dashboard.py    # precomputes results for all 60 filter combinations and builds dashboard.html
templates/            # dashboard HTML template
data/members.csv      # 80,000 members (not committed: 5.5 MB; created by generate_data.py)
data/daily_by_arm.csv # daily aggregates for the novelty analysis
```

## Run it

```bash
pip install -r requirements.txt
python generate_data.py      # creates data/members.csv (the notebook also does this if it's missing)
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
python build_dashboard.py   # rebuilds dashboard.html
```

## Limitations

Simulated data; a 28-day window is too short for retention (a long-term holdout would be the next step);
account-level randomisation ignores shared household profiles; and device/region cuts are exploratory.
