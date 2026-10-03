"""
Synthetic data generator for a Netflix-STYLE A/B test.

!! ALL DATA IS SIMULATED. It is not Netflix data and the effect sizes are
!! assumptions chosen for illustration, not real Netflix results.

Experiment: "Personalized artwork on the homepage"
  Control (A):   one default thumbnail per title
  Treatment (B): thumbnail chosen per member based on viewing history
Randomisation unit: member account, 50/50, 28-day test window.

Built-in "ground truth" (so the analysis can be checked against it):
  - New members (tenure < 90d): +5% streaming hours, constant over time
  - Tenured members: +1.2% long-run lift PLUS a novelty bump that decays
    (+5% extra on day 1, half-life ~3.5 days)
  - Play-start rate (play starts / sessions): +1.5 percentage points
  - 28-day cancellation: 5% relative reduction (small, likely not significant)
  - Playback error rate: no change (guardrail)
  - Homepage load time: +14 ms in treatment (guardrail cost of image personalisation)
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_MEMBERS = 80_000
N_DAYS = 28
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------- member attributes ----------------
member_id = np.arange(1, N_MEMBERS + 1)
variant = np.where(rng.random(N_MEMBERS) < 0.5, "control", "treatment")
is_t = variant == "treatment"

tenure = rng.choice(["new", "tenured"], N_MEMBERS, p=[0.2, 0.8])
is_new = tenure == "new"
device = rng.choice(["tv", "mobile", "web"], N_MEMBERS, p=[0.55, 0.30, 0.15])
region = rng.choice(["APAC", "EMEA", "NA", "LATAM"], N_MEMBERS, p=[0.25, 0.30, 0.30, 0.15])
plan = rng.choice(["basic", "standard", "premium"], N_MEMBERS, p=[0.35, 0.45, 0.20])

# latent viewing propensity (hours per 28 days)
base_mu = np.where(is_new, np.log(18), np.log(26))
base_mu = base_mu + np.select([device == "tv", device == "mobile"], [0.15, -0.10], -0.20)
base_mu = base_mu + np.select([plan == "premium", plan == "standard"], [0.20, 0.05], -0.10)
propensity = np.exp(rng.normal(base_mu, 0.75))  # heavy right tail, like real viewing

# pre-period (28 days before test) -- used for CUPED; noisy view of propensity
pre_hours = propensity * rng.gamma(shape=6, scale=1 / 6, size=N_MEMBERS)

# ---------------- cancellation ----------------
p_cancel = np.where(is_new, 0.060, 0.025)
p_cancel = np.where(is_t, p_cancel * 0.95, p_cancel)
cancelled = rng.random(N_MEMBERS) < p_cancel
cancel_day = np.where(cancelled, rng.integers(1, N_DAYS + 1, N_MEMBERS), N_DAYS + 1)

# ---------------- daily streaming hours ----------------
days = np.arange(1, N_DAYS + 1)
lift_tenured = 0.012 + 0.05 * np.exp(-(days - 1) / 5)   # novelty decays
lift_new = np.full(N_DAYS, 0.05)
lift = np.where(is_new[:, None], lift_new[None, :], lift_tenured[None, :])
mult = 1 + lift * is_t[:, None]

daily_mean = (propensity / N_DAYS)[:, None] * mult
watched_today = rng.random((N_MEMBERS, N_DAYS)) < 0.62          # not everyone watches daily
daily = np.where(watched_today, rng.gamma(2.0, daily_mean / 2.0 / 0.62), 0.0)
active = days[None, :] < cancel_day[:, None]                     # zero after cancelling
daily = daily * active
hours_28d = daily.sum(axis=1)

# ---------------- sessions, play starts, guardrails ----------------
sessions = rng.poisson(1.6 * watched_today.sum(axis=1) * active.mean(axis=1) + 2)
p_play = np.clip(rng.normal(0.62, 0.08, N_MEMBERS) + 0.015 * is_t, 0.05, 0.98)
play_starts = rng.binomial(sessions, p_play)
playback_errors = rng.binomial(play_starts, 0.012)
homepage_load_ms = rng.lognormal(np.log(820), 0.25, N_MEMBERS) + 14 * is_t

members = pd.DataFrame({
    "member_id": member_id, "variant": variant, "tenure_segment": tenure,
    "primary_device": device, "region": region, "plan": plan,
    "pre_hours_28d": pre_hours.round(3), "hours_28d": hours_28d.round(3),
    "sessions": sessions, "play_starts": play_starts,
    "playback_errors": playback_errors, "homepage_load_ms": homepage_load_ms.round(1),
    "cancelled": cancelled.astype(int),
    "cancel_day": np.where(cancelled, cancel_day, np.nan),
})
members.to_csv(OUT / "members.csv", index=False)

# daily aggregate for novelty-effect analysis (per arm x tenure x day)
rows = []
for v in ["control", "treatment"]:
    for t in ["new", "tenured"]:
        m = (variant == v) & (tenure == t)
        for d in range(N_DAYS):
            act = active[m, d]
            rows.append((d + 1, v, t, int(act.sum()), float(daily[m, d][act].sum()),
                         float((daily[m, d][act] ** 2).sum())))
pd.DataFrame(rows, columns=["day", "variant", "tenure_segment", "active_members",
                            "total_hours", "sum_sq_hours"]).to_csv(OUT / "daily_by_arm.csv", index=False)

print(members.groupby("variant").agg(n=("member_id", "size"), hours=("hours_28d", "mean"),
                                     pre=("pre_hours_28d", "mean")))
print("pre/post corr:", np.corrcoef(pre_hours, hours_28d)[0, 1].round(3))
