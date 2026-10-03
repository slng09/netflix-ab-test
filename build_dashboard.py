"""
Builds dashboard.html: precomputes results for every tenure x device x region filter
combination and embeds them in an interactive, self-contained HTML page.

Re-runs generate_data.py (seed 42, so data is identical) to get member-level daily viewing.
Run:  python build_dashboard.py
"""
import json
import runpy
import numpy as np
from scipy import stats
from pathlib import Path

HERE = Path(__file__).parent
g = runpy.run_path(str(HERE / "generate_data.py"), run_name="dashboard")  # deterministic
variant, tenure, device, region = g["variant"], g["tenure"], g["device"], g["region"]
pre, hours = g["pre_hours"], g["hours_28d"]
daily = g["daily"]                         # members x 28, zero after cancellation (intent-to-treat)
cum = daily.cumsum(axis=1)
sessions, plays, errors = g["sessions"], g["play_starts"], g["playback_errors"]
load_ms, cancelled = g["homepage_load_ms"], g["cancelled"].astype(float)
is_t = variant == "treatment"

ERR_MARGIN, LOAD_MARGIN = 0.10, 50.0
Z = 1.959964


def lift_means(yt, yc):
    """Relative lift of means with delta-method CI (works column-wise on 2-D arrays)."""
    mt, mc = yt.mean(0), yc.mean(0)
    vt, vc = yt.var(0, ddof=1) / len(yt), yc.var(0, ddof=1) / len(yc)
    lift = mt / mc - 1
    se = np.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    return lift, se, mt, mc


def ratio(num, den):
    n, r = len(num), num.sum() / den.sum()
    var = (num.var(ddof=1) - 2 * r * np.cov(num, den)[0, 1] + r**2 * den.var(ddof=1)) / (n * den.mean() ** 2)
    return r, var


def pack(lift, se, extra=None):
    d = {"lift": float(lift), "lo": float(lift - Z * se), "hi": float(lift + Z * se),
         "p": float(2 * stats.norm.sf(abs(lift / se))) if se > 0 else 1.0}
    if extra:
        d.update({k: float(v) for k, v in extra.items()})
    return d


def analyse(mask):
    t, c = mask & is_t, mask & ~is_t
    nt, nc = int(t.sum()), int(c.sum())
    out = {"n_t": nt, "n_c": nc}
    out["srm_p"] = float(stats.chisquare([nc, nt])[1])

    # primary: unadjusted
    l, se, mt, mc = lift_means(hours[t], hours[c])
    out["unadj"] = pack(l, se, {"ctrl": mc, "trt": mt})
    # primary: CUPED (theta estimated within the filtered population)
    X, Y = pre[mask], hours[mask]
    theta = np.cov(X, Y)[0, 1] / X.var(ddof=1)
    adj = hours - theta * (pre - X.mean())
    diff = adj[t].mean() - adj[c].mean()
    se_d = np.sqrt(adj[t].var(ddof=1) / nt + adj[c].var(ddof=1) / nc)
    out["cuped"] = pack(diff / mc, se_d / mc, {"ctrl": mc, "trt": mc + diff,
                                               "var_red": 1 - adj[mask].var() / Y.var()})

    # secondary
    rt, vt = ratio(plays[t].astype(float), sessions[t].astype(float))
    rc, vc = ratio(plays[c].astype(float), sessions[c].astype(float))
    se = np.sqrt(vt + vc)
    out["play"] = pack((rt - rc) / rc, se / rc, {"ctrl": rc, "trt": rt, "pp": (rt - rc) * 100})
    pc, pt = cancelled[c].mean(), cancelled[t].mean()
    se = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    out["cancel"] = pack((pt - pc) / pc if pc > 0 else 0.0, se / pc if pc > 0 else 1.0, {"ctrl": pc, "trt": pt})

    # guardrails (non-inferiority, one-sided 95% upper bound vs margin)
    et, vet = ratio(errors[t].astype(float), plays[t].astype(float))
    ec, vec = ratio(errors[c].astype(float), plays[c].astype(float))
    se = np.sqrt(vet + vec) / ec
    el = et / ec - 1
    out["errors"] = pack(el, se, {"ctrl": ec, "trt": et, "upper": el + 1.645 * se, "margin": ERR_MARGIN})
    out["errors"]["pass"] = bool(el + 1.645 * se < ERR_MARGIN)
    d = load_ms[t].mean() - load_ms[c].mean()
    se = np.sqrt(load_ms[t].var(ddof=1) / nt + load_ms[c].var(ddof=1) / nc)
    mcl = load_ms[c].mean()
    out["load"] = pack(d / mcl, se / mcl, {"ctrl": mcl, "trt": load_ms[t].mean(), "diff_ms": d,
                                           "upper_ms": d + 1.645 * se, "margin_ms": LOAD_MARGIN})
    out["load"]["pass"] = bool(d + 1.645 * se < LOAD_MARGIN)

    # time series: daily and cumulative lift per day
    l, se, _, _ = lift_means(daily[t], daily[c])
    out["daily"] = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(l, se)]
    l, se, _, _ = lift_means(cum[t], cum[c])
    out["cumul"] = [[round(float(a), 5), round(float(b), 5)] for a, b in zip(l, se)]
    wk = []
    for w in range(4):
        sl = slice(w * 7, w * 7 + 7)
        l, se, _, _ = lift_means(daily[t][:, sl].sum(1), daily[c][:, sl].sum(1))
        wk.append(pack(l, se))
    out["weekly"] = wk
    return out


def rnd(o):
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {k: rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rnd(v) for v in o]
    return o


DIMS = {"tenure": (tenure, ["new", "tenured"]), "device": (device, ["tv", "mobile", "web"]),
        "region": (region, ["APAC", "EMEA", "NA", "LATAM"])}
results = {}
for te in ["all"] + DIMS["tenure"][1]:
    for de in ["all"] + DIMS["device"][1]:
        for re_ in ["all"] + DIMS["region"][1]:
            m = np.ones(len(variant), bool)
            for (arr, _), v in zip(DIMS.values(), [te, de, re_]):
                if v != "all":
                    m &= arr == v
            results[f"{te}|{de}|{re_}"] = rnd(analyse(m))

# one-dimension segment cuts for the forest plot
forest = []
for dim, (arr, vals) in DIMS.items():
    for v in vals:
        key = {"tenure": f"{v}|all|all", "device": f"all|{v}|all", "region": f"all|all|{v}"}[dim]
        r = results[key]
        forest.append({"dim": dim, "seg": v, "key": key, "n": r["n_t"] + r["n_c"], "cuped": r["cuped"], "unadj": r["unadj"]})

# baseline stats for the power calculator
power = {"mean": float(pre.mean()), "sd": float(pre.std(ddof=1)), "rho": float(np.corrcoef(pre, hours)[0, 1]),
         "members": len(variant)}

payload = json.dumps({"results": results, "forest": rnd(forest), "power": rnd(power)}, separators=(",", ":"))
tpl = (HERE / "templates" / "dashboard.tmpl.html").read_text(encoding="utf-8")
body = tpl.replace("__DATA__", payload)

head = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
(HERE / "dashboard.html").write_text(head + body.replace("</style>", "</style>\n</head><body>", 1) + "\n</body></html>",
                                     encoding="utf-8")
a = results["all|all|all"]
print(f"combos: {len(results)}  payload: {len(payload)/1024:.0f} KB")
print(f"ALL  cuped {a['cuped']['lift']:+.4f}  unadj {a['unadj']['lift']:+.4f}  play {a['play']['pp']:+.3f}pp  "
      f"cancel {a['cancel']['lift']:+.4f}  err {a['errors']['lift']:+.4f}  load {a['load']['diff_ms']:+.2f}ms")
print("weekly:", [round(w["lift"], 4) for w in a["weekly"]])
print("smallest cell:", min(results.items(), key=lambda kv: kv[1]["n_t"])[0], min(r["n_t"] for r in results.values()))
