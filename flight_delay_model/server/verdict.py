# -*- coding: utf-8 -*-
"""
Verdict: turn the scenario probability distribution into one plain-English
summary phrase. Rule confirmed by the user (2026-10-06):

  The phrase always follows the single highest-probability scenario —
  pure argmax, no additional tier rules.
    - "On time" most likely      -> "Expected on-time departure" / "... arrival"
    - e.g. "1-15 min delay" wins -> "Expected 1-15 min departure delay" / "... arrival delay"

`level` only drives the badge colour and follows the severity of the winning
bucket: 0 -> good, 1 -> notice, 2 -> warn, 3/4 -> alert.
"""

# English display labels for the delay buckets (index-aligned with config.BUCKET_LABELS)
BUCKET_LABELS_EN = [
    "On time",
    "1-15 min delay",
    "16-30 min delay",
    "31-60 min delay",
    "Over 60 min delay",
]

# Duration wording used inside the verdict phrase ("" for the on-time bucket)
_DURATION_EN = ["", "1-15 min", "16-30 min", "31-60 min", "over 60 min"]

_BUCKET_LEVEL = ["good", "notice", "warn", "alert", "alert"]


def verdict(seg: str, scenarios: list, expected_minutes: float = None) -> dict:
    """seg: 'dep' | 'arr'; scenarios: [{bucket, label, probability}, ...]
    (the bucket-model scenario list, any order). Pure argmax over probability;
    ties resolve to the less severe (smaller) bucket."""
    seg_word = "departure" if seg == "dep" else "arrival"
    best = max(scenarios, key=lambda s: (s["probability"], -s["bucket"]))
    k = int(best["bucket"])
    if k == 0:
        phrase = f"Expected on-time {seg_word}"
    else:
        phrase = f"Expected {_DURATION_EN[k]} {seg_word} delay"

    p_no_delay = next((s["probability"] for s in scenarios if s["bucket"] == 0), None)
    out = {
        "level": _BUCKET_LEVEL[k],
        "bucket": k,
        "phrase": phrase,
        "p_delay": round(1.0 - p_no_delay, 4) if p_no_delay is not None else None,
    }
    if expected_minutes is not None:
        out["expected_delay_min"] = round(float(expected_minutes), 1)
    return out


def verdict_from_prob(seg: str, p_delay: float, expected_minutes: float = None) -> dict:
    """Fallback when only the binary classifier is available (no bucket scenarios).
    Same argmax idea with the two plain outcomes."""
    seg_word = "departure" if seg == "dep" else "arrival"
    if p_delay <= 0.5:
        out = {"level": "good", "bucket": 0,
               "phrase": f"Expected on-time {seg_word}",
               "p_delay": round(float(p_delay), 4)}
    else:
        out = {"level": "warn", "bucket": None,
               "phrase": f"Expected {seg_word} delay",
               "p_delay": round(float(p_delay), 4)}
    if expected_minutes is not None:
        out["expected_delay_min"] = round(float(expected_minutes), 1)
    return out
