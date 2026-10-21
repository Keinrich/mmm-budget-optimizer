# -*- coding: utf-8 -*-
"""
Spillover ve Cannibalization içeren sentetik MMM veri seti üreticisi.
Gerçek parametreler modül düzeyinde tutulur; böylece evaluator, tahmin
edilen parametreleri gerçek değerlerle karşılaştırabilir.
"""

from copy import deepcopy
from typing import Dict

import numpy as np
import pandas as pd

from transformations import (
    apply_geometric_adstock,
    apply_hill_saturation,
    apply_cross_channel_interactions,
)

# Kanal parametreleri (Adstock alpha, Hill K, Hill S, Max beta)
TRUE_CHANNEL_PARAMS = {
    "tv":            {"alpha": 0.70, "half_sat": 50.0, "slope": 1.0, "beta": 80.0},
    "influencer":    {"alpha": 0.40, "half_sat": 12.0, "slope": 1.3, "beta": 45.0},
    "meta_ads":      {"alpha": 0.35, "half_sat": 20.0, "slope": 1.1, "beta": 50.0},
    "google_search": {"alpha": 0.15, "half_sat": 25.0, "slope": 1.2, "beta": 70.0},
}
TRUE_SPILLOVER = {
    ("tv", "google_search"): 0.35,
    ("influencer", "google_search"): 0.25,
    ("tv", "meta_ads"): 0.15,
}
TRUE_CANNIBALIZATION = {("meta_ads", "google_search"): 0.12}
TRUE_BASELINE = {"intercept": 120.0, "trend": 0.08, "seasonality": 20.0}
NOISE_SD = 5.0


def get_true_params() -> Dict:
    """Evaluator._unpack_params ile aynı yapıda gerçek parametre sözlüğü."""
    return {
        "channels": deepcopy(TRUE_CHANNEL_PARAMS),
        "spillover": dict(TRUE_SPILLOVER),
        "cannibalization": dict(TRUE_CANNIBALIZATION),
        "baseline": dict(TRUE_BASELINE),
    }


def generate_spillover_mmm_data(n_weeks: int = 156, seed: int = 42) -> pd.DataFrame:
    """156 haftalık (3 yıllık) sentetik veri seti üretir."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    b = TRUE_BASELINE
    baseline = b["intercept"] + b["trend"] * t + b["seasonality"] * np.sin(2 * np.pi * t / 52.14)

    # Ham harcamalar (haftalık Bin TL)
    tv = np.where(rng.random(n_weeks) > 0.7, rng.uniform(30, 70, n_weeks), 0.0)
    influencer = rng.exponential(scale=6.0, size=n_weeks)
    meta_ads = rng.gamma(shape=5.0, scale=3.0, size=n_weeks)
    google_search = rng.gamma(shape=8.0, scale=2.0, size=n_weeks)

    raw_spends = {
        "tv": tv,
        "influencer": influencer,
        "meta_ads": meta_ads,
        "google_search": google_search,
    }

    adstocked = {
        ch: apply_geometric_adstock(raw_spends[ch], TRUE_CHANNEL_PARAMS[ch]["alpha"])
        for ch in raw_spends
    }
    effective_spends = apply_cross_channel_interactions(
        raw_spends, adstocked, TRUE_SPILLOVER, TRUE_CANNIBALIZATION
    )

    total_marketing_impact = np.zeros(n_weeks)
    contributions = {}
    for ch, p in TRUE_CHANNEL_PARAMS.items():
        sat = apply_hill_saturation(effective_spends[ch], p["half_sat"], p["slope"])
        contrib = sat * p["beta"]
        contributions[ch] = contrib
        total_marketing_impact += contrib

    noise = rng.normal(0, NOISE_SD, n_weeks)
    total_sales = baseline + total_marketing_impact + noise

    df = pd.DataFrame(
        {"date": dates, "total_sales": np.round(total_sales, 2), "baseline": np.round(baseline, 2)}
    )
    for ch in raw_spends:
        df[f"spend_{ch}"] = np.round(raw_spends[ch], 2)
        df[f"contrib_{ch}"] = np.round(contributions[ch], 2)
    return df


if __name__ == "__main__":
    df = generate_spillover_mmm_data()
    print("--- Veri Seti Başarıyla Üretildi ---")
    print(df[["date", "total_sales", "spend_tv", "spend_google_search", "contrib_google_search"]].head())
