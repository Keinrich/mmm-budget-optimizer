# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:42:59 2026

@author: mcany
"""

import numpy as np
import pandas as pd
from transformations import (
    apply_geometric_adstock,
    apply_hill_saturation,
    apply_cross_channel_interactions,
)


def generate_spillover_mmm_data(n_weeks: int = 156, seed: int = 42) -> pd.DataFrame:
    """
    Spillover (sıçrama) ve Cannibalization (yamyamlık) dinamiklerini içeren 
    156 haftalık (3 yıllık) sentetik veri seti üretir.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)
    
    # Organik baz satış ve yıllık mevsimsellik
    baseline = 120 + 0.08 * t + 20 * np.sin(2 * np.pi * t / 52.14)
    
    # Ham Harcamalar (Haftalık Bin TL bazında)
    tv = np.where(np.random.rand(n_weeks) > 0.7, np.random.uniform(30, 70, n_weeks), 0.0)
    influencer = np.random.exponential(scale=6.0, size=n_weeks)
    meta_ads = np.random.gamma(shape=5.0, scale=3.0, size=n_weeks)
    google_search = np.random.gamma(shape=8.0, scale=2.0, size=n_weeks)
    
    raw_spends = {
        "tv": tv,
        "influencer": influencer,
        "meta_ads": meta_ads,
        "google_search": google_search
    }
    
    # Kanal Parametreleri (Adstock Alpha, Hill K, Hill S, Max Beta)
    params = {
        "tv":            {"alpha": 0.70, "half_sat": 50.0, "slope": 1.0, "beta": 80.0},
        "influencer":    {"alpha": 0.40, "half_sat": 12.0, "slope": 1.3, "beta": 45.0},
        "meta_ads":      {"alpha": 0.35, "half_sat": 20.0, "slope": 1.1, "beta": 50.0},
        "google_search": {"alpha": 0.15, "half_sat": 25.0, "slope": 1.2, "beta": 70.0},
    }
    
    # Etkileşim Matrisleri
    spillover_matrix = {
        ("tv", "google_search"): 0.35,
        ("influencer", "google_search"): 0.25,
        ("tv", "meta_ads"): 0.15
    }
    
    cannibalization_matrix = {
        ("meta_ads", "google_search"): 0.12
    }
    
    # 1. Adstock Uygula
    adstocked = {ch: apply_geometric_adstock(raw_spends[ch], params[ch]["alpha"]) for ch in raw_spends}
    
    # 2. Çapraz Etkileşimleri İşle
    effective_spends = apply_cross_channel_interactions(
        raw_spends, adstocked, spillover_matrix, cannibalization_matrix
    )
    
    # 3. Satış Katkılarını Hesapla
    total_marketing_impact = np.zeros(n_weeks)
    contributions = {}
    
    for ch, p in params.items():
        sat = apply_hill_saturation(effective_spends[ch], p["half_sat"], p["slope"])
        contrib = sat * p["beta"]
        contributions[f"{ch}_contrib"] = contrib
        total_marketing_impact += contrib
        
    noise = np.random.normal(0, 5.0, n_weeks)
    total_sales = baseline + total_marketing_impact + noise
    
    # DataFrame Yapısı
    df = pd.DataFrame({"date": dates, "total_sales": np.round(total_sales, 2), "baseline": np.round(baseline, 2)})
    for ch in raw_spends:
        df[f"spend_{ch}"] = np.round(raw_spends[ch], 2)
        df[f"contrib_{ch}"] = np.round(contributions[f"{ch}_contrib"], 2)
        
    return df


if __name__ == "__main__":
    df = generate_spillover_mmm_data()
    print("--- Veri Seti Başarıyla Üretildi ---")
    print(df[["date", "total_sales", "spend_tv", "spend_google_search", "contrib_google_search"]].head())