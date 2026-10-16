# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:49:51 2026

@author: mcany
"""

import sys
import os

# Bulunduğumuz klasörü Python arama yoluna ekliyoruz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple

from transformations import (
    apply_geometric_adstock,
    apply_hill_saturation,
    apply_cross_channel_interactions,
)
from data_generator import generate_spillover_mmm_data
from evaluator import SpilloverMMMEvaluator


class SpilloverBudgetOptimizer:
    def __init__(self, fitted_params: Dict, historical_means: Dict[str, float]):
        """
        Args:
            fitted_params: Evaluator modülünden elde edilen eğitilmiş katsayılar.
            historical_means: Cannibalization cezasında referans alınacak geçmiş harcama ortalamaları.
        """
        self.params = fitted_params
        self.channels = list(fitted_params["channels"].keys())
        self.historical_means = historical_means

    def _steady_state_sales(self, spends_array: np.ndarray) -> float:
        """
        Verilen bütçe dağılımının kararlı durumda (steady-state) getireceği toplam satışı hesaplar.
        """
        x = {ch: spend for ch, spend in zip(self.channels, spends_array)}

        # 1. Steady-State Adstock Birikimi: A = x / (1 - alpha)
        adstocked = {}
        for ch in self.channels:
            alpha = self.params["channels"][ch]["alpha"]
            adstocked[ch] = x[ch] / (1.0 - min(alpha, 0.99))

        # 2. Spillover Etkisi
        adjusted = adstocked.copy()
        for (source, target), gamma in self.params.get("spillover", {}).items():
            if source in adstocked and target in adjusted:
                adjusted[target] += gamma * adstocked[source]

        # 3. Cannibalization Etkisi
        final_effective = {}
        for ch in self.channels:
            penalty = 0.0
            for (comp_ch, target_ch), eta in self.params.get("cannibalization", {}).items():
                if target_ch == ch and comp_ch in x:
                    mean_val = self.historical_means.get(comp_ch, 1e-9)
                    penalty += eta * (x[comp_ch] / mean_val)

            retention = np.clip(1.0 - penalty, 0.1, 1.0)
            final_effective[ch] = adjusted[ch] * retention

        # 4. Hill Saturation & Sales Contribution
        total_sales = 0.0
        for ch in self.channels:
            K = self.params["channels"][ch]["half_sat"]
            S = self.params["channels"][ch]["slope"]
            beta = self.params["channels"][ch]["beta"]

            eff_spend = final_effective[ch]
            sat = (eff_spend ** S) / ((K ** S) + (eff_spend ** S) + 1e-9)
            total_sales += sat * beta

        return total_sales

    def _objective(self, spends_array: np.ndarray) -> float:
        """Minimize fonksiyonu için satışı negatife çevirir."""
        return -1.0 * self._steady_state_sales(spends_array)

    def optimize_budget(self, total_budget: float, channel_bounds: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Verilen toplam bütçeyi kanal kısıtlarına uyarak satış getirisini maksimize edecek şekilde dağıtır.
        """
        n_channels = len(self.channels)
        initial_guess = np.array([total_budget / n_channels] * n_channels)

        bounds = [channel_bounds.get(ch, (0.0, total_budget)) for ch in self.channels]
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget})

        result = minimize(
            self._objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )

        optimized_spends = {ch: round(float(spend), 2) for ch, spend in zip(self.channels, result.x)}
        max_sales = round(float(self._steady_state_sales(result.x)), 2)

        return {
            "optimized_allocation": optimized_spends,
            "expected_steady_state_sales": max_sales,
            "success": result.success
        }


if __name__ == "__main__":
    # 1. Veriyi üret ve modeli eğit
    df_data = generate_spillover_mmm_data()
    channels_list = ["tv", "influencer", "meta_ads", "google_search"]
    spillover_list = [("tv", "google_search"), ("influencer", "google_search"), ("tv", "meta_ads")]
    cannibalization_list = [("meta_ads", "google_search")]

    evaluator = SpilloverMMMEvaluator(
        df=df_data,
        channels=channels_list,
        spillover_edges=spillover_list,
        cannibalization_edges=cannibalization_list
    )
    fitted_results = evaluator.fit()

    # 2. Geçmiş harcama ortalamalarını hesapla
    hist_means = {ch: float(df_data[f"spend_{ch}"].mean()) for ch in channels_list}

    # 3. Bütçe Optimizasyonunu Çalıştır
    optimizer = SpilloverBudgetOptimizer(fitted_results, hist_means)

    # Senaryo: Toplam 100 Bin TL haftalık bütçe dağıtımı
    target_budget = 100.0
    budget_bounds = {
        "tv": (10.0, 50.0),
        "influencer": (5.0, 25.0),
        "meta_ads": (5.0, 30.0),
        "google_search": (5.0, 40.0)
    }

    opt_result = optimizer.optimize_budget(total_budget=target_budget, channel_bounds=budget_bounds)

    print("\n--- Bütçe Optimizasyon Sonucu (SLSQP) ---")
    print(f"Toplam Bütçe: {target_budget} Bin TL")
    print("Optimal Bütçe Dağılımı:")
    for ch, spend in opt_result["optimized_allocation"].items():
        print(f"  * {ch}: {spend} Bin TL")
    print(f"Beklenen Kararlı Durum Satışı: {opt_result['expected_steady_state_sales']}")