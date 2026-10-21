# -*- coding: utf-8 -*-
"""
MMM dönüşümleri: Adstock, Hill doygunluğu ve kanallar arası etkileşimler.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import lfilter


def apply_geometric_adstock(spend: np.ndarray, alpha: float) -> np.ndarray:
    """
    Geometric adstock: A[t] = spend[t] + alpha * A[t-1]

    Python döngüsü yerine scipy.signal.lfilter (IIR filtre) kullanır;
    sonuç döngü versiyonuyla birebir aynıdır, ancak çok daha hızlıdır.
    """
    spend = np.asarray(spend, dtype=float)
    if spend.size == 0:
        return np.zeros(0, dtype=float)
    return lfilter([1.0], [1.0, -alpha], spend)


def steady_state_adstock(spend: np.ndarray, alpha: float) -> np.ndarray:
    """Sabit harcamada adstock'un ulaştığı denge değeri: A = x / (1 - alpha)."""
    return np.asarray(spend, dtype=float) / (1.0 - min(alpha, 0.99))


def apply_hill_saturation(x: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    """
    Hill fonksiyonu ile azalan verimler (doygunluk): x^S / (K^S + x^S).
    Çıktı 0 ile 1 arasındadır.
    """
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    x_s = np.power(x, slope)
    half_sat_s = np.power(half_sat, slope)
    return x_s / (half_sat_s + x_s + 1e-9)


def apply_cross_channel_interactions(
    spends: Dict[str, np.ndarray],
    adstocked_spends: Dict[str, np.ndarray],
    spillover_matrix: Dict[Tuple[str, str], float],
    cannibalization_matrix: Dict[Tuple[str, str], float],
    reference_means: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Sıçrama (Spillover) ve Yamyamlık (Cannibalization) etkilerini uygular.

    Args:
        spends: Ham harcama sözlüğü {kanal: dizi}.
        adstocked_spends: Adstock uygulanmış harcama sözlüğü.
        spillover_matrix: {(kaynak, hedef): gamma}.
        cannibalization_matrix: {(rakip, hedef): eta}.
        reference_means: Yamyamlık cezasında bölen olarak kullanılacak kanal
            ortalamaları. Verilmezse her kanalın kendi serisinin ortalaması
            kullanılır (eğitim sırasındaki davranış). Optimizasyonda tek
            noktalık dizilerle çalışırken geçmiş ortalamaları buradan verin.
    """
    channels = list(spends.keys())
    adjusted_spends = {ch: np.array(adstocked_spends[ch], dtype=float) for ch in channels}

    # 1. Spillover (üst huniden alt huniye hacim sıçraması)
    for (source_ch, target_ch), gamma in spillover_matrix.items():
        if source_ch in adstocked_spends and target_ch in adjusted_spends:
            adjusted_spends[target_ch] = adjusted_spends[target_ch] + gamma * np.asarray(
                adstocked_spends[source_ch], dtype=float
            )

    # 2. Cannibalization (performans kanallarında çakışma ve marjinal baskılanma)
    final_effective_spends = {}
    for ch in channels:
        penalty = np.zeros_like(np.asarray(spends[ch], dtype=float))
        for (comp_ch, target_ch), eta in cannibalization_matrix.items():
            if target_ch == ch and comp_ch in spends:
                if reference_means is not None and comp_ch in reference_means:
                    mean_spend = float(reference_means[comp_ch]) + 1e-9
                else:
                    mean_spend = float(np.mean(spends[comp_ch])) + 1e-9
                penalty = penalty + eta * (np.asarray(spends[comp_ch], dtype=float) / mean_spend)

        retention_factor = np.clip(1.0 - penalty, 0.1, 1.0)
        final_effective_spends[ch] = adjusted_spends[ch] * retention_factor

    return final_effective_spends
