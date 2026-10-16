# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:38:34 2026

@author: mcany
"""

import numpy as np
from typing import Dict, Tuple


def apply_geometric_adstock(spend: np.ndarray, alpha: float) -> np.ndarray:
    """
    Geometric adstock (gecikmeli taşıma etkisi) hesaplar.
    
    Args:
        spend (np.ndarray): Kanala ait zaman serisi harcama dizisi.
        alpha (float): Sönümlenme parametresi (0 <= alpha < 1).
        
    Returns:
        np.ndarray: Adstock uygulanmış efektif harcama dizisi.
    """
    adstocked = np.zeros_like(spend, dtype=float)
    if len(spend) == 0:
        return adstocked
    
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + alpha * adstocked[t - 1]
    return adstocked


def apply_hill_saturation(x: np.ndarray, half_sat: float, slope: float) -> np.ndarray:
    """
    Hill fonksiyonu ile azalan verimler (doygunluk) etkisini hesaplar.
    
    Args:
        x (np.ndarray): Adstock veya efektif harcama dizisi.
        half_sat (float): Yarı-doyum noktası (K > 0).
        slope (float): Eğim / şekil parametresi (S > 0).
        
    Returns:
        np.ndarray: Doyum katsayısı (0 ile 1 arası oransal etki).
    """
    x_s = np.power(x, slope)
    half_sat_s = np.power(half_sat, slope)
    return x_s / (half_sat_s + x_s + 1e-9)


def apply_cross_channel_interactions(
    spends: Dict[str, np.ndarray],
    adstocked_spends: Dict[str, np.ndarray],
    spillover_matrix: Dict[Tuple[str, str], float],
    cannibalization_matrix: Dict[Tuple[str, str], float]
) -> Dict[str, np.ndarray]:
    """
    Kanallar arasındaki Sıçrama (Spillover) ve Yamyamlık (Cannibalization)
    etkilerini hesaplayarak nihai efektif girdi dizilerini döndürür.
    
    Args:
        spends: Ham harcama sözlüğü {kanal_adı: np.ndarray}.
        adstocked_spends: Adstock uygulanmış harcama sözlüğü.
        spillover_matrix: {(kaynak_kanal, hedef_kanal): gamma_katsayısı}.
        cannibalization_matrix: {(rakip_kanal, hedef_kanal): eta_katsayısı}.
        
    Returns:
        Dict[str, np.ndarray]: Çapraz etkileşimleri işlenmiş efektif harcamalar.
    """
    channels = list(spends.keys())
    adjusted_spends = {ch: adstocked_spends[ch].copy() for ch in channels}
    
    # 1. Spillover (Üst huniden alt huniye hacim sıçraması)
    for (source_ch, target_ch), gamma in spillover_matrix.items():
        if source_ch in adstocked_spends and target_ch in adjusted_spends:
            adjusted_spends[target_ch] += gamma * adstocked_spends[source_ch]
            
    # 2. Cannibalization (Performans kanallarındaki çakışma ve marjinal baskılanma)
    final_effective_spends = {}
    for ch in channels:
        penalty = np.zeros_like(spends[ch], dtype=float)
        for (comp_ch, target_ch), eta in cannibalization_matrix.items():
            if target_ch == ch and comp_ch in spends:
                mean_spend = np.mean(spends[comp_ch]) + 1e-9
                penalty += eta * (spends[comp_ch] / mean_spend)
        
        # Baskılama çarpanını [0.1, 1.0] aralığında tutuyoruz
        retention_factor = np.clip(1.0 - penalty, 0.1, 1.0)
        final_effective_spends[ch] = adjusted_spends[ch] * retention_factor
        
    return final_effective_spends