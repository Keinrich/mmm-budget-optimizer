# -*- coding: utf-8 -*-
"""
Kararlı durum (steady-state) bütçe optimizasyonu.
Eğitimde kullanılan dönüşümlerin AYNISI kullanılır (kopya formül yok).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from transformations import (
    apply_cross_channel_interactions,
    apply_hill_saturation,
    steady_state_adstock,
)


def check_feasibility(
    total_budget: float,
    channel_bounds: Dict[str, Tuple[float, float]],
    channels: List[str],
) -> Optional[str]:
    """Kısıtlar bir çözüme izin vermiyorsa Türkçe hata mesajı, aksi halde None döner."""
    lo = [min(channel_bounds.get(ch, (0.0, total_budget))[0], total_budget) for ch in channels]
    hi = [min(channel_bounds.get(ch, (0.0, total_budget))[1], total_budget) for ch in channels]
    for ch, l, h in zip(channels, lo, hi):
        if l > h + 1e-9:
            return f"{ch.upper()} için alt sınır ({l:.1f}) üst sınırdan ({h:.1f}) büyük."
    if sum(lo) > total_budget + 1e-9:
        return f"Alt sınırların toplamı ({sum(lo):.1f}) toplam bütçeyi ({total_budget:.1f}) aşıyor."
    if sum(hi) < total_budget - 1e-9:
        return f"Üst sınırların toplamı ({sum(hi):.1f}) toplam bütçenin ({total_budget:.1f}) altında; bütçe dağıtılamaz."
    return None


class SpilloverBudgetOptimizer:
    def __init__(self, fitted_params: Dict, historical_means: Dict[str, float]):
        """
        Args:
            fitted_params: Evaluator.fit() çıktısı.
            historical_means: Cannibalization cezasında referans alınan geçmiş
                haftalık harcama ortalamaları.
        """
        self.params = fitted_params
        self.channels = list(fitted_params["channels"].keys())
        self.historical_means = historical_means

    # ------------------------------------------------------------- model
    def _steady_state_sales(
        self,
        spends_array: np.ndarray,
        spillover: Optional[Dict] = None,
        cannibalization: Optional[Dict] = None,
    ) -> float:
        """
        Sabit haftalık harcamanın kararlı durumda getireceği PAZARLAMA KAYNAKLI
        satış (baseline hariç). spillover/cannibalization verilirse model
        değerlerinin yerine geçer (ör. {} vererek etkiyi kapatmak için).
        """
        spillover = self.params.get("spillover", {}) if spillover is None else spillover
        cannibalization = (
            self.params.get("cannibalization", {}) if cannibalization is None else cannibalization
        )

        x = np.maximum(np.asarray(spends_array, dtype=float), 0.0)
        spends = {ch: np.array([v]) for ch, v in zip(self.channels, x)}
        adstocked = {
            ch: steady_state_adstock(spends[ch], self.params["channels"][ch]["alpha"])
            for ch in self.channels
        }
        effective = apply_cross_channel_interactions(
            spends, adstocked, spillover, cannibalization, reference_means=self.historical_means
        )

        total = 0.0
        for ch in self.channels:
            p = self.params["channels"][ch]
            sat = apply_hill_saturation(effective[ch], p["half_sat"], p["slope"])[0]
            total += float(sat) * p["beta"]
        return total

    def evaluate(self, allocation: Dict[str, float]) -> float:
        """Sözlük biçimindeki bir dağılımın kararlı durum satışını döndürür."""
        return self._steady_state_sales(np.array([allocation[ch] for ch in self.channels]))

    def _to_array(self, allocation: Dict[str, float]) -> np.ndarray:
        return np.array([allocation[ch] for ch in self.channels], dtype=float)

    # --------------------------------------------------------- tanılama
    def marginal_returns(self, allocation: Dict[str, float], eps: float = 1e-3) -> Dict[str, float]:
        """Kanal başına marjinal getiri (mROAS): +1 Bin TL harcamanın yarattığı ek satış."""
        x = self._to_array(allocation)
        out = {}
        for i, ch in enumerate(self.channels):
            up, dn = x.copy(), x.copy()
            up[i] += eps
            dn[i] = max(dn[i] - eps, 0.0)
            out[ch] = (self._steady_state_sales(up) - self._steady_state_sales(dn)) / (up[i] - dn[i])
        return out

    def channel_response_curve(
        self, channel: str, grid: np.ndarray, allocation: Dict[str, float]
    ) -> np.ndarray:
        """
        Diğer kanallar `allocation`'daki değerde sabitken, `channel` harcaması
        değiştikçe oluşan ARTIMLI toplam satış (spillover ve yamyamlık dahil).
        Optimizer'ın kullandığı fonksiyonun birebir aynısıdır.
        """
        base = self._to_array(allocation)
        i = self.channels.index(channel)
        zero = base.copy()
        zero[i] = 0.0
        floor = self._steady_state_sales(zero)
        curve = []
        for g in grid:
            x = base.copy()
            x[i] = g
            curve.append(self._steady_state_sales(x) - floor)
        return np.array(curve)

    def spillover_contribution(self, allocation: Dict[str, float]) -> Tuple[float, float]:
        """(spillover'ın eklediği satış, spillover'lı toplam satış) çifti."""
        x = self._to_array(allocation)
        with_sp = self._steady_state_sales(x)
        without = self._steady_state_sales(x, spillover={})
        return with_sp - without, with_sp

    # ------------------------------------------------------ optimizasyon
    @staticmethod
    def _proportional_start(lo: np.ndarray, hi: np.ndarray, budget: float) -> np.ndarray:
        room = hi - lo
        frac = 0.0 if room.sum() <= 0 else (budget - lo.sum()) / room.sum()
        return lo + room * np.clip(frac, 0.0, 1.0)

    @staticmethod
    def _random_start(rng, lo: np.ndarray, hi: np.ndarray, budget: float) -> np.ndarray:
        """Sınırlara uyan, toplamı bütçeye eşit rastgele başlangıç noktası."""
        x = lo.copy()
        remaining = budget - x.sum()
        free = hi - x
        for _ in range(30):
            if remaining <= 1e-12:
                break
            active = free > 1e-12
            if not active.any():
                break
            w = rng.random(int(active.sum()))
            w = w / w.sum()
            add = np.zeros_like(x)
            add[active] = np.minimum(w * remaining, free[active])
            x += add
            free -= add
            remaining -= add.sum()
        return x

    def optimize_budget(
        self,
        total_budget: float,
        channel_bounds: Dict[str, Tuple[float, float]],
        n_starts: int = 6,
        seed: int = 0,
    ) -> Dict:
        """
        Toplam bütçeyi kanal sınırlarına uyarak satışı maksimize edecek şekilde
        dağıtır. Hill eğimi S>1 olan kanallarda getiri fonksiyonu konveks
        bölgeler içerebileceği için SLSQP çoklu başlangıçla çalıştırılır.
        """
        msg = check_feasibility(total_budget, channel_bounds, self.channels)
        if msg:
            raise ValueError(msg)

        bounds = [
            (
                min(channel_bounds.get(ch, (0.0, total_budget))[0], total_budget),
                min(channel_bounds.get(ch, (0.0, total_budget))[1], total_budget),
            )
            for ch in self.channels
        ]
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        rng = np.random.default_rng(seed)
        starts = [self._proportional_start(lo, hi, total_budget)]
        starts += [self._random_start(rng, lo, hi, total_budget) for _ in range(max(n_starts - 1, 0))]

        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - total_budget},)
        results = []
        for s in starts:
            res = minimize(
                lambda x: -self._steady_state_sales(x),
                s, method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            # SLSQP'nin küçük sınır/toplam ihlallerini temizle
            if res.success:
                results.append(res)

        success = len(results) > 0
        if not success:
            # Hiçbiri yakınsamadıysa en azından uygun başlangıç noktasını döndür
            x_best = starts[0]
            spread = float("nan")
        else:
            best = min(results, key=lambda r: r.fun)
            x_best = np.clip(best.x, lo, hi)
            spread = float(max(r.fun for r in results) - best.fun)

        return {
            "optimized_allocation": {ch: round(float(v), 2) for ch, v in zip(self.channels, x_best)},
            "expected_steady_state_sales": round(float(self._steady_state_sales(x_best)), 2),
            "success": success,
            "n_converged": len(results),
            "n_starts": len(starts),
            "multistart_spread": spread,
            "bounds": dict(zip(self.channels, bounds)),
        }


if __name__ == "__main__":
    from data_generator import generate_spillover_mmm_data
    from evaluator import SpilloverMMMEvaluator

    df_data = generate_spillover_mmm_data()
    channels_list = ["tv", "influencer", "meta_ads", "google_search"]
    evaluator = SpilloverMMMEvaluator(
        df=df_data,
        channels=channels_list,
        spillover_edges=[("tv", "google_search"), ("influencer", "google_search"), ("tv", "meta_ads")],
        cannibalization_edges=[("meta_ads", "google_search")],
    )
    fitted = evaluator.fit()
    hist_means = {ch: float(df_data[f"spend_{ch}"].mean()) for ch in channels_list}

    optimizer = SpilloverBudgetOptimizer(fitted, hist_means)
    budget = 100.0
    bounds_ = {"tv": (10.0, 50.0), "influencer": (5.0, 25.0), "meta_ads": (5.0, 30.0), "google_search": (5.0, 40.0)}
    res = optimizer.optimize_budget(budget, bounds_)

    print("\n--- Bütçe Optimizasyon Sonucu ---")
    for ch, v in res["optimized_allocation"].items():
        print(f"  * {ch}: {v} Bin TL")
    print(f"Beklenen kararlı durum (pazarlama kaynaklı) satış: {res['expected_steady_state_sales']}")
    print("mROAS:", {k: round(v, 3) for k, v in optimizer.marginal_returns(res['optimized_allocation']).items()})
