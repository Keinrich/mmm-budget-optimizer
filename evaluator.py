# -*- coding: utf-8 -*-
"""
Spillover + Cannibalization farkındalıklı MMM'in parametre tahmini (fit),
tanılama ve gerçek-vs-tahmin parametre karşılaştırması.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from transformations import (
    apply_cross_channel_interactions,
    apply_geometric_adstock,
    apply_hill_saturation,
)
from data_generator import generate_spillover_mmm_data, get_true_params

CHANNEL_KEYS = ("alpha", "half_sat", "slope", "beta")
CHANNEL_INIT = (0.3, 20.0, 1.0, 50.0)
CHANNEL_BOUNDS = ((0.0, 0.9), (1.0, 100.0), (0.1, 3.0), (0.0, 200.0))
BASELINE_KEYS = ("intercept", "trend", "seasonality")
BASELINE_INIT = (100.0, 0.0, 10.0)
BASELINE_BOUNDS = ((0.0, 300.0), (-2.0, 2.0), (-30.0, 30.0))


def flatten_params(p: Dict) -> Dict[str, float]:
    """İç içe parametre sözlüğünü {ad: değer} olarak düzleştirir (_param_layout ile aynı sıra)."""
    flat = {}
    for ch, d in p["channels"].items():
        for k in CHANNEL_KEYS:
            flat[f"{ch}.{k}"] = float(d[k])
    for (a, b), v in p["spillover"].items():
        flat[f"gamma.{a}->{b}"] = float(v)
    for (a, b), v in p["cannibalization"].items():
        flat[f"eta.{a}->{b}"] = float(v)
    for k in BASELINE_KEYS:
        flat[f"baseline.{k}"] = float(p["baseline"][k])
    return flat


class SpilloverMMMEvaluator:
    def __init__(
        self,
        df: pd.DataFrame,
        channels: List[str],
        spillover_edges: List[Tuple[str, str]],
        cannibalization_edges: List[Tuple[str, str]],
    ):
        self.df = df
        self.channels = channels
        self.spillover_edges = spillover_edges
        self.cannibalization_edges = cannibalization_edges

        self.y_true = df["total_sales"].values
        self.spends = {ch: df[f"spend_{ch}"].values for ch in channels}

        self.t = np.arange(len(df))
        self.seasonality = np.sin(2 * np.pi * self.t / 52.14)

        self.fitted_params: Dict = {}
        self.fit_report: Dict = {}
        self._names: List[str] = []
        self._bounds: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------ yapı
    def _param_layout(self) -> Tuple[List[str], List[float], List[Tuple[float, float]]]:
        """Parametre adları, başlangıç değerleri ve sınırları TEK yerde tanımlanır."""
        names, x0, bounds = [], [], []
        for ch in self.channels:
            for k, g, b in zip(CHANNEL_KEYS, CHANNEL_INIT, CHANNEL_BOUNDS):
                names.append(f"{ch}.{k}"); x0.append(g); bounds.append(b)
        for a, b_ in self.spillover_edges:
            names.append(f"gamma.{a}->{b_}"); x0.append(0.1); bounds.append((0.0, 1.0))
        for a, b_ in self.cannibalization_edges:
            names.append(f"eta.{a}->{b_}"); x0.append(0.05); bounds.append((0.0, 0.5))
        for k, g, b in zip(BASELINE_KEYS, BASELINE_INIT, BASELINE_BOUNDS):
            names.append(f"baseline.{k}"); x0.append(g); bounds.append(b)
        return names, x0, bounds

    def _unpack_params(self, params: np.ndarray) -> Dict:
        """1D parametre dizisini okunabilir sözlük yapısına dönüştürür."""
        idx = 0
        unpacked = {"channels": {}, "spillover": {}, "cannibalization": {}, "baseline": {}}

        for ch in self.channels:
            unpacked["channels"][ch] = {k: params[idx + i] for i, k in enumerate(CHANNEL_KEYS)}
            idx += len(CHANNEL_KEYS)
        for edge in self.spillover_edges:
            unpacked["spillover"][edge] = params[idx]
            idx += 1
        for edge in self.cannibalization_edges:
            unpacked["cannibalization"][edge] = params[idx]
            idx += 1
        unpacked["baseline"] = {k: params[idx + i] for i, k in enumerate(BASELINE_KEYS)}
        return unpacked

    # ---------------------------------------------------------------- model
    def _predict_sales(self, p: Dict) -> np.ndarray:
        adstocked = {
            ch: apply_geometric_adstock(self.spends[ch], p["channels"][ch]["alpha"])
            for ch in self.channels
        }
        effective_spends = apply_cross_channel_interactions(
            self.spends, adstocked, p["spillover"], p["cannibalization"]
        )

        y_pred = (
            p["baseline"]["intercept"]
            + p["baseline"]["trend"] * self.t
            + p["baseline"]["seasonality"] * self.seasonality
        )
        for ch in self.channels:
            sat = apply_hill_saturation(
                effective_spends[ch], p["channels"][ch]["half_sat"], p["channels"][ch]["slope"]
            )
            y_pred = y_pred + sat * p["channels"][ch]["beta"]
        return y_pred

    def _objective_function(self, params: np.ndarray) -> float:
        """RMSE kayıp fonksiyonu."""
        y_pred = self._predict_sales(self._unpack_params(params))
        return float(np.sqrt(np.mean((self.y_true - y_pred) ** 2)))

    # ------------------------------------------------------------------ fit
    def fit(self, n_starts: int = 8, seed: int = 0, verbose: bool = True) -> Dict:
        """
        Çoklu başlangıç (multi-start) L-BFGS-B. Problem konveks olmadığı için
        ilk başlangıç noktası sabit, kalanlar sınırlar içinde rastgele seçilir;
        en düşük RMSE'li çözüm tutulur.
        """
        names, x0, bounds = self._param_layout()
        self._names, self._bounds = names, bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        rng = np.random.default_rng(seed)
        starts = [np.array(x0, dtype=float)]
        starts += [rng.uniform(lo, hi) for _ in range(max(n_starts - 1, 0))]

        if verbose:
            print(f"Model eğitimi başladı (L-BFGS-B, {len(starts)} başlangıç noktası)...")

        results = []
        for s in starts:
            res = minimize(
                self._objective_function, s, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            results.append(res)

        best_idx = int(np.argmin([r.fun for r in results]))
        best = results[best_idx]
        self.fitted_params = self._unpack_params(best.x)

        y_pred = self._predict_sales(self.fitted_params)
        self.fit_report = {
            "r2": float(r2_score(self.y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(self.y_true, y_pred)),
            "rmse": float(best.fun),
            "n_starts": len(starts),
            "best_start": best_idx,
            "converged": bool(best.success),
            "rmse_per_start": [float(r.fun) for r in results],
        }

        if verbose:
            r = self.fit_report
            print(f"R² (eğitim, in-sample): {r['r2']:.4f} | MAPE: %{r['mape'] * 100:.2f} | RMSE: {r['rmse']:.3f}")
            print(f"Başlangıçlar arası RMSE aralığı: {min(r['rmse_per_start']):.3f} – {max(r['rmse_per_start']):.3f}")
        return self.fitted_params

    # ------------------------------------------------------ gerçek vs tahmin
    def parameter_recovery_table(self, true_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Tahmin edilen parametreleri gerçek (veriyi üreten) değerlerle karşılaştırır.
        Sentetik verinin asıl avantajı: modelin parametreleri GERİ KAZANIP
        kazanamadığı (identifiability) ölçülebilir.
        """
        if not self.fitted_params:
            raise RuntimeError("Önce fit() çağrılmalı.")
        true_params = true_params or get_true_params()
        est = flatten_params(self.fitted_params)
        tru = flatten_params(true_params)
        bound_map = dict(zip(self._names, self._bounds))

        rows = []
        for name, e in est.items():
            t = tru.get(name, np.nan)
            lo, hi = bound_map[name]
            span = hi - lo
            if abs(e - lo) <= 1e-3 * span:
                status = "Alt sınırda"
            elif abs(e - hi) <= 1e-3 * span:
                status = "Üst sınırda"
            else:
                status = "-"
            rel = (e - t) / abs(t) * 100 if t not in (0, np.nan) and not np.isnan(t) else np.nan
            rows.append({
                "Parametre": name, "Gerçek": t, "Tahmin": e,
                "Mutlak Hata": abs(e - t), "Hata %": rel, "Sınır Durumu": status,
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    df_data = generate_spillover_mmm_data()
    evaluator = SpilloverMMMEvaluator(
        df=df_data,
        channels=["tv", "influencer", "meta_ads", "google_search"],
        spillover_edges=[("tv", "google_search"), ("influencer", "google_search"), ("tv", "meta_ads")],
        cannibalization_edges=[("meta_ads", "google_search")],
    )
    evaluator.fit()
    pd.set_option("display.width", 140)
    print(evaluator.parameter_recovery_table().round(3).to_string(index=False))
