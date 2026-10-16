import sys
import os

# Bulunduğumuz klasörü Python modül arama yolunun en başına ekliyoruz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from typing import List, Tuple, Dict

from transformations import (
    apply_geometric_adstock,
    apply_hill_saturation,
    apply_cross_channel_interactions,
)
from data_generator import generate_spillover_mmm_data


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

        # Trend ve mevsimsellik bileşenleri
        self.t = np.arange(len(df))
        self.seasonality = np.sin(2 * np.pi * self.t / 52.14)

        self.fitted_params = {}

    def _unpack_params(self, params: np.ndarray) -> Dict:
        """1D parametre dizisini okunabilir sözlük yapısına dönüştürür."""
        idx = 0
        unpacked = {"channels": {}, "spillover": {}, "cannibalization": {}, "baseline": {}}

        # 1. Kanal parametreleri (alpha, K, S, beta)
        for ch in self.channels:
            unpacked["channels"][ch] = {
                "alpha": params[idx],
                "half_sat": params[idx + 1],
                "slope": params[idx + 2],
                "beta": params[idx + 3],
            }
            idx += 4

        # 2. Spillover (gamma)
        for edge in self.spillover_edges:
            unpacked["spillover"][edge] = params[idx]
            idx += 1

        # 3. Cannibalization (eta)
        for edge in self.cannibalization_edges:
            unpacked["cannibalization"][edge] = params[idx]
            idx += 1

        # 4. Baseline (Intercept, Trend, Seasonality)
        unpacked["baseline"] = {
            "intercept": params[idx],
            "trend": params[idx + 1],
            "seasonality": params[idx + 2],
        }

        return unpacked

    def _predict_sales(self, p: Dict) -> np.ndarray:
        """Verilen parametre setiyle satış tahmin dizisi üretir."""
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
                effective_spends[ch],
                p["channels"][ch]["half_sat"],
                p["channels"][ch]["slope"],
            )
            y_pred += sat * p["channels"][ch]["beta"]

        return y_pred

    def _objective_function(self, params: np.ndarray) -> float:
        """Tahmin hatasını (RMSE) hesaplayan kayıp fonksiyonu."""
        p = self._unpack_params(params)
        y_pred = self._predict_sales(p)
        return float(np.sqrt(np.mean((self.y_true - y_pred) ** 2)))

    def fit(self) -> Dict:
        """Sınırlandırılmış L-BFGS-B algoritmasıyla parametreleri bulur."""
        initial_guess = []
        bounds = []

        # Kanal sınırları: alpha [0, 0.9], K [1, 100], S [0.1, 3.0], beta [0, 200]
        for _ in self.channels:
            initial_guess.extend([0.3, 20.0, 1.0, 50.0])
            bounds.extend([(0.0, 0.9), (1.0, 100.0), (0.1, 3.0), (0.0, 200.0)])

        # Spillover (gamma) sınırları
        for _ in self.spillover_edges:
            initial_guess.append(0.1)
            bounds.append((0.0, 1.0))

        # Cannibalization (eta) sınırları
        for _ in self.cannibalization_edges:
            initial_guess.append(0.05)
            bounds.append((0.0, 0.5))

        # Baseline sınırları
        initial_guess.extend([100.0, 0.0, 10.0])
        bounds.extend([(0.0, 300.0), (-2.0, 2.0), (-30.0, 30.0)])

        print("Model Eğitimi Başladı (L-BFGS-B Optimization)...")

        result = minimize(
            self._objective_function,
            np.array(initial_guess),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        self.fitted_params = self._unpack_params(result.x)

        y_pred = self._predict_sales(self.fitted_params)
        r2 = r2_score(self.y_true, y_pred)
        mape = mean_absolute_percentage_error(self.y_true, y_pred)

        print(f"--- Eğitim Tamamlandı ---")
        print(f"Model R² Skoru: {r2:.4f}")
        print(f"Model MAPE (Hata Oranı): %{mape * 100:.2f}")

        return self.fitted_params


if __name__ == "__main__":
    df_data = generate_spillover_mmm_data()

    channels_list = ["tv", "influencer", "meta_ads", "google_search"]
    spillover_list = [
        ("tv", "google_search"),
        ("influencer", "google_search"),
        ("tv", "meta_ads"),
    ]
    cannibalization_list = [("meta_ads", "google_search")]

    evaluator = SpilloverMMMEvaluator(
        df=df_data,
        channels=channels_list,
        spillover_edges=spillover_list,
        cannibalization_edges=cannibalization_list,
    )

    fitted_results = evaluator.fit()