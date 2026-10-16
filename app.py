# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 09:51:38 2026

@author: mcany
"""

import sys
import os

# Bulunduğumuz klasörü Python yoluna ekliyoruz
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from transformations import apply_hill_saturation
from data_generator import generate_spillover_mmm_data
from evaluator import SpilloverMMMEvaluator
from optimizer import SpilloverBudgetOptimizer

# Sayfa Yapılandırması
st.set_page_config(page_title="MMM Budget Optimizer", layout="wide", initial_sidebar_state="expanded")

st.title("🚀 Spillover & Cannibalization Aware MMM Optimizer")
st.markdown("""
Bu karar destek sistemi, kanallar arası **Sıçrama (Spillover)** ve **Yamyamlık (Cannibalization)** 
etkilerini hesaba katarak kısıtlı pazarlama bütçesini maksimum satış getirecek şekilde dağıtır.
""")

# Model Eğitimini Ön Belleğe Alma (Her slider değişiminde modeli tekrar eğitip yavaşlamasın diye)
@st.cache_resource
def load_and_train_model():
    df = generate_spillover_mmm_data()
    channels = ["tv", "influencer", "meta_ads", "google_search"]
    spillover = [("tv", "google_search"), ("influencer", "google_search"), ("tv", "meta_ads")]
    cannibalization = [("meta_ads", "google_search")]

    evaluator = SpilloverMMMEvaluator(df, channels, spillover, cannibalization)
    fitted_params = evaluator.fit()
    hist_means = {ch: float(df[f"spend_{ch}"].mean()) for ch in channels}
    return fitted_params, hist_means

with st.spinner("Ekonometrik parametreler eğitiliyor..."):
    fitted_params, historical_means = load_and_train_model()

# --- YAN PANEL: BÜTÇE VE KISIT AYARLARI ---
st.sidebar.header("💰 Bütçe & Kısıt Yönetimi")
total_budget = st.sidebar.slider("Haftalık Toplam Bütçe (Bin TL)", min_value=50.0, max_value=300.0, value=100.0, step=5.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Kanal Sınırları (Min - Max)")

channels = list(fitted_params["channels"].keys())
channel_bounds = {}

for ch in channels:
    min_v, max_v = st.sidebar.slider(
        f"{ch.upper()} Harcama Sınırı",
        min_value=0.0,
        max_value=total_budget,
        value=(5.0, total_budget * 0.5),
        step=1.0
    )
    channel_bounds[ch] = (min_v, max_v)

# --- OPTİMİZASYONU ÇALIŞTIRMA ---
optimizer = SpilloverBudgetOptimizer(fitted_params, historical_means)
opt_result = optimizer.optimize_budget(total_budget=total_budget, channel_bounds=channel_bounds)

# --- METRİKLER ---
col1, col2, col3 = st.columns(3)
col1.metric("Toplam Bütçe", f"{total_budget:,.1f} Bin TL")
col2.metric("Tahmini Kararlı Durum Satışı", f"{opt_result['expected_steady_state_sales']:,.2f}")
col3.metric("Optimizasyon Durumu", "Başarılı (SLSQP)" if opt_result["success"] else "Başarısız")

st.markdown("---")

# --- GRAFİKLER ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Optimal Bütçe Dağılımı")
    alloc_data = opt_result["optimized_allocation"]
    df_alloc = pd.DataFrame({
        "Kanal": [ch.upper() for ch in alloc_data.keys()],
        "Tahsis Edilen Bütçe (Bin TL)": list(alloc_data.values())
    })
    fig_bar = px.bar(
        df_alloc, x="Kanal", y="Tahsis Edilen Bütçe (Bin TL)", 
        text_auto=True, color="Kanal", color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("Azalan Verimler (Hill Saturation) Eğrileri")
    x_grid = np.linspace(0, total_budget, 100)
    fig_hill = go.Figure()

    for ch in channels:
        p = fitted_params["channels"][ch]
        y_grid = apply_hill_saturation(x_grid, p["half_sat"], p["slope"]) * p["beta"]
        fig_hill.add_trace(go.Scatter(x=x_grid, y=y_grid, mode="lines", name=ch.upper()))

        # Optimal harcama noktasını işaretle
        opt_x = alloc_data[ch]
        opt_y = apply_hill_saturation(np.array([opt_x]), p["half_sat"], p["slope"])[0] * p["beta"]
        fig_hill.add_trace(go.Scatter(x=[opt_x], y=[opt_y], mode="markers", marker=dict(size=10), showlegend=False))

    fig_hill.update_layout(xaxis_title="Harcama (Bin TL)", yaxis_title="Marjinal Satış Katkısı")
    st.plotly_chart(fig_hill, use_container_width=True)

st.info("💡 **Mülakat Notu:** TV ve Influencer gibi üst huni kanallarına ayrılan bütçe, Google Search üzerindeki Spillover (sıçrama) etkisinden dolayı satışları dolaylı olarak besler.")