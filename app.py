# -*- coding: utf-8 -*-
"""
Spillover & Cannibalization Aware MMM Budget Optimizer (Streamlit)
Çalıştırma:  streamlit run app.py
"""

import os
import sys

# Dosyanın bulunduğu klasörü modül yoluna ekle (Spyder / farklı çalışma dizinleri için)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate_spillover_mmm_data
from evaluator import SpilloverMMMEvaluator
from optimizer import SpilloverBudgetOptimizer, check_feasibility

MAX_BUDGET = 300.0

st.set_page_config(page_title="MMM Budget Optimizer", layout="wide", initial_sidebar_state="expanded")

st.title("🚀 Spillover & Cannibalization Aware MMM Optimizer")
st.markdown(
    "Bu karar destek sistemi, kanallar arası **sıçrama (spillover)** ve **yamyamlık (cannibalization)** "
    "etkilerini hesaba katarak kısıtlı pazarlama bütçesini maksimum satış getirecek şekilde dağıtır."
)


@st.cache_resource
def load_and_train_model():
    """Model her slider değişiminde yeniden eğitilmesin diye önbelleğe alınır."""
    df = generate_spillover_mmm_data()
    channels = ["tv", "influencer", "meta_ads", "google_search"]
    spillover = [("tv", "google_search"), ("influencer", "google_search"), ("tv", "meta_ads")]
    cannibalization = [("meta_ads", "google_search")]

    evaluator = SpilloverMMMEvaluator(df, channels, spillover, cannibalization)
    fitted_params = evaluator.fit(n_starts=6, verbose=False)
    return {
        "fitted_params": fitted_params,
        "hist_means": {ch: float(df[f"spend_{ch}"].mean()) for ch in channels},
        "hist_max": {ch: float(df[f"spend_{ch}"].max()) for ch in channels},
        "fit_report": evaluator.fit_report,
        "recovery": evaluator.parameter_recovery_table(),
    }


with st.spinner("Ekonometrik parametreler eğitiliyor (ilk açılışta ~20 sn)..."):
    art = load_and_train_model()

fitted_params = art["fitted_params"]
historical_means = art["hist_means"]
historical_max = art["hist_max"]
channels = list(fitted_params["channels"].keys())

# --- YAN PANEL ---
st.sidebar.header("💰 Bütçe & Kısıt Yönetimi")
total_budget = st.sidebar.slider(
    "Haftalık Toplam Bütçe (Bin TL)", min_value=50.0, max_value=MAX_BUDGET, value=100.0, step=5.0
)
st.sidebar.markdown("---")
st.sidebar.subheader("Kanal Sınırları (Min - Max)")

# Slider aralığı sabit ve key verildiği için bütçe değişince kanal sınırları sıfırlanmaz.
channel_bounds = {}
for ch in channels:
    channel_bounds[ch] = st.sidebar.slider(
        f"{ch.upper()} Harcama Sınırı",
        min_value=0.0, max_value=MAX_BUDGET, value=(5.0, 120.0), step=1.0, key=f"bounds_{ch}",
    )

# --- KISIT KONTROLÜ ---
problem = check_feasibility(total_budget, channel_bounds, channels)
if problem:
    st.error(f"⚠️ Kısıtlar uygulanamaz: {problem} Yan paneldeki sınırları gevşetin.")
    st.stop()

# --- OPTİMİZASYON ---
optimizer = SpilloverBudgetOptimizer(fitted_params, historical_means)
opt = optimizer.optimize_budget(total_budget=total_budget, channel_bounds=channel_bounds)
alloc = opt["optimized_allocation"]

# Karşılaştırma tabanı: geçmiş harcama oranlarının aynen korunduğu dağılım
hist_total = sum(historical_means.values())
current = {ch: historical_means[ch] / hist_total * total_budget for ch in channels}
current_sales = optimizer.evaluate(current)
opt_sales = opt["expected_steady_state_sales"]
lift_pct = (opt_sales / current_sales - 1.0) * 100 if current_sales > 0 else 0.0

tab_opt, tab_model = st.tabs(["📊 Optimizasyon", "🔬 Model Doğrulama"])

with tab_opt:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Bütçe", f"{total_budget:,.1f} Bin TL")
    c2.metric(
        "Pazarlama Kaynaklı Satış (kararlı durum)", f"{opt_sales:,.2f}",
        delta=f"{lift_pct:+.1f}% (mevcut dağılıma göre)",
    )
    c3.metric("Mevcut Dağılımın Satışı", f"{current_sales:,.2f}")
    c4.metric(
        "Optimizasyon",
        "Başarılı" if opt["success"] else "Başarısız",
        help=f"SLSQP, {opt['n_converged']}/{opt['n_starts']} başlangıç noktası yakınsadı.",
    )
    st.caption(
        "Satış değerleri yalnızca pazarlamanın yarattığı kısmı gösterir (organik baseline hariç). "
        "'Mevcut dağılım', geçmiş ortalama harcama oranlarının bu bütçeye ölçeklenmiş halidir."
    )

    # Veri aralığı dışı uyarıları
    if total_budget > 2 * hist_total:
        st.warning(
            f"Seçilen bütçe geçmiş ortalama haftalık toplamın ({hist_total:,.0f} Bin TL) "
            f"{total_budget / hist_total:.1f} katı. Model bu aralıkta gözlem görmedi; sonuçlar ekstrapolasyondur."
        )
    outside = [ch.upper() for ch in channels if alloc[ch] > historical_max[ch]]
    if outside:
        st.warning(
            "Şu kanallarda önerilen harcama geçmişteki en yüksek haftalık değeri aşıyor: "
            + ", ".join(outside) + ". Bu kanallar için tahminler güvenilir değildir."
        )

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Mevcut vs Önerilen Dağılım")
        names = [ch.upper() for ch in channels]
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="Mevcut", x=names, y=[current[ch] for ch in channels],
                                 text=[f"{current[ch]:.1f}" for ch in channels]))
        fig_cmp.add_trace(go.Bar(name="Önerilen", x=names, y=[alloc[ch] for ch in channels],
                                 text=[f"{alloc[ch]:.1f}" for ch in channels]))
        fig_cmp.update_layout(barmode="group", yaxis_title="Bin TL / hafta", legend=dict(orientation="h"))
        st.plotly_chart(fig_cmp)

    with right:
        st.subheader("Kanal Yanıt Eğrileri")
        grid = np.linspace(0, total_budget, 60)
        fig_resp = go.Figure()
        for ch in channels:
            curve = optimizer.channel_response_curve(ch, grid, alloc)
            fig_resp.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name=ch.upper()))
            fig_resp.add_trace(go.Scatter(
                x=[alloc[ch]], y=[optimizer.channel_response_curve(ch, np.array([alloc[ch]]), alloc)[0]],
                mode="markers", marker=dict(size=10), showlegend=False,
            ))
        fig_resp.update_layout(
            xaxis_title="Kanal harcaması (Bin TL / hafta)",
            yaxis_title="Artımlı satış (spillover + yamyamlık dahil)",
        )
        st.plotly_chart(fig_resp)
        st.caption("Diğer kanallar optimal değerde sabit tutulur. Noktalar önerilen harcamayı gösterir.")

    # Kanal bazlı özet tablo
    st.subheader("Kanal Bazlı Özet")
    mroas = optimizer.marginal_returns(alloc)
    rows = []
    for ch in channels:
        lo, hi = opt["bounds"][ch]
        if abs(alloc[ch] - hi) < 0.05:
            state = "Üst sınırda"
        elif abs(alloc[ch] - lo) < 0.05:
            state = "Alt sınırda"
        else:
            state = "Serbest"
        rows.append({
            "Kanal": ch.upper(),
            "Mevcut (Bin TL)": round(current[ch], 2),
            "Önerilen (Bin TL)": alloc[ch],
            "Fark": round(alloc[ch] - current[ch], 2),
            "Marjinal getiri (satış / Bin TL)": round(mroas[ch], 3),
            "Kısıt Durumu": state,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "Doğrulama: Serbest kanalların marjinal getirisi birbirine eşit olmalıdır. "
        "Sınırda kalan kanallar, getirisi daha yüksek olsa da kısıt nedeniyle daha fazla bütçe alamamıştır."
    )

    # Model çıktısından türetilen içgörü
    upper_share = (alloc.get("tv", 0.0) + alloc.get("influencer", 0.0)) / total_budget * 100
    sp_gain, sp_total = optimizer.spillover_contribution(alloc)
    sp_pct = sp_gain / sp_total * 100 if sp_total > 0 else 0.0
    st.info(
        f"💡 Bütçenin %{upper_share:.0f}'i üst huni kanallarına (TV + Influencer) ayrılıyor. "
        f"Modele göre spillover etkisi olmasaydı pazarlama kaynaklı satış yaklaşık %{sp_pct:.1f} daha düşük olurdu."
    )

with tab_model:
    rep = art["fit_report"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (eğitim verisi)", f"{rep['r2']:.3f}")
    m2.metric("MAPE", f"%{rep['mape'] * 100:.2f}")
    m3.metric("RMSE", f"{rep['rmse']:.2f}")
    m4.metric("Başlangıç noktası", f"{rep['n_starts']}",
              help=f"RMSE aralığı: {min(rep['rmse_per_start']):.2f} – {max(rep['rmse_per_start']):.2f}")
    st.caption(
        "Bu metrikler eğitim verisi üzerinde (in-sample) hesaplanır; veri modelle aynı fonksiyonel formdan "
        "üretildiği için yüksek çıkması beklenir. Asıl sınav aşağıdaki parametre geri kazanımıdır."
    )

    st.subheader("Parametre Geri Kazanımı: Gerçek vs Tahmin")
    rec = art["recovery"].copy()
    rec["Gerçek"] = rec["Gerçek"].round(3)
    rec["Tahmin"] = rec["Tahmin"].round(3)
    rec["Mutlak Hata"] = rec["Mutlak Hata"].round(3)
    rec["Hata %"] = rec["Hata %"].round(1)
    st.dataframe(rec, hide_index=True)

    big = int((rec["Hata %"].abs() > 25).sum())
    st.caption(
        f"{len(rec)} parametrenin {big} tanesinde sapma %25'ten büyük. Büyük sapmalar, toplam satıştan bu "
        "parametrelerin ayrıştırılmasının zor olduğunu (tanılanabilirlik sorunu) gösterir. Beta ve yarı-doyum (K) "
        "birbirini telafi edebilir; spillover (γ) ve doğrudan etki (β) de aynı şekilde."
    )
