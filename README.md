# 🚀 Spillover-Aware Marketing Mix Modeling (MMM) & Budget Optimizer

## 📌 About The Author

**Mertcan Yoldaş**

*Senior Double Major Student in Management Information Systems (MIS) & Public Relations/Advertising.*

This project bridges the gap between technical data science (constrained optimization, predictive modeling) and real-world marketing strategy (brand halo effects, media buying dynamics).

## 🎯 The Business Problem

Traditional Marketing Mix Models (MMM) often evaluate advertising channels in isolated silos. They assume spending on TV only affects TV metrics, and spending on Meta only affects Meta metrics. However, real-world consumer behavior is highly interconnected:

1. **Spillover (Halo Effect):** Upper-funnel campaigns (e.g., TV, Influencers) generate brand awareness that directly triggers lower-funnel search volumes (e.g., Google Search).

2. **Cannibalization:** Overspending in overlapping performance channels (e.g., Meta Retargeting vs. Google Brand Search) results in diminishing marginal returns and stolen attribution.

**The Solution:** I developed an end-to-end MVP that estimates these cross-channel interactions mathematically and utilizes constrained optimization to allocate marketing budgets for maximum steady-state ROI.

## ✨ Key Features & Architecture

* **Custom Synthetic Data Engine:** Since real historical budget data is confidential, this MVP includes a generator that simulates 3 years of weekly e-commerce data with built-in seasonality, baseline sales, Adstock (carryover), and Hill saturation dynamics. *(Ready to ingest real CSV data in production).*

* **Cross-Channel Interaction Matrix:** Models the $\gamma$ (Spillover) and $\eta$ (Cannibalization) coefficients between upper and lower funnel channels.

* **Bounded Optimization Evaluator:** Uses the `SciPy L-BFGS-B` algorithm to reverse-engineer and extract transformation parameters from historical data, avoiding the massive computational costs of black-box MCMC simulations.

* **Steady-State Budget Optimizer:** Employs the `SLSQP` algorithm to allocate budgets under strict business constraints (min/max channel spends enforced by management), maximizing total expected sales.

* **Interactive Dashboard:** A Streamlit web application designed for C-level executives (CMOs) to simulate budget scenarios, apply constraints, and visualize diminishing returns curves in real-time.

## 🧮 Mathematical Framework

The model is built on three core econometric transformations:

**1. Geometric Adstock (Carryover Effect)**

Advertising impact doesn't vanish overnight. The effective spend $A_t$ at time $t$ is:

> $A_t = X_t + \alpha \cdot A_{t-1}$ *(where* $0 \le \alpha < 1$*)*

**2. Cross-Channel Interaction (Spillover & Cannibalization)**

Before saturation, effective spends are adjusted for cross-channel impacts:

> $X_{target}^* = A_{target} + \sum (\gamma \cdot A_{source}) \times (1 - \text{Cannibalization Penalty})$

**3. Hill Saturation (Diminishing Returns)**

Marginal ROI decreases as spend increases. Modeled via the Hill function:

> $S = \frac{X^{*S}}{K^S + X^{*S}}$ *(where* $K$ *is half-saturation point,* $S$ *is shape/slope)*

## 📂 Project Structure

```
mmm-budget-optimizer/
│
├── src/
│   ├── transformations.py     # Core mathematical functions (Adstock, Hill, Matricies)
│   ├── data_generator.py      # Synthetic data pipeline (The feeding engine)
│   ├── evaluator.py           # Parameter estimation via L-BFGS-B
│   └── optimizer.py           # Steady-state budget allocation via SLSQP
│
├── app.py                     # Streamlit interactive dashboard
├── requirements.txt           # Environment dependencies
└── README.md                  # Project documentation

```

## ⚙️ Installation & Usage

**1. Clone the repository:**

```
git clone https://github.com/yourusername/mmm-budget-optimizer.git
cd mmm-budget-optimizer

```

**2. Install dependencies:**

```
pip install -r requirements.txt

```

**3. Run the interactive dashboard:**

```
streamlit run app.py

```

*Once running, navigate to `http://localhost:8501` to use the interactive budget simulation UI.*