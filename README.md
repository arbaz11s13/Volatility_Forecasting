# Equity Volatility Forecasting and Volatility Targeting

## Overview

This project studies daily equity volatility forecasting across short and medium horizons and evaluates whether improvements in statistical forecast accuracy translate into economically meaningful improvements in portfolio risk management.

The analysis uses **SPY**, representing the broad U.S. equity market, and **JPM**, representing a large individual stock with greater idiosyncratic risk.

The project has three main goals:

1. Compare progressively richer volatility forecasting models.
2. Understand which volatility dynamics matter at different forecast horizons.
3. Translate volatility forecasts into a practical **volatility-targeting strategy** and evaluate their economic value.

The central empirical result is horizon-dependent:

* **GJR-GARCH performs best for 1-day volatility forecasting.**
* **HAR-RV performs best for 5-day volatility forecasting.**
* Dynamic volatility targeting substantially stabilizes portfolio risk and improves risk-adjusted performance relative to static risk scaling.

---

## Research Questions

The project addresses four questions:

1. How much improvement do increasingly sophisticated volatility models provide over simple historical benchmarks?
2. Does modeling the asymmetric response to negative equity returns improve short-horizon volatility forecasts?
3. Does multi-scale volatility persistence become more useful at longer forecast horizons?
4. Do statistically better volatility forecasts produce economically useful improvements in portfolio risk control?

---

## Data

Daily OHLCV data are downloaded using `yfinance`.

Assets:

* **SPY** — S&P 500 ETF
* **JPM** — JPMorgan Chase

The dataset begins in January 2000.

Adjusted close prices are used to calculate returns, while raw OHLC data are retained for potential extensions involving range-based volatility estimators.

Daily percentage log returns are defined as:

$$
r_t=100\log\left(\frac{P_t}{P_{t-1}}\right).
$$

The data are stored in tidy panel format with a unique `(date, ticker)` key.

---

## Forecast Targets

Volatility is latent, so squared daily returns are used as a daily realized-variance proxy.

### 1-day realized variance

$$
RV^{(1)}_t=r_{t+1}^2
$$

### 5-day realized variance

$$
RV^{(5)}_t = \sum_{i=1}^{5}r_{t+i}^2
$$

All targets are explicitly aligned forward in time. Features and model estimation at time \(t\) use only information available at or before \(t\).

---

## Evaluation Design

The final evaluation period begins on:

```text
2005-01-01
```

Earlier observations are used for model initialization and development where necessary.

The workflow uses chronological / expanding-window estimation rather than random train-test splits in order to preserve the time-series structure and avoid look-ahead bias.

Two market-stress periods are also evaluated separately:

* **Global Financial Crisis:** July 2007 – June 2009
* **COVID-19 shock:** February 2020 – May 2020

Regime labels are used only for evaluation and visualization, not as forecasting inputs.

---

# Models

## 1. Historical Variance

A rolling historical variance forecast provides the simplest benchmark.

For a rolling window \(N\),

$$
\hat\sigma_t^2 = Var(r_{t-N+1},...,r_t).
$$

A 20-day window is used.

### Motivation

Historical variance captures the recent local volatility level but responds slowly to abrupt changes in market conditions.

---

## 2. EWMA

The RiskMetrics-style exponentially weighted moving average updates variance recursively:

$$
\sigma_{t+1}^2 = \lambda\sigma_t^2 + (1-\lambda)r_t^2
$$

The baseline uses

$$
\lambda=0.94.
$$

### Motivation

EWMA assigns greater importance to recent observations and therefore adapts more quickly to volatility clustering than a fixed historical window.

It remains highly persistent, however, and can react slowly to sudden regime changes.

---

## 3. GARCH(1,1)

The GARCH model estimates volatility as

$$
\sigma_{t+1}^2 = \omega + \alpha r_t^2 + \beta\sigma_t^2
$$

Student-\(t\) innovations are used to better accommodate heavy-tailed equity returns.

Model parameters are periodically refitted using an expanding historical sample, while conditional variance is updated recursively each trading day.

### Interpretation

* $\omega$: long-run variance component
* $\alpha$: sensitivity to recent shocks
* $\beta$: volatility persistence

The estimated values generally show high persistence, with $\alpha+\beta$ close to one.

### Motivation

Unlike EWMA, GARCH learns both shock sensitivity and persistence from the data and allows volatility to mean-revert toward a long-run level.

---

## 4. GJR-GARCH

Standard GARCH treats positive and negative returns symmetrically.

Equity markets often exhibit a **leverage effect** in which negative returns increase subsequent volatility more than positive returns of equal magnitude.

GJR-GARCH adds an asymmetric term:

$$
\sigma_{t+1}^2=
\omega
+
\alpha r_t^2
+
\gamma r_t^2I(r_t<0)
+
\beta\sigma_t^2.
$$

where $\gamma$ measures the additional volatility impact of negative returns.

### Motivation

The asymmetric specification is particularly relevant to equities, where market declines tend to coincide with rapid increases in volatility.

### Result

GJR-GARCH produced the strongest **1-day forecasts**, including during the GFC and COVID stress periods.

---

## 5. HAR-RV

The Heterogeneous Autoregressive Realized Volatility model uses volatility information at multiple time scales.

Features include:

### Daily component

$$
RV_{d,t}
$$

### Weekly component

$$
RV_{w,t}=
\frac{1}{5}\sum_{i=0}^{4}RV_{t-i}
$$

### Monthly component

$$
RV_{m,t}=
\frac{1}{22}\sum_{i=0}^{21}RV_{t-i}.
$$

A linear regression combines these daily, weekly and monthly volatility components.

### Motivation

Market participants operate at different horizons. HAR-RV approximates the resulting long-memory behavior using a simple and interpretable multi-scale structure.

### Result

HAR-RV was substantially stronger at the **5-day forecasting horizon**, showing that slower-moving volatility information becomes more important at longer horizons.

---

## 6. GJR–HAR Ensemble

A convex ensemble was considered:

$$
\hat\sigma^2_{\text{ensemble}}=
w\hat\sigma^2_{\text{GJR}}
+
(1-w)\hat\sigma^2_{\text{HAR}}.
$$

The weight $w$ was selected by minimizing development-period QLIKE.

The optimized weights were:

$$
w_{1d}=1
$$

and

$$
w_{5d}=0.
$$

Therefore:

* the optimal 1-day ensemble is pure **GJR-GARCH**;
* the optimal 5-day ensemble is pure **HAR-RV**.

Rather than showing complementarity within each horizon, the ensemble optimization effectively performs **model selection by forecast horizon**.

---

## Machine-Learning Experiment

Elastic Net was explored as a regularized machine-learning benchmark using:

* daily, weekly and monthly realized-variance features;
* lagged returns;
* lagged squared returns;
* rolling return and variance features.

Direct variance prediction created positivity issues, while log-transformed targets produced unstable extreme forecasts at longer horizons.

After stabilization, Elastic Net remained less robust and did not outperform GJR-GARCH or HAR-RV.

It was therefore not promoted into the final model set.

This was treated as a model-selection result rather than forcing an ML model into the final specification.

---

# Forecast Evaluation

## QLIKE

The primary forecast loss is:

$$
QLIKE(y,\hat y)=
\log(\hat y)
+
\frac{y}{\hat y}.
$$

Lower values indicate better forecasts.

QLIKE is particularly useful for volatility forecasting because realized variance is itself a noisy proxy for latent conditional variance.

It also strongly penalizes forecasts that severely underestimate risk.

---

## Mean Squared Error

MSE is reported as a secondary metric:

$$
MSE=
\frac{1}{N}
\sum_t
(y_t-\hat y_t)^2.
$$

QLIKE is treated as the main model-selection metric, while MSE provides an intuitive measure of absolute forecast error.

---

## Forecast Performance

The results reveal a clear horizon dependence.

### 1-day horizon

**GJR-GARCH is the strongest model.**

Its advantage suggests that modeling the asymmetric response of equity volatility to negative returns provides meaningful short-horizon predictive information.

### 5-day horizon

**HAR-RV is the strongest model.**

The result suggests that multi-scale persistence and longer-memory volatility structure become increasingly valuable as the forecast horizon increases.

---

# Statistical Significance

Average forecast losses alone do not establish whether performance differences are distinguishable from sampling variation.

Pairwise QLIKE loss differentials are therefore evaluated using a Diebold-Mariano-style framework:

$$
d_t=L_{A,t}-L_{B,t}.
$$

The null hypothesis is

$$
H_0:E[d_t]=0.
$$

Because volatility forecast losses may be heteroskedastic and serially correlated, **HAC / Newey-West standard errors** are used.

The results strongly support the horizon-dependent ranking.

### 1-day

GJR-GARCH significantly outperforms the main competing forecasts for both SPY and JPM.

### 5-day

HAR-RV significantly outperforms GJR-GARCH, GARCH and EWMA for both SPY and JPM.

The GARCH vs GJR comparison is interpreted with additional care because standard GARCH is nested within GJR-GARCH.

---

# Economic Application: Volatility Targeting

Forecast accuracy is only useful if it leads to meaningful decisions.

The final section therefore applies the forecasts to portfolio risk sizing.

The desired exposure is

$$
w_t=
\frac{\sigma_{\text{target}}}
{\hat\sigma_t}.
$$

A **10% annualized volatility target** is used.

When forecast volatility rises, exposure is reduced. When forecast volatility falls, exposure is increased.

Forecast-based weights are shifted forward so that a forecast formed at time $t$ determines exposure during $t+1$

A leverage cap is imposed and transaction costs are deducted based on changes in exposure.

---

# SPY Volatility Targeting

The SPY experiment compares:

1. Buy & Hold
2. Static Risk Scaling
3. EWMA Volatility Targeting
4. GJR Volatility Targeting
5. HAR Volatility Targeting

## Static-Risk Benchmark

A fixed exposure is estimated using only the pre-evaluation development period.

Development-period SPY annualized volatility was approximately:


$$ 20.54 \\% $$

Therefore the fixed exposure required to target 10% risk was:

$$
w_{\text{static}}\approx0.487.
$$

The weight remains unchanged throughout the evaluation period.

This benchmark distinguishes the value of **dynamic volatility forecasting** from the trivial effect of simply taking less market risk.

## SPY Results

| Strategy    | Ann. Return |   Ann. Vol |    Sharpe | Max Drawdown |  Worst Day |
| ----------- | ----------: | ---------: | --------: | -----------: | ---------: |
| Buy & Hold  |      10.91% |     18.91% |     0.642 |      -55.19% |    -10.94% |
| Static Risk |       5.64% |      9.21% |     0.642 |      -30.59% |     -5.33% |
| EWMA VT     |       7.59% |     10.70% | **0.738** |  **-23.72%** |     -5.68% |
| GJR VT      |       7.09% | **10.04%** |     0.733 |      -25.48% | **-5.04%** |
| HAR VT      |       7.57% |     11.43% |     0.696 |      -28.00% |     -7.26% |

Several conclusions follow.

### Static scaling does not improve Sharpe

Buy & Hold and Static Risk both have a Sharpe of approximately 0.64.

This is expected because multiplying every return by a constant scales expected return and volatility proportionally.

### Dynamic targeting stabilizes risk

The dynamic strategies remain close to the 10% target across changing volatility regimes.

GJR targeting achieved approximately:

$$
10.04 \\%
$$

realized annualized volatility.

### Crisis behavior

During the GFC and COVID shock, Buy & Hold and Static Risk experienced large volatility spikes because static exposure cannot react to changing conditions.

Dynamic volatility targeting reduced exposure as forecast risk increased and maintained much more stable realized volatility.

### Better forecast accuracy does not automatically imply the highest Sharpe

GJR provides the strongest 1-day statistical volatility forecast and hits the risk target most accurately, while EWMA produces a slightly higher realized Sharpe.

This is not contradictory: the models forecast **risk**, not future return direction.

---

# 50/50 SPY–JPM Portfolio Extension

The final experiment applies the framework to a fixed-weight portfolio:

$$
w_{SPY}=w_{JPM}=0.5.
$$

Portfolio variance is forecast using individual GJR variance forecasts and a rolling estimate of SPY–JPM correlation:

$$
\sigma_p^2=
w_S^2\sigma_S^2
+
w_J^2\sigma_J^2
+
2w_Sw_J\rho_{SJ}\sigma_S\sigma_J.
$$

A 60-day rolling correlation is used.

This extends the project from individual volatility forecasting to the portfolio relationship

$$
\sigma_p^2=w^\top\Sigma w.
$$

---

## Portfolio Static Benchmark

Development-period portfolio volatility was approximately:

$$
28.92 \\%
$$

The corresponding static exposure required to target 10% volatility was:

$$
w_{\text{static}}\approx0.346.
$$

---

## Portfolio Results

| Strategy          | Ann. Return |  Ann. Vol |    Sharpe | Max Drawdown |  Worst Day |
| ----------------- | ----------: | --------: | --------: | -----------: | ---------: |
| 50/50 Buy & Hold  |      13.24% |    25.42% |     0.616 |      -58.52% |    -13.18% |
| 50/50 Static Risk |       5.16% |     8.79% |     0.616 |      -22.17% |     -4.56% |
| 50/50 GJR VT      |   **7.51%** | **9.99%** | **0.775** |  **-21.26%** | **-4.08%** |

The dynamic strategy maintained realized volatility almost exactly at the desired 10% level.

At roughly comparable risk, GJR volatility targeting increased the Sharpe ratio from approximately

$$
0.62
$$

for static risk scaling to

$$
0.78.
$$

This provides the clearest evidence in the project that dynamic conditional-volatility forecasts contain economically useful information.

---

# Main Findings

The project produces five central conclusions.

### 1. Volatility dynamics depend on forecast horizon

Short-horizon volatility benefits strongly from asymmetric shock modeling, while longer-horizon forecasts benefit more from multi-scale persistence.

### 2. Negative-return asymmetry matters

GJR-GARCH consistently improves 1-day volatility forecasting relative to symmetric GARCH-style alternatives.

### 3. HAR-RV is particularly effective at longer horizons

Daily, weekly and monthly volatility components provide substantial information for 5-day forecasting.

### 4. Lower risk alone does not create better risk-adjusted performance

Static exposure scaling reduces volatility and drawdowns but leaves Sharpe essentially unchanged.

### 5. Dynamic volatility forecasts have economic value

Forecast-based position sizing substantially stabilizes realized volatility during both calm and crisis periods and improves risk-adjusted portfolio performance.

---

# Research Workflow

The project follows the full empirical research chain:

$$
\text{Market feature}
\rightarrow
\text{Model specification}
\rightarrow
\text{Out-of-sample forecast}
\rightarrow
\text{Statistical evaluation}
\rightarrow
\text{Portfolio decision}
\rightarrow
\text{Economic outcome}.
$$

Examples include:

$$
\text{negative-return asymmetry}
\rightarrow
\text{GJR-GARCH}
\rightarrow
\text{better 1-day QLIKE}
$$

and

$$
\text{multi-scale persistence}
\rightarrow
\text{HAR-RV}
\rightarrow
\text{better 5-day forecasts}.
$$

Finally,

$$
\text{conditional-volatility forecast}
\rightarrow
\text{dynamic exposure}
\rightarrow
\text{more stable portfolio risk}.
$$

---

# Project Structure

```text
Volatility-Forecasting/
│
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_visualization.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_garch.ipynb
│   ├── 05_har_rv_ensemble.ipynb
│   └── 06_volatility_targeting.ipynb
│
├── src/
│   ├── data_pipeline.py
│   ├── baselines.py
│   ├── metrics.py
│   ├── garch.py
│   └── har_rv_ensemble.py
│
├── experiments/
│   └── elastic_net.ipynb
│
└── README.md
```

---

# Key Python Libraries

* `numpy`
* `pandas`
* `yfinance`
* `arch`
* `scikit-learn`
* `statsmodels`
* `plotly`

---

# Limitations

Several limitations remain.

### Daily realized-variance proxy

Squared daily returns are noisy measures of latent daily variance. Intraday data would allow construction of higher-quality realized-volatility measures.

### Limited asset universe

The main analysis focuses on SPY and JPM. A larger cross-section would provide stronger evidence about generalization.

### Simplified covariance model

The portfolio application combines GJR marginal variance forecasts with a rolling historical correlation rather than a full dynamic covariance model.

### Simplified execution assumptions

The volatility-targeting analysis includes turnover-based transaction costs but does not fully model funding costs, bid-ask spreads, market impact, or returns earned on unused cash.

### Unpredictable jumps

Conditional volatility models estimate expected future risk given currently available information. They cannot predict genuinely unexpected information shocks before they occur.

---

# Possible Extensions

Potential extensions include:

* intraday realized variance;
* VIX / option-implied volatility as an additional information source;
* larger equity and ETF universes;
* dynamic covariance or DCC-GARCH models;
* volatility-of-volatility features;
* richer transaction-cost and funding assumptions.

---

# Conclusion

The project demonstrates that successful volatility modeling is not simply a matter of selecting the most complicated model.

Different structures dominate at different horizons:

$$
\boxed{\text{GJR-GARCH for short-horizon asymmetric risk}}
$$

and

$$
\boxed{\text{HAR-RV for longer-horizon volatility persistence}}.
$$

More importantly, the project connects statistical forecasting to an economic application.

Dynamic volatility forecasts allow portfolio exposure to respond to changing market conditions, maintaining substantially more stable realized risk through both normal markets and major crises.

The 50/50 SPY–JPM portfolio provides the strongest economic result: GJR-based volatility targeting maintained approximately **10% realized volatility**, reduced maximum drawdown from approximately **59% to 21%**, and increased the Sharpe ratio from approximately **0.62 to 0.78** relative to the unscaled portfolio.
