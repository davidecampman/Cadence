---
name: data-analysis
version: 1.0.0
description: Analyze structured data with code execution, statistics, and plain-language interpretation
author: cadence
dependencies: []
tags:
  - data
  - analysis
  - python
  - statistics
---

# Instructions

When asked to analyze data, follow this process:

1. **Understand the data first** — Read a sample before writing analysis code. Check shape, dtypes, null counts, and obvious anomalies.
2. **State your approach** — Before running code, briefly describe what you will compute and why.
3. **Write clean, runnable code** — Use pandas, numpy, and matplotlib/seaborn. Handle missing values explicitly. Do not silently drop rows.
4. **Execute and verify** — Run the code and include actual output in your response. If it errors, debug and fix before presenting results.
5. **Interpret in plain language** — Explain what the numbers mean. Never present a table or chart without a sentence explaining the key takeaway.

Analysis checklist:
- **Descriptive stats**: mean, median, std, min/max, percentiles for numeric columns
- **Distribution check**: are values normally distributed, skewed, or bimodal?
- **Correlations**: which variables move together? Use Pearson for linear, Spearman for rank
- **Outliers**: flag values beyond 3σ or IQR × 1.5 — ask whether to include or exclude them
- **Time series** (if applicable): trend, seasonality, anomalies

Visualization rules:
- Use matplotlib with `plt.tight_layout()` and clear axis labels
- Save figures with `plt.savefig('output.png', dpi=150, bbox_inches='tight')` and reference them as `[[FILE:output.png]]`
- Choose chart type intentionally: bar for categories, line for time series, scatter for correlation, histogram for distribution

## Examples

- Analyze this CSV of sales data and identify the top-performing regions
- What's the correlation between these variables in my dataset?
- Find anomalies in this time series of server latency measurements
- Summarize the distribution of user ages in this JSON export
