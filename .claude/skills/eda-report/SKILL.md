---
name: eda-report
description: >
  Generate an exploratory data analysis report for a dataset.
  Use when the user asks to explore data, profile a dataset,
  or get a quick data overview.
---

# EDA Report Skill

## Steps
1. Load the dataset with pandas, print shape and dtypes
2. Run `df.describe()` and flag any nulls > 5%
3. For numeric cols: plot histograms via @scripts/plot_hist.py
4. For categorical cols: show value_counts top 10
5. Summarize findings in a markdown report saved to reports/eda_{dataset}.md

## Output format
Always end with a "Red Flags" section listing anything anomalous.