# Dataset Auditor

An automated dataset quality analysis tool that profiles CSV and Excel datasets, detects data quality issues, and generates a weighted quality score — reducing manual exploratory data analysis from hours to seconds.

**Live Demo:** https://dataset-auditor.streamlit.app  
**Author:** Muhammed Rhasil

---

## What It Does

Upload a CSV or Excel file and the auditor automatically detects:

- Missing values and missingness patterns across columns
- Duplicate rows
- Outliers using the IQR method
- Constant, empty, and high-cardinality columns
- Class imbalance in the target column
- Potential data leakage via mutual information and correlation analysis

Results are presented as an interactive dashboard with visualizations and a weighted **Dataset Quality Score (0–100)**.

---

## Demo

Upload the included `sample_data/titanic.csv` to see a full audit in action.

For a larger test, the UCI Adult Income dataset (~49,000 rows) completes analysis in approximately 15 seconds.

---

## Features

### Dataset Overview
- Row count, column count, memory usage
- Numerical vs categorical column breakdown
- Total missing values and duplicate rows
- Dataset preview

### Dataset Quality Score
A weighted 0–100 score based on six penalty factors:

| Factor | Weight | Method |
|---|---|---|
| Overall missingness | 20 | Squared cell-level missing ratio |
| Severe missing columns | 10 | Columns with >20% missing |
| Duplicate rows | 20 | Linear duplicate percentage |
| Outlier columns | 20 | Columns with >10% IQR outliers |
| Constant/empty columns | 15 | Squared problem column ratio |
| High cardinality columns | 15 | Squared high cardinality ratio |

### Missing Value Analysis
- Horizontal bar chart — missing percentage per column (yellow to red gradient)
- Missing value correlation matrix (advanced view)

### Outlier Analysis
- Bar chart — outlier percentage per numerical column
- Box plot per column with outlier points highlighted (advanced view)

### Cardinality Analysis
- Summary table of all categorical columns
- Status classification: Normal, High Cardinality, Constant, Empty
- Row highlighting for problematic columns

### Class Imbalance Analysis *(requires target column)*
- Class distribution bar chart
- Minority class percentage and imbalance ratio metrics

### Data Leakage Analysis *(requires target column)*
- Mutual information scores per feature
- Flagged suspicious columns highlighted
- Feature vs target correlation heatmap (numerical targets)

---

## Getting Started

### Requirements

- Python 3.11+
- See `requirements.txt` for full dependency list

### Installation

```bash
git clone https://github.com/its-rhasil/dataset-auditor
cd dataset-auditor
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
dataset-auditor/
├── app.py                          # Streamlit dashboard
├── analyzer/
│   ├── profiler.py                 # Orchestrates all detectors
│   ├── scorer.py                   # Weighted quality scoring engine
│   ├── schemas.py                  # DataProfile dataclass
│   ├── visualizations.py           # Plotly and matplotlib charts
│   ├── utils.py                    # Shared utilities (is_categorical)
│   └── detectors/
│       ├── missing.py              # Missing value detection
│       ├── duplicates.py           # Duplicate row detection
│       ├── outliers.py             # IQR and z-score outlier detection
│       ├── cardinality.py          # Feature cardinality analysis
│       ├── imbalance.py            # Class imbalance detection
│       └── leakage.py              # Data leakage detection
├── sample_data/
│   └── titanic.csv
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | Scikit-learn |
| Visualizations | Plotly, Matplotlib, Missingno |
| File Support | OpenPyXL (Excel) |

---

## Known Limitations

- **Column type inference** uses a heuristic based on unique value count (`nunique <= 15` = categorical). Columns like `bedrooms`, `bathrooms`, and `experience` may be misclassified as categorical.
- **Inconsistent casing** in categorical columns (e.g. `"pending"` vs `"PENDING"`) inflates unique value counts and may trigger false high cardinality flags.
- **Format inconsistencies** in numeric columns — columns storing numbers as strings (e.g. "$245.61", "1,000") are classified as categorical due to their string dtype. Automatic format detection and type coercion is planned.
- **Impossible values** (e.g. negative prices) are not currently detected — accuracy validation is planned.
- **Leakage detection** surfaces statistical candidates only and cannot confirm actual leakage without domain knowledge. High correlation or mutual information does not necessarily indicate leakage.
- **Performance** degrades on datasets above ~50,000 rows due to mutual information computation. A 96,000-row dataset takes approximately 15 seconds to analyze.

---

## Planned Enhancements

- **User-guided column type override** — allow users to correct misclassified columns before analysis
- **ML Readiness Score** — separate score evaluating dataset suitability for model training, including sample adequacy and feature-target relationships
- **Correlated missing value penalty** — penalize columns that are both highly correlated and missing together
- **Audit history** — SQLite-backed persistence of past audits with comparison support
- **LLM-generated summary** — plain-language executive summary of audit findings
- **Dataset fingerprinting** — detect if the same dataset has been audited before
- **Impossible value detection** — domain-aware flagging of out-of-range values

---

## License

MIT