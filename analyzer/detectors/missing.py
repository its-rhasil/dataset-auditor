def detect_missing(df):
    counts = df.isnull().sum().to_dict()
    pct = df.isnull().mean().mul(100).to_dict()
    return {
        "total_missing": int(df.isnull().sum().sum()),
        "columns": {key: {"count": counts[key],"pct": round(pct[key],2)} for key in counts if counts[key] > 0},
        "high_missing_cols": [key for key, value in pct.items() if value > 20]
        }