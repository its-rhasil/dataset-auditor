def detect_duplicates(df):
    dupes = df.duplicated()
    return {
        "duplicated_count": int(dupes.sum()),
        "duplicated_pct": float(dupes.mean() * 100),
        "duplicated_indices": df[dupes].index.to_list()
    }