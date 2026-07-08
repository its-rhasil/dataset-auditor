def imbalance(df, target_column):
    missing_count = df[target_column].isnull().sum()
    target = df[target_column].dropna()
    vc = target.value_counts()
    minority_count = float(vc.iloc[-1])
    majority_count = float(vc.iloc[0])
    is_imbalanced = majority_count != minority_count
    return {
        "target": target_column,
        "missing_count": missing_count,
        "class_counts": vc.to_dict(),
        "class_pct": ((vc/len(target))*100).to_dict(),
        "num_classes": target.nunique(),
        "minority_class": vc.index[-1] if is_imbalanced else None,
        "minority_pct": (minority_count/len(target))*100 if is_imbalanced else None,
        "majority_class": vc.index[0] if is_imbalanced else None,
        "majority_pct": (majority_count/len(target)) * 100 if is_imbalanced else None,
        "imbalance_ratio": majority_count/minority_count
        }
