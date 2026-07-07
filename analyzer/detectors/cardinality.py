import pandas as pd
import numpy as np

def cardinality(df):
    results = {}
    for col in df.columns:
        results[col] = {}
        nunique = df[col].nunique()
        unique_ratio = nunique / len(df[col])
        is_constant = True if nunique <= 1 else False
        if nunique == 1:
            top_value_freq = 1
            top_value = df[col].value_counts().index[0]
            results[col]["top_value"] = top_value
        elif nunique == 0:
            top_value_freq = 0
            results[col]["nunique"] = None
            results[col]["unique_ratio"] = None
            results[col]["is_constant"] = None
            results[col]["dtype_category"] = "NaN"
            results[col]["top_value_freq"] = None
            results[col]["cv"] = None
            results[col]["top_value"] = None
            continue
        else:
            vc = df[col].value_counts(normalize = True)
            top_value_freq = vc.iloc[0]
            top_value = vc.index[0]
        results[col]["nunique"] = nunique
        results[col]["unique_ratio"] = unique_ratio
        results[col]["is_constant"] = is_constant
        if pd.api.types.is_numeric_dtype(df[col]):
            if nunique <= 10:
                results[col]["dtype_category"] = "Categorical"
                results[col]["top_value_freq"] = top_value_freq
                results[col]["cv"] = None
                results[col]["top_value"] = top_value
            else:
                results[col]["dtype_category"] = "Numerical"
                results[col]["top_value_freq"] = None
                cv = (df[col].std() / abs(df[col].mean())) if abs(df[col].mean()) > 1e-10 else np.nan
                results[col]["cv"] = cv
                results[col]["top_value"] = None
        else:
            results[col]["dtype_category"] = "Categorical"
            results[col]["top_value_freq"] = top_value_freq
            results[col]["top_value"] = top_value
            results[col]["cv"] = None
    return results