import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from analyzer.utils import is_categorical

def leakage(df, target_column):
    df1 = df.copy()
    corr = df1.select_dtypes(include="number").corr()[target_column].drop(target_column).to_dict() if not is_categorical(df[target_column]) else None

    encoder = OrdinalEncoder()
    cat_columns = df1.drop(target_column,axis=1).select_dtypes(exclude='number').columns
    if len(cat_columns)> 0:
        df1[cat_columns] = encoder.fit_transform(df1[cat_columns])
    structural_clones = []
    mi_scores = {}
    for col in df1.drop(target_column,axis=1).columns:
        temp = df1[[col, target_column]].dropna().copy()
        if temp.empty:
            continue
        if temp[col].nunique() == temp[target_column].nunique():
            if temp.groupby(col)[target_column].nunique().max() == 1:
                structural_clones.append(col)
        x = temp[[col]]
        y = temp[target_column]
        mi = mutual_info_classif(x, y,discrete_features=is_categorical(df[col]), random_state=42) if is_categorical(df[target_column]) else mutual_info_regression(x, y,random_state=42)
        mi_scores[col] = mi[0]
    mi_scores = pd.Series(mi_scores)
    median = np.median(mi_scores)
    mad = np.median(np.abs(mi_scores - median))
    threshold = median + (2.5 * mad)

    flagged_corr = [key for key, value in corr.items() if abs(value) > 0.95] if corr is not None else []
    flagged_mi = mi_scores[mi_scores > threshold].index.to_list()
    flagged_columns = list(set(flagged_corr + flagged_mi+structural_clones))

    return {
        "target_column": target_column,
        "target_type": "Categorical" if is_categorical(df[target_column]) else "Numerical",
        "corr_scores": corr,
        "mi_scores": mi_scores.to_dict(),
        "structural_clones": structural_clones,
        "flagged_corr": flagged_corr,
        "flagged_mi": flagged_mi,
        "flagged_columns": flagged_columns
    }
