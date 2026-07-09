import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from analyzer.utils import is_categorical

def leakage(df, target_column):
    df1 = df.dropna().copy()
    target = df1[target_column]
    corr = df1.select_dtypes(include="number").corr()[target_column].drop(target_column).to_dict() if not is_categorical(target) else None

    encoder = OrdinalEncoder()
    x = df1.drop(target_column, axis = 1)
    cat_columns = x.select_dtypes(exclude='number').columns
    if len(cat_columns)> 0:
        x[cat_columns] = encoder.fit_transform(x[cat_columns])
    mi_scores = mutual_info_classif(x, target) if is_categorical(target) else mutual_info_regression(x, target)
    mi_scores = pd.Series(mi_scores, index =x.columns)
    median = np.median(mi_scores)
    mad = np.median(np.abs(mi_scores - median))
    threshold = median + (2.5 * mad)

    structural_clones = []
    for col in x.columns:
        if x[col].nunique() == target.nunique():
            if df1.groupby(col)[target_column].nunique().max() == 1:
                structural_clones.append(col)

    flagged_corr = [key for key, value in corr.items() if value > 0.95] if corr is not None else []
    flagged_mi = mi_scores[mi_scores > threshold].index.to_list()
    flagged_columns = list(dict.fromkeys(flagged_corr + flagged_mi))
    flagged_columns.extend(structural_clones)
    return {
        "target_column": target_column,
        "target_type": "Categorical" if is_categorical(target) else "Numerical",
        "corr_scores": corr,
        "mi_scores": mi_scores.to_dict(),
        "structural_clones": structural_clones,
        "flagged_corr": [key for key,value in corr.items() if value > 0.95] if corr is not None else None,
        "flagged_mi": mi_scores[mi_scores > threshold].index.to_list(),
        "flagged_columns": flagged_columns
    }