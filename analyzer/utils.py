import pandas as pd
def is_categorical(column:pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(column) or column.nunique() <= 10