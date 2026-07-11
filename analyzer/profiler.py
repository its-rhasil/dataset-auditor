from analyzer.schemas import DataProfile
from analyzer.detectors.cardinality import cardinality
from analyzer.detectors.duplicates import detect_duplicates
from analyzer.detectors.imbalance import imbalance
from analyzer.detectors.leakage import leakage
from analyzer.detectors.missing import detect_missing
from analyzer.detectors.outliers import detect_outliers
import pandas as pd

def profile(filename: str, df: pd.DataFrame, target_column: str = None) -> DataProfile:
    
    imbalance_dict = imbalance(df,target_column) if target_column else {}
    leakage_dict = leakage(df, target_column) if target_column else {}

    return DataProfile(
        rows = df.shape[0],
        columns= df.shape[1],
        filename= filename,
        missing= detect_missing(df),
        duplicates= detect_duplicates(df),
        outliers= detect_outliers(df),
        cardinality= cardinality(df),
        imbalance= imbalance_dict,
        leakage= leakage_dict
                       )