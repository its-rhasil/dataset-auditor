from analyzer.schemas import DataProfile

def score(profile: DataProfile) -> dict:
    data_quality_score = 100
    missing_score = ((profile.missing["total_missing"] / (profile.rows * profile.columns))**2) * 20
    high_missing_col_score =  ((len(profile.missing["high_missing_cols"]) / profile.columns) ** 2 )* 10
    duplicates_score = (profile.duplicates["duplicated_pct"] / 100) * 20
    outlier_problem_col = 0
    for key, value in profile.outliers.items():
        if value["iqr_pct"]  > 10:
            outlier_problem_col += 1
    outlier_score = ((outlier_problem_col / profile.columns)**2) * 20

    constant_empty_cols = 0
    high_cardinality_cols = 0
    for _, value in profile.cardinality.items():
        if value["is_constant"] or value["nunique"] == 0:
            constant_empty_cols += 1
        elif value["dtype_category"] == "Categorical" and value["unique_ratio"] > 0.8:
            high_cardinality_cols += 1
    cardinality_constempt_score = ((constant_empty_cols/profile.columns) ** 2) * 15
    high_cardinality_score = ((high_cardinality_cols/profile.columns) ** 2) * 15
    data_quality_score = data_quality_score - missing_score - high_missing_col_score - duplicates_score - outlier_score - cardinality_constempt_score - high_cardinality_score
    data_quality_score = max(0,data_quality_score) # safety floor: prevents negative scores if detector outputs exceed expected ranges
    
    return {"Data Quality": data_quality_score}