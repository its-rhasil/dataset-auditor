from analyzer.schemas import DataProfile

def score(profile: DataProfile) -> dict:
    data_quality_score = 100
    ml_readiness_score = None
    missing_score = ((profile.missing["total_missing"] / (profile.rows * profile.columns))**2) * 30
    duplicates_score = (profile.duplicates["duplicated_pct"] / 100) * 20
    outlier_problem_col = 0
    for key, value in profile.outliers.items():
        if value["iqr_pct"]  > 10:
            outlier_problem_col += 1
    outlier_score = ((outlier_problem_col / profile.columns)**2) * 25

    cardinality_problem_col = 0
    for key, value in profile.cardinality.items():
        if value["is_constant"] or value["nunique"] == 0:
            cardinality_problem_col += 1
    cardinality_score = ((cardinality_problem_col/profile.columns) ** 2) * 25

    data_quality_score = data_quality_score - missing_score - duplicates_score - outlier_score - cardinality_score
    data_quality_score = max(0,data_quality_score) # safety floor: prevents negative scores if detector outputs exceed expected ranges

    if profile.imbalance:
        ml_readiness_score = 100
        missing_ml_score = ((profile.missing["total_missing"] / (profile.rows * profile.columns))**2) * 25
        target_missing_ml_score = ((profile.imbalance["missing_count"]/profile.rows) ** 2) * 20
        cardinality_ml_score = ((cardinality_problem_col/profile.columns) ** 2) * 20
        imbalance_ratio = profile.imbalance["imbalance_ratio"]
        if imbalance_ratio <= 1.5:
            imbalance_score = 0
        elif imbalance_ratio > 1.5 and imbalance_ratio <= 3:
            imbalance_score = 1
        elif imbalance_ratio > 3 and imbalance_ratio <= 10:
            imbalance_score = 2
        elif imbalance_ratio > 10 and imbalance_ratio <= 50:
            imbalance_score = 3
        else:
            imbalance_score  = 4
        imbalance_ml_score = (imbalance_score/4) * 20
        leakage_ml_score = (len(profile.leakage["structural_clones"])/profile.columns) * 15
        ml_readiness_score = ml_readiness_score - missing_ml_score - target_missing_ml_score - imbalance_ml_score - cardinality_ml_score - leakage_ml_score
        ml_readiness_score = max(0, ml_readiness_score) # safety floor: prevents negative scores if detector outputs exceed expected ranges

    
    return {"Data Quality": data_quality_score, "ML Readiness": ml_readiness_score}