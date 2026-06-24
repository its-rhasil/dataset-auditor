def detect_outliers(df):
    numerical_cols = df.select_dtypes(include=["number"]).columns
    result = {}
    for col in numerical_cols:
        col_data = df[col].dropna()

        # Calculate quantiles and IQR
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        upper_bound, lower_bound = q3 + (1.5 * iqr) , q1 - (1.5 * iqr)
        iqr_count = len(col_data[(col_data < lower_bound) | (col_data > upper_bound)])
        iqr_pct = (iqr_count/len(col_data)) * 100

        # Calculate Z Score
        if col_data.std() == 0:
            z_score_count = 0
            z_score_pct = 0
        else:
            z_score = (col_data - col_data.mean()) / col_data.std()
            z_score_count = len(z_score[z_score.abs() >= 3])
            z_score_pct = (z_score_count/len(col_data)) * 100
        
        result[col] = {"iqr_count": iqr_count, "iqr_pct": iqr_pct, "z_score_count": z_score_count,"z_score_pct": z_score_pct, "iqr": iqr, "lower_bound": lower_bound, "upper_bound": upper_bound}
    return result

