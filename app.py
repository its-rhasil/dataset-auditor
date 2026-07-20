import streamlit as st
import pandas as pd
import time 
from analyzer.utils import is_categorical
import traceback
from analyzer.profiler import profile
from analyzer.scorer import score
from analyzer.visualization import (
    plot_missing_bar,
    plot_missing_heatmap,
    plot_outlier_bar,
    plot_boxplot,
    plot_cardinality_summary,
    plot_imbalance_bar,
    plot_mi_scores,
    plot_corr_heatmap,
)


# Page Configuration


st.set_page_config(
    page_title="Dataset Auditor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "df" not in st.session_state:
    st.session_state.df = None

if "data_profile" not in st.session_state:
    st.session_state.data_profile = None

if "scores" not in st.session_state:
    st.session_state.scores = None

if "target_column" not in st.session_state:
    st.session_state.target_column = None

# Custom CSS


st.markdown(
    """
    <style>

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        .app-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            font-size: 1rem;
            color: #888888;
            margin-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 1rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
        }

        .section-description {
            color: #888888;
            margin-bottom: 1rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)



# Helper Functions


def load_dataset(uploaded_file):
    """
    Load CSV or XLSX dataset.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format.")


def format_memory(memory_bytes):
    """
    Convert memory size in bytes to a human-readable format.
    """

    if memory_bytes < 1024:
        return f"{memory_bytes} B"

    elif memory_bytes < 1024 ** 2:
        return f"{memory_bytes / 1024:.2f} KB"

    elif memory_bytes < 1024 ** 3:
        return f"{memory_bytes / (1024 ** 2):.2f} MB"

    else:
        return f"{memory_bytes / (1024 ** 3):.2f} GB"




st.markdown(
    '<div class="app-title">Dataset Auditor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
        Analyze dataset quality, missing values, outliers,
        cardinality, class imbalance, and potential data leakage.
    </div>
    """,
    unsafe_allow_html=True,
)



st.header("Dataset Input")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx"],
    help="Supported file formats: CSV and XLSX",
)

target_input = st.text_input(
    "Target Column (Optional)",
    placeholder="Enter the target column name",
    help=(
        "Provide the target column to enable "
        "class imbalance and data leakage analysis."
    ),
)

analyze_button = st.button(
    "Analyze",
    type="primary",
)



if analyze_button:
    
    if uploaded_file is None:

        st.warning(
            "Please upload a dataset before running the analysis."
        )

    else:

        try:
            df = load_dataset(uploaded_file)

            if df.empty:
                st.warning(
                    "The uploaded dataset is empty."
                )
                st.stop()


            target_column = target_input.strip()

            if target_column:

                if target_column not in df.columns:

                    st.warning(
                        f"Target column '{target_column}' "
                        "was not found in the dataset. "
                        "Target-dependent analysis will be skipped."
                    )

                    target_column = None

            else:

                target_column = None



            with st.spinner("Analyzing dataset..."):
                start = time.time()
                data_profile = profile(
                    uploaded_file.name,
                    df,
                    target_column,
                )

                scores = score(
                    data_profile
                )
                end = time.time()
                print(f"Analysis took {end - start:.2f} seconds")

            st.session_state.df = df

            st.session_state.data_profile = (
                data_profile
            )

            st.session_state.scores = scores

            st.session_state.target_column = (
                target_column
            )

            st.session_state.analysis_complete = True


            st.success(
                "Dataset analysis completed successfully."
            )


        except Exception as e:

            st.session_state.analysis_complete = False

            st.error(
                f"Dataset analysis failed: {e}"
            )
            st.error(traceback.format_exc())


# Display Analysis



if st.session_state.analysis_complete:


    df = st.session_state.df

    data_profile = (
        st.session_state.data_profile
    )

    scores = (
        st.session_state.scores
    )

    target_column = (
        st.session_state.target_column
    )

    # Dataset Overview


    st.divider()

    st.header(
        "Dataset Overview"
    )

    st.markdown(
        """
        <div class="section-description">
            General information and structural properties
            of the uploaded dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    memory_usage = (
        df.memory_usage(deep=True).sum()
    )

    numerical_columns = [
    col
    for col in df.columns
    if not is_categorical(df[col])
    ]

    categorical_columns = [
    col
    for col in df.columns
    if is_categorical(df[col])
    ]

    total_missing = (
        df.isna().sum().sum()
    )

    duplicate_rows = (
        df.duplicated().sum()
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            label="Filename",
            value=data_profile.filename,
        )


    with col2:

        st.metric(
            label="Rows",
            value=f"{data_profile.rows:,}",
        )


    with col3:

        st.metric(
            label="Columns",
            value=f"{data_profile.columns:,}",
        )


    with col4:

        st.metric(
            label="Memory Usage",
            value=format_memory(
                memory_usage
            ),
        )



    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            label="Numerical Columns",
            value=len(
                numerical_columns
            ),
        )


    with col2:

        st.metric(
            label="Categorical Columns",
            value=len(
                categorical_columns
            ),
        )


    with col3:

        st.metric(
            label="Missing Values",
            value=f"{total_missing:,}",
        )


    with col4:

        st.metric(
            label="Duplicate Rows",
            value=f"{duplicate_rows:,}",
        )


   
    # Audit Scores


    st.subheader(
        "Audit Scores"
    )


    score_col1, score_col2 = (
        st.columns(2)
    )



    # Data Quality Score


    data_quality_score = (
        scores["Data Quality"]
    )


    with score_col1:

        st.metric(
            label="Data Quality Score",
            value=f"{data_quality_score:.1f} / 100",
        )

        st.progress(
            min(
                max(
                    data_quality_score / 100,
                    0.0,
                ),
                1.0,
            )
        )


    # ML Readiness Score


    # ml_readiness_score = (
    #     scores["ML Readiness"]
    # )


    # with score_col2:

    #     if ml_readiness_score is not None:

    #         st.metric(
    #             label="ML Readiness Score",
    #             value=f"{ml_readiness_score:.1f} / 100",
    #         )

    #         st.progress(
    #             min(
    #                 max(
    #                     ml_readiness_score / 100,
    #                     0.0,
    #                 ),
    #                 1.0,
    #             )
    #         )

    #     else:

    #         st.metric(
    #             label="ML Readiness Score",
    #             value="N/A",
    #         )

    #         st.caption(
    #             "Provide a valid target column "
    #             "to calculate ML readiness."
    #         )



    # Missing Value Analysis

    preview_df = df.head().copy()
    bool_cols = preview_df.select_dtypes(include='bool').columns
    preview_df[bool_cols] = preview_df[bool_cols].astype(str)
    st.dataframe(preview_df, width='stretch', hide_index=True)
    st.divider()

    st.header(
        "Missing Value Analysis"
    )

    st.caption(
        "Identify features containing missing data "
        "and examine relationships between "
        "missingness patterns."
    )



    # Find Columns with Significant Missing Values


    missing_columns = [

        col

        for col, stats
        in data_profile.missing[
            "columns"
        ].items()

        if stats["pct"] > 1.0
    ]


    # Missing Value Bar Chart


    if missing_columns:

        missing_fig = (
            plot_missing_bar(
                data_profile.missing
            )
        )

        st.plotly_chart(
            missing_fig,
            width = 'stretch',
        )

    else:

        st.success(
            "No columns contain more "
            "than 1% missing values."
        )


    # Advanced Missing Analysis


    with st.expander(
        "Advanced Missing Analysis",
        expanded=False,
    ):

        st.subheader(
            "Missing Value Correlation Matrix"
        )

        st.caption(
            "Shows correlations between "
            "missingness patterns across columns."
        )


        columns_with_missing = (

            df.columns[
                df.isna().any()
            ]
            .tolist()

        )


        if len(columns_with_missing) >= 2:

            try:

                heatmap_fig = (
                    plot_missing_heatmap(
                        df
                    )
                )

                st.pyplot(
                    heatmap_fig,
                    width = 'stretch',
                )

            except Exception as e:

                st.warning(
                    "Unable to generate the "
                    "missing value correlation "
                    f"matrix: {e}"
                )

        else:

            st.info(
                "At least two columns containing "
                "missing values are required to "
                "calculate missingness correlations."
            )

    # Outlier Analysis


    st.divider()

    st.header(
        "Outlier Analysis"
    )

    st.caption(
        "Analyze potential outliers detected "
        "in numerical features using the IQR method."
    )



    # Outlier Bar Chart

    if data_profile.outliers:

        outlier_fig = (
            plot_outlier_bar(
                data_profile.outliers,
                method="iqr",
            )
        )

        st.plotly_chart(
            outlier_fig,
            width = 'stretch',
        )

    else:

        st.info(
            "No numerical columns are "
            "available for outlier analysis."
        )


    # Advanced Outlier Analysis


    with st.expander(
        "Advanced Outlier Analysis",
        expanded=False,
    ):

        st.subheader(
            "Feature Distribution"
        )


        outlier_columns = list(
            data_profile.outliers.keys()
        )


        if outlier_columns:

            selected_column = (
                st.selectbox(
                    "Select a numerical column",
                    options=outlier_columns,
                    key="outlier_column",
                )
            )


            boxplot_fig = (
                plot_boxplot(
                    df,
                    selected_column,
                )
            )


            st.plotly_chart(
                boxplot_fig,
                width = 'stretch',
            )


        else:

            st.info(
                "No numerical columns are "
                "available for advanced "
                "outlier analysis."
            )


    # Cardinality Analysis


    st.divider()

    st.header(
        "Cardinality Analysis"
    )

    st.caption(
        "Review the uniqueness and value "
        "distribution of categorical features."
    )



    # Create Cardinality Summary


    try:

        cardinality_df = (
            plot_cardinality_summary(
                data_profile.cardinality
            )
        )


        if not cardinality_df.empty:

            cardinality_df = (

                cardinality_df

                .reset_index()

                .rename(
                    columns={
                        "index": "Column",
                        "nunique": "Unique Values",
                        "unique_ratio": "Unique Ratio",
                        "is_constant": "Constant",
                        "top_value": "Top Value",
                        "top_value_freq": "Top Value Frequency",
                        "status": "Status",
                    }
                )

            )

            cardinality_df["Top Value"] = cardinality_df["Top Value"].astype(str)


            cardinality_df[
                "Constant"
            ] = (

                cardinality_df[
                    "Constant"
                ]

                .map(
                    {
                        True: "Yes",
                        False: "No",
                    }
                )

            )

            def highlight_cardinality(
                row
            ):

                status = (
                    row["Status"]
                )


                if status in [
                    "Constant",
                    "Empty",
                ]:

                    return [

                        "background-color: "
                        "rgba(203, 24, 29, 0.20)"

                    ] * len(row)


                elif (
                    status
                    == "High Cardinality"
                ):

                    return [

                        "background-color: "
                        "rgba(253, 141, 60, 0.20)"

                    ] * len(row)


                return [
                    ""
                ] * len(row)

            styled_cardinality = (

                cardinality_df.style

                .apply(
                    highlight_cardinality,
                    axis=1,
                )

            )

            st.dataframe(
                styled_cardinality,
                width = 'stretch',
                hide_index=True,
            )


            st.caption(
                "Highlighted rows indicate potentially "
                "problematic constant, empty, or "
                "high-cardinality columns."
            )


        else:

            st.info(
                "No categorical columns are "
                "available for cardinality analysis."
            )


    except Exception as e:

        st.warning(
            "Unable to display cardinality "
            f"analysis: {e}"
        )


    # Target Class Imbalance

    if target_column:
        # Target Class Imbalance Analysis

        

        




        if data_profile.imbalance:

            st.divider()

            st.header(
                "Target Class Imbalance Analysis"
            )

            st.caption(
                f"Analyze the class distribution "
                f"of the target feature "
                f"'{target_column}'."
            )
            # Imbalance Chart

            imbalance_fig = (
                plot_imbalance_bar(
                    data_profile.imbalance
                )
            )


            st.plotly_chart(
                imbalance_fig,
                width = 'stretch',
            )


            # Imbalance Metrics

            col1, col2, col3 = (
                st.columns(3)
            )


            with col1:

                st.metric(
                    label="Number of Classes",
                    value=(
                        data_profile
                        .imbalance[
                            "num_classes"
                        ]
                    ),
                )


            with col2:

                st.metric(
                    label="Minority Class",
                    value=str(
                        data_profile
                        .imbalance[
                            "minority_class"
                        ]
                    ),
                )


            with col3:

                st.metric(
                    label="Imbalance Ratio",
                    value=(
                        f"{data_profile.imbalance['imbalance_ratio']:.2f}"
                    ),
                )

    # Data Leakage Analysis

        st.divider()

        st.header(
            "Data Leakage Analysis"
        )

        st.caption(
            "Analyze statistical relationships "
            "between features and the target to "
            "identify potentially suspicious features."
        )


        if data_profile.leakage:


            # ---------------------------------------------
            # Mutual Information Scores
            # ---------------------------------------------

            mi_scores = (
                data_profile.leakage.get(
                    "mi_scores"
                )
            )


            if mi_scores:

                mi_fig = (
                    plot_mi_scores(
                        data_profile.leakage
                    )
                )


                st.plotly_chart(
                    mi_fig,
                    width = 'stretch',
                )


            # Flagged Features

            st.subheader(
                "Potentially Suspicious Features"
            )


            flagged_columns = (

                data_profile
                .leakage
                .get(
                    "flagged_columns",
                    [],
                )

            )


            if flagged_columns:

                st.warning(
                    "Flagged columns: "
                    + ", ".join(
                        map(
                            str,
                            flagged_columns,
                        )
                    )
                )


            else:

                st.success(
                    "No features were flagged "
                    "as potentially suspicious "
                    "for data leakage."
                )

            # Feature vs Target Correlation
            #
            # plot_corr_heatmap returns None
            # when corr_scores is None.

            corr_fig = (
                plot_corr_heatmap(
                    data_profile.leakage
                )
            )


            if corr_fig is not None:

                st.subheader(
                    "Feature vs Target Correlation"
                )


                st.plotly_chart(
                    corr_fig,
                    width = 'stretch',
                )

            
            # Leakage Disclaimer

            st.info(
                """
                These columns have been flagged as potentially
                suspicious for data leakage based on statistical
                analysis. This is not a final verdict. High
                correlation or mutual information does not
                necessarily indicate data leakage. The flagged
                features should be reviewed based on the dataset's
                context and how the features were created.
                """
            )


    # End of Report
    st.divider()

    st.caption(
        "End of Dataset Audit Report"
    )