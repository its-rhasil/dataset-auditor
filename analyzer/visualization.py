import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from analyzer.profiler import profile
import seaborn as sns
import missingno as msno

df = sns.load_dataset("titanic")
test = profile("titanic",df,"survived")

def plot_missing_bar(missing_dict, threshold=1.0):
    
    pairs = [(col, stat["pct"]) for col, stat in missing_dict["columns"].items() if stat["pct"] > 1.0]

    df = pd.DataFrame(pairs, columns=["Column","Pct"])
    df = df.sort_values("Pct",ascending=False)
    fig = px.bar(df,x="Pct",y="Column",orientation='h',color="Pct",color_continuous_scale="YlOrRd")
    fig.update_layout(title="Missing Values by Column", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",width =1000, height=400)
    fig.update_traces(text=df["Pct"].round(1).astype(str) + "%", textposition="outside")
    return fig


def plot_missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    msno.heatmap(
        df,
        ax=ax,
        fontsize=10,
        cmap="RdBu"
    )

    ax.set_title(
        "Missing Value Correlation Matrix",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()
    #plt.close()
    return fig

def plot_outlier_bar(outlier_dict: dict, method: str = "iqr"):
    outliers = pd.DataFrame(outlier_dict).T
    pct_col = f"{method}_pct"

    plot_df = (
        outliers[[pct_col, f"{method}_count"]]
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values(by=pct_col, ascending=False)
    )

    fig = px.bar(
        plot_df,
        x=pct_col,
        y="column",
        orientation="h",
        color=pct_col,
        color_continuous_scale="YlOrRd",
        custom_data = [f"{method}_count"]
    )

    fig.update_layout(
        title=f"Outlier Percentage by Column ({method.upper()})",
        xaxis_title="Outlier Percentage",
        yaxis_title="Column",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.update_traces(
        texttemplate="%{x:.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
        "<b>%{y}</b><br>"
        "Outliers: %{customdata[0]}<br>"
        "Percentage: %{x:.2f}%"
        "<extra></extra>"
    )
    )
    return fig

def plot_boxplot(df: pd.DataFrame, column: str):
    plot_df = df[[column]].dropna()

    fig = px.box(
        plot_df,
        y=column,
        points="outliers"
    )

    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title="",
        yaxis_title=column,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.25)",
            gridwidth=1
        )
    )
    fig.update_traces(
    fillcolor="rgba(255, 243, 179, 0.55)",  
    line=dict(color="#FFC107", width=2),    
    marker=dict(
        color="#D32F2F",                     
        size=7
    )
    )

    return fig

def get_status(row):
    if row["is_constant"]:
        return "Constant"
    elif row["nunique"] == 0:
        return "Empty"
    elif row["unique_ratio"] > 0.8:
        return "High Cardinality"
    else:
        return "Normal"

def plot_cardinality_summary(cardinality_dict):
    df = pd.DataFrame(cardinality_dict).T
    df = df[df["dtype_category"] != "Numerical"].drop(["cv","dtype_category"],axis=1)
    df["status"] = df.apply(get_status, axis = 1)
    df["top_value_freq"] = (df["top_value_freq"] * 100).round(2).astype(str) + "%"
    return df


def plot_imbalance_bar(imbalance_dict):
    df = pd.DataFrame({
        "Class": imbalance_dict["class_counts"].keys(),
        "Count": imbalance_dict["class_counts"].values(),
        "Percentage": imbalance_dict["class_pct"].values()
    })
    df["Class"] = df["Class"].astype(str)
    fig = px.bar(
        df,
        y = "Class",
        x = "Percentage",
        text= "Percentage",
        color="Class",
        orientation='h',
        custom_data=["Count"]
    )

    fig.update_traces(
    hovertemplate=(
        "<b>Class %{y}</b><br>"
        "Percentage: %{x:.2f}%<br>"
        "Count: %{customdata[0]}"
        "<extra></extra>"
    ),
      marker_color="#FD8D3C",
    texttemplate="%{x:.2f}%",
    textposition="outside"
)

    fig.update_layout(
        title="Class Distribution",
        yaxis_title="Class",
        xaxis_title="Count",
        showlegend=False
    )

    return fig

def plot_mi_scores(leakage_dict):
    df = (
    pd.Series(leakage_dict["mi_scores"])
    .sort_values(ascending=False)
    .reset_index()
    )
    df.columns = ["Feature", "MI Score"]

    df["Flagged"] = df["Feature"].isin(leakage_dict["flagged_columns"])
    fig = px.bar(
        df,
        y="Feature",
        x="MI Score",
        color="Flagged",
        orientation="h",
        text="MI Score",
        color_discrete_map={
        False: "#FD8D3C",
        True: "#CB181D"
        }
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "MI Score: %{x:.3f}<br>"
            "Flagged: %{customdata[0]}"
            "<extra></extra>"
        ),
        customdata=df[["Flagged"]],
        texttemplate="%{x:.3f}",
        textposition="outside"
    )
    fig.update_layout(
        title="Mutual Information Scores",
        xaxis_title="Mutual Information",
        yaxis_title="Feature",
        yaxis={
            'categoryorder': 'array',
            'categoryarray': df["Feature"].tolist()
        },
        showlegend=True
    )
    return fig

def plot_corr_heatmap(leakage_dict):
    corr_scores = leakage_dict["corr_scores"]

    if corr_scores is None:
        return None

    df = pd.DataFrame.from_dict(
        corr_scores,
        orient="index",
        columns=["Correlation"]
    )

    fig = px.imshow(
        df,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    fig.update_layout(
        title="Feature vs Target Correlation",
        xaxis_title=leakage_dict["target_column"],
        yaxis_title="Feature"
    )

    return fig