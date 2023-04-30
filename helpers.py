import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway  # for ANOVA
from sklearn.feature_selection import chi2  # for chi2
from sklearn.preprocessing import StandardScaler


def show_distribution(df, col):
    # creating two subplots:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    textb = (
        "Quantiles:"
        + "\n\n"
        + " 25%: "
        + str(round(df[col].quantile(0.25), 2))
        + "\n"
        + " 50%: "
        + str(round(df[col].quantile(0.50), 2))
        + "\n"
        + " 75%: "
        + str(round(df[col].quantile(0.75), 2))
    )

    plt.text(0.8, 0.65, textb, fontsize=12, transform=plt.gcf().transFigure)
    # prepare text:
    text = (
        "Stats:"
        + "\n\n"
        + "Mean: "
        + str(round(df[col].mean(), 2))
        + "\n"
        + "Median: "
        + str(round(df[col].median(), 2))
        + "\n"
        + "Std dev: "
        + str(round(df[col].std(), 2))
        + "\n"
        "Mode: "
        + str((df[col].mode().values)[0])
        + "\n"
        + "Skew: "
        + str(round(df[col].skew(), 2))
        + "\n"
    )
    plt.text(0.35, 0.55, text, fontsize=12, transform=plt.gcf().transFigure)
    plt.axvline(df[col].mean(), color="k", linestyle="dashed", linewidth=1)
    plt.axvline(df[col].median(), color="red", linestyle="dashed", linewidth=1)
    # histplot:
    sns.histplot(x=df[col], ax=axes[0]).set(title="Histogram")
    axes[0].axvline(df[col].mean(), color="k", linestyle="dashed", linewidth=2)
    axes[0].axvline(df[col].median(), color="red", linestyle="dashed", linewidth=2)

    # boxplot:
    sns.boxplot(
        ax=axes[1],
        x=df[col],
        showmeans=True,
        meanline=True,
        meanprops={"color": "white"},
        boxprops={"linewidth": 2},
    ).set(title="Box Plot")
    # main title:
    fig.suptitle(col, fontsize=20, fontweight="bold")


def show_distributions(df):
    continuous_features = df.select_dtypes(include="number").columns
    # create the plot for each columns
    for col in continuous_features:
        show_distribution(df, col)


def calc_anova(df, group_column, values_column):
    # Get list of unique group values in provided column
    unique_group_values = df[group_column].drop_duplicates().to_list()

    # Iterate through each unique group value and filter the dataframe to get the values
    # of the provided column for that group, then store them in a list
    values_by_group = []
    for group_value in unique_group_values:
        group_filter = df[group_column] == group_value
        values_by_group.append(df[values_column][group_filter])

    # Perform ANOVA test on the list of value arrays using the `f_oneway` function from the `scipy.stats` module
    return f_oneway(*values_by_group)


# categorical_columns = df_messy.select_dtypes(include='object').columns # can also include date


def anova_feature_selection(df, target, alpha=0.05):
    # Set significance level
    alpha = 0.05
    # Create a dataframe to store ANOVA results for each numeric column
    df_anova = pd.DataFrame({"F-statistic": [], "p-value": [], "Indicative": []})
    # Create a list to store redundant columns (i.e. those that are not significant)
    redundant_cols = []

    # Loop through each numeric column and calculate ANOVA results
    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:
        # Calculate F-statistic and p-value using calc_anova function
        F, p_value = calc_anova(df, df[target], col)

        # Store ANOVA results in the df_anova dataframe
        df_anova.loc[col, :] = [
            round(F, 3),
            round(p_value, 3),
            "Yes" if p_value < alpha else "No",
        ]

        # If the p-value is greater than the significance level, add the column to the redundant_cols list
        if p_value > alpha:
            redundant_cols.append(col)

    # Sort the df_anova dataframe by the "Indicative" column
    df_anova.sort_values("Indicative", ascending=True)


def chi2_feature_selection(df, target, alpha=0.05):
    df_chisq = pd.DataFrame({"statistic": [], "p-value": []})

    redundant_cols = []

    for col in df.drop(target, axis=1).columns:
        observed = pd.crosstab(df[col], df[target])
        statistic, p_value, dof, expected_freq = stats.chi2_contingency(
            observed=observed
        )
        df_chisq.loc[col, :] = [round(statistic, 3), round(p_value, 3)]
        if p_value > alpha:
            redundant_cols.append(col)
    print(f"Chi-Squared test of independence regarding \n{target}:")
    df_chisq.sort_values("p-value")


def show_hist(col, title=None):
    text = "Stats:" + "\n\n"
    text += "Mean: " + str(round(col.mean(), 2)) + "\n"
    text += "Median: " + str(round(col.median(), 2)) + "\n"
    text += "Mode: " + str(list(col.mode().values)[0]) + "\n"
    text += "Std dev: " + str(round(col.std(), 2)) + "\n"
    text += "Skew: " + str(round(col.skew(), 2)) + "\n"
    bn = round(col.count() ** (1 / 3)) * 2
    col.plot(kind="hist", bins=bn)
    plt.axvline(col.mean(), color="k", linestyle="dashed", linewidth=1)
    plt.axvline(col.median(), color="red", linestyle="dashed", linewidth=1)
    plt.text(0.95, 0.45, text, fontsize=12, transform=plt.gcf().transFigure)
    plt.title(title, fontsize=16, fontweight="bold")


def show_box(col, title=None):
    plt.figure(figsize=(8, 4))
    text = "quantile 25: " + str(round(col.quantile(0.25), 2)) + "\n"
    text += "quantile 50: " + str(round(col.quantile(0.50), 2)) + "\n"
    text += "quantile 75: " + str(round(col.quantile(0.75), 2)) + "\n"
    text += "iqr: " + str(round(col.quantile(0.75) - col.quantile(0.25), 2)) + "\n"
    sns.boxplot(x=col, showmeans=True, meanline=True, meanprops={"color": "white"})

    plt.text(0.65, 0.65, text, fontsize=10, transform=plt.gcf().transFigure)
    plt.title(title, fontsize=16, fontweight="bold")


def show_counts(column1, column2=None, title=None):
    ax = sns.countplot(x=column1, hue=column2)
    for p in ax.patches:
        ax.annotate(
            f"\n{p.get_height()}",
            (p.get_x() + 0.01, p.get_height() - 0.01),
            ha="left",
            va="center",
            color="white",
            size=12,
        )
    plt.title(title, fontsize=16, fontweight="bold")


def show_bar(df):
    ax = df.plot(kind="bar")
    for p in ax.patches:
        ax.annotate(
            np.round(p.get_height(), 2),
            (p.get_x() + 0.15, p.get_height()),
            ha="center",
            va="top",
            color="white",
            size=10,
        )


def get_numeric_details(df):
    res = pd.DataFrame()
    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        data = pd.DataFrame(
            {
                "min": [df[column].min()],
                "quantile 25": df[column].quantile(0.25),
                "quantile 50": df[column].quantile(0.50),
                "quantile 75": df[column].quantile(0.75),
                "max": df[column].max(),
                "median": df[column].median(),
                "mode": ",".join(df[column].astype(str).mode().tolist()),
                "std": df[column].std(),
                "count": df[column].count(),
                "nunique": df[column].nunique(),
                "skew": df[column].skew(),
            },
            index=[column],
        )
        res = res.append(data)
    return res


def cat_count(df, column):
    g1 = (
        df.groupby(column)
        .agg(
            num_of_observations=(column, "size"),
            pct=(column, lambda x: x.count() / len(df)),
        )
        .sort_values("pct", ascending=False)
    )

    g2 = (
        df.agg(
            num_of_observations=(column, "size"),
            pct=(column, lambda x: x.count() / len(df)),
        )
        .transpose()
        .rename(index={column: "Total"})
    )

    res = pd.concat([g1, g2])

    res.reset_index(inplace=True)
    res.rename(columns={"index": column}, inplace=True)

    res = res.assign(
        num_of_observations=res["num_of_observations"].astype(int)
    ).style.format({"pct": "{:.2%}"})
    return res