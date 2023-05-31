import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway  # for ANOVA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2  # for chi2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def elbow_method(iters_num, data_set, k):
    from sklearn.cluster import KMeans

    iters_num = 10
    clustering_scores = []
    for i in range(1, iters_num):  # range of numbers for clusters
        # print('cluster numbers:', i)
        kmeans = KMeans(  # This is where we create our model
            n_clusters=i,  # number of clusters
            n_init="auto",
            random_state=42,
        )
        kmeans.fit(data_set)  # Training the model
        # print('cluster\'s Interia score: ', kmeans.inertia_)
        clustering_scores.append(
            kmeans.inertia_
        )  # inertia_ = Sum of squared distances of samples to their closest cluster center.

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, iters_num), clustering_scores, marker=".")
    plt.scatter(
        k, clustering_scores[k - 1], s=300, c="red", marker="*"
    )  # and now you know how to add a star to a plot :)
    plt.title("The Elbow Method")
    plt.xlabel("No. of Clusters")
    plt.ylabel("Clustering Score (Inertia)")
    plt.show()

def calc_corr(df, target):
    res = pd.DataFrame()
    features = list(df.select_dtypes(include="number").columns)
    features.remove(target)

    for feature in features:
        corr = stats.pearsonr(df[target], df[feature])
        data = pd.DataFrame({f"Correlation with `{str(target).capitalize()}`": round(corr[0], 4),"p-value": round(corr[1], 4),},index=[feature],)
        res = pd.concat([res, data])
    return res.sort_values(f"Correlation with `{str(target).capitalize()}`", ascending=False)

def find_high_correlations(df, threshold):
    corr_mat = df.corr().unstack().sort_values(kind="quicksort").drop_duplicates()
    high_corr = corr_mat[abs(corr_mat) >= threshold].reset_index()
    high_corr.columns = ["feature_1", "feature_2", "correlation"]
    high_corr = high_corr[high_corr["feature_1"] != high_corr["feature_2"]]
    return list(high_corr[["feature_1", "feature_2"]].to_records(index=False))


def viz3D(clusters):
    import plotly as py  # import plotly library
    import plotly.graph_objs as go  # import graph objects as go

    trace1 = go.Scatter3d(  # create trace for scatterplot
        x=df["daily_ammount_sold"],  # x-axis will be Age column
        y=df["shippment_duration"],  # y-axis will be Spending Score
        z=df["price"],  # z-axis will be Annual Income
        mode="markers",
        marker=dict(
            color=z_kmeans,
            size=10,
            line=dict(color=z_kmeans, width=12),  # color points by clusters
            opacity=0.8,
        ),
    )

    data = [trace1]  # create data list with trace1

    layout = go.Layout(  # create layout for scatterplot
        title="Clusters by daily_ammount_sold, shippment_duration, and price",  # Set Graph Title
        scene=dict(
            xaxis=dict(title="daily_ammount_sold"),
            yaxis=dict(title="shippment_duration"),
            zaxis=dict(title="price"),
        ),  # Axis titles
    )
    # create a figure with data and layout that we just defined
    fig = go.Figure(data=data, layout=layout)

    # plot the figure using plotly's offline mode
    py.offline.iplot(fig)


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
        F, p_value = calc_anova(df, target, col)

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
    return df_chisq.sort_values("p-value"), redundant_cols


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
