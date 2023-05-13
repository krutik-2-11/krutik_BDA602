import math
import os
import sys

import globals as gb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api
from baseball_data_loader import BaseballDataLoader
from brute_force_mean_of_response import BruteForceMeanOfResponse
from correlation_metrics import CorrelationMetrics

# from data_loader import TestDatasets
from mean_of_response import MeanOfResponse
from predictor_vs_response_plots import PredictorVsResponsePlots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

"""
References:
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
"""


def load_dataset():
    """
    Function to load the dataset with the user prompt.
    Returns dataset name, dataframe, predictors list and response
    :return: dataset, df, predictors, response
    """

    """
    test_datasets = TestDatasets()
    dataset_dict = {}
    for test in test_datasets.get_all_available_datasets():
        df, predictors, response = test_datasets.get_test_data_set(data_set_name=test)
        dataset_dict[test] = [df, predictors, response]

    dataset_list = list(dataset_dict.keys())
    print("Available Datasets: ")
    for dataset in dataset_list:
        print(dataset)

    dataset = input("Enter your dataset: ").strip().lower()

    df = dataset_dict[dataset][0]
    predictors = dataset_dict[dataset][1]
    response = dataset_dict[dataset][2]
    """

    # Loading the baseball dataset
    baseball = BaseballDataLoader()
    dataset, df, predictors, response = baseball.get_baseball_data()

    return dataset, df, predictors, response


def create_resp_pred_plot_folder(dataset):
    """
    Function to create a folder for response predictor plots
    :param dataset: Dataset text string
    :return: path string
    """
    path = f"{gb.PATH_RESP_PRED}"
    if not os.path.exists(path):
        os.mkdir(path)
    path = f"{path}/{dataset}"
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def create_mean_of_response_plot_folder(dataset):
    """
    Function to create a folder for mean of response plots
    :param dataset: Dataset text string
    :return: path string
    """
    path = f"{gb.PATH_MEAN_OF_RESPONSE}"
    if not os.path.exists(path):
        os.mkdir(path)
    path = f"{path}/{dataset}"
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def create_brute_force_plot_folder(dataset):
    """
    Function to create a folder for brute force plots
    :param dataset: Dataset text string
    :return: path string
    """
    path = f"{gb.PATH_BRUTE_FORCE_PLOTS}"
    if not os.path.exists(path):
        os.mkdir(path)
    path = f"{path}/{dataset}"
    if not os.path.exists(path):
        os.mkdir(path)
    return path


# Define function to format local plot image as clickable link
def make_clickable(path):
    """
    Function to make the url clickable
    :param path: Path string
    :return: HTML link for path
    """
    if path:
        if "," in path:
            x = path.split(",")
            return f'{x[0]} <a target="_blank" href="{x[1]}">Plot</a>'
        else:
            return f'<a target="_blank" href="{path}">Plot</a>'


def save_dataframe_to_HTML(df, plot_link_mor, plot_link, caption):
    """
    Function to convert dataframe with 2 links into HTML
    :param df: Pandas Dataframe
    :param plot_link_mor: Mean of Response plot url
    :param plot_link: Predictor vs Response plot url
    :param caption: Caption String of the table
    :return: None
    """
    # Apply styles to the DataFrame using the Styler class
    styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {
            "selector": "th, td",
            "props": [("padding", "8px"), ("border", "1px solid black")],
        },
        {"selector": "th", "props": [("background-color", "#f2f2f2")]},
    ]
    styled_table = (
        df.style.format(
            {f"{plot_link_mor}": make_clickable, f"{plot_link}": make_clickable}
        )
        .set_table_styles(styles)
        .set_caption(
            f"<h1 style='text-align:center; font-size:30px; font-weight:bold;'>{caption}</h1>"
        )
    )

    # Generate an HTML table from the styled DataFrame
    html_table = styled_table.to_html()

    with open(gb.HTML_FILE, "a") as f:
        f.write(html_table)


def save_brute_force_dataframe_to_HTML(df, link, caption):
    """
    Function to convert dataframe with 1 link into HTML
    :param df: Pandas dataframe
    :param link: Link for the brute force heat plot
    :param caption: Caption for the brute force table
    :return: None
    """
    # Apply styles to the DataFrame using the Styler class
    styles = [
        {"selector": "table", "props": [("border-collapse", "collapse")]},
        {
            "selector": "th, td",
            "props": [("padding", "8px"), ("border", "1px solid black")],
        },
        {"selector": "th", "props": [("background-color", "#f2f2f2")]},
    ]
    styled_table = (
        df.style.format({f"{link}": make_clickable})
        .set_table_styles(styles)
        .set_caption(
            f"<h1 style='text-align:center; font-size:30px; font-weight:bold;'>{caption}</h1>"
        )
    )

    # Generate an HTML table from the styled DataFrame
    html_table = styled_table.to_html()

    with open(gb.HTML_FILE, "a") as f:
        f.write(html_table)


def return_predictor_type(column):
    """
    Function to categorize the predictor into categorical or continuous type
    :param column: Pandas Series
    :return: string "categorical" or "continuous"
    """
    if column.dtype in ["object", "bool"]:
        return gb.CATEGORICAL_TYPE_PRED
    else:
        return gb.CONTINUOUS_TYPE_PRED


def return_response_type(column):
    """
    Function to categorize the response into boolean or continuous type
    :param column: Pandas Series
    :return: string "continuous" or "boolean"
    """
    if len(column.unique()) > 2:
        return gb.CONTINUOUS_TYPE_RESP
    else:
        return gb.BOOLEAN_TYPE_RESP


def return_categorical_continuous_predictor_list(df, predictors):
    """
    Function to return the list of categorical and continuous predictors
    :param df: Pandas Dataframe
    :param predictors: List of Predictors
    :return: categorical_predictors_list, continuous_predictors_list
    """
    categorical_predictors_list = []
    continuous_predictors_list = []
    for predictor in predictors:
        predictor_type = return_predictor_type(df[predictor])
        if predictor_type == gb.CATEGORICAL_TYPE_PRED:
            categorical_predictors_list.append(predictor)
        elif predictor_type == gb.CONTINUOUS_TYPE_PRED:
            continuous_predictors_list.append(predictor)

    return categorical_predictors_list, continuous_predictors_list


def random_forest_variable_importance_ranking(df, continuous_predictors, response):
    """
    Function to calculate the random forest variable importance for continuous predictors
    :param df: Pandas Dataframe
    :param continuous_predictors: Continuous Predictors List
    :param response: Response String
    :return: Feature Importance dataframe
    """

    X = df[continuous_predictors]
    y = df[response]

    # create a random forest regressor
    rf = RandomForestRegressor(random_state=42)

    # Fit the model into the data
    rf.fit(X, y)

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame(
        {gb.PREDICTOR: continuous_predictors, gb.IMPORTANCE: importances}
    )

    # Sort the dataframe by feature importance in descending order
    feature_importances = feature_importances.sort_values(
        gb.IMPORTANCE, ascending=False
    ).reset_index(drop=True)

    return feature_importances


def create_linear_regression(df, predictor, response):
    """
    Function to generate p-value, t-value for a predictors with continuous response
    :param df: Pandas Dataframe
    :param predictor: Predictor name string
    :param response: Response name string
    :return: dictionary with predictor name, p-value, t-value
    """
    summary_dict = {}
    # numpy array for predictor column
    X = df[predictor]

    # numpy array for response column
    y = df[response]

    predict = statsmodels.api.add_constant(X)
    linear_regression_model = statsmodels.api.OLS(y, predict)
    linear_regression_model_fitted = linear_regression_model.fit()

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    summary_dict[gb.PREDICTOR] = predictor
    summary_dict[gb.P_VALUE] = p_value
    summary_dict[gb.T_VALUE] = t_value

    return summary_dict


def create_logistic_regression(df, predictor, response):
    """
    Function to generate p-value, t-value for a predictors with categorical response
    :param df: Pandas Dataframe
    :param predictor: Predictor name string
    :param response: Response name string
    :return: dictionary with predictor name, p-value, t-value
    """
    summary_dict = {}

    # numpy array for predictor column
    X = df[predictor]

    # numpy array for response column
    y = df[response]

    predict = statsmodels.api.add_constant(X)
    logistic_regression_model = statsmodels.api.Logit(y, predict)
    logistic_regression_model_fitted = logistic_regression_model.fit()

    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    summary_dict[gb.PREDICTOR] = predictor
    summary_dict[gb.P_VALUE] = p_value
    summary_dict[gb.T_VALUE] = t_value

    return summary_dict


def correlation_cont_cont_table(cm, df, cont_preds, df_mor_continuous):
    """
    Function to generate correlation matrix plot and table for cont/cont predictors
    :param cm: correlation metrics object
    :param df: Pandas Dataframe
    :param cont_preds: Continuous Predictors list
    :param df_mor_continuous: Mean of Response dataframe for continuous predictors
    :return: df_corr_cont_cont_table: Continuous/ Continuous correlation table dataframe
    """
    lst_corr_cont_cont_table = []
    for pred_1 in cont_preds:
        for pred_2 in cont_preds:
            dict_corr_cont_cont_table = {
                gb.CONT_1: pred_1,
                gb.CONT_2: pred_2,
                gb.CORR: cm.cont_cont_correlation(df[pred_1], df[pred_2]),
                gb.CONT_1_URL: df_mor_continuous.loc[
                    df_mor_continuous[gb.PREDICTOR] == pred_1, gb.PLOT_LINK_MOR
                ].values[0],
                gb.CONT_2_URL: df_mor_continuous.loc[
                    df_mor_continuous[gb.PREDICTOR] == pred_2, gb.PLOT_LINK_MOR
                ].values[0],
            }
            lst_corr_cont_cont_table.append(dict_corr_cont_cont_table)

    df_corr_cont_cont_table = pd.DataFrame(
        lst_corr_cont_cont_table,
        columns=[
            gb.CONT_1,
            gb.CONT_2,
            gb.CORR,
            gb.CONT_1_URL,
            gb.CONT_2_URL,
        ],
    )

    df_corr_cont_cont_table = df_corr_cont_cont_table.sort_values(
        by=[gb.CORR], ascending=False
    )
    # plot the cont/cont correlation heatmap
    cont_cont_corr_heatmap = cm.corr_heatmap_plots(
        df_corr_cont_cont_table,
        gb.CONT_1,
        gb.CONT_2,
        gb.PEARSON,
        gb.CORR,
    )

    # Writing the matrix into HTML
    with open(gb.HTML_FILE, "a") as f:
        heading = "Continuous/Continuous Correlations"
        f.write(
            f"<h1 style='text-align:center; font-size:50px; font-weight:bold;'>{heading}</h1>"
        )
        f.write(cont_cont_corr_heatmap)

    # Adding the condition to remove the self correlation values from the correlation table
    df_corr_cont_cont_table = df_corr_cont_cont_table[
        df_corr_cont_cont_table[gb.CONT_1] != df_corr_cont_cont_table[gb.CONT_2]
    ]

    return df_corr_cont_cont_table


def correlation_cat_cat_table(cm, df, cat_preds, df_mor_categorical):
    """
    Function to generate correlation matrix for cramer's and tschuprow's categorical correlation
    :param cm: correlation metrics object
    :param df: Pandas Dataframe
    :param cat_preds: Categorical Predictors list
    :param df_mor_categorical: Mean of Response dataframe for categorical predictors
    :return: df_corr_cat_cat_table_cramers_v, df_corr_cat_cat_table_tschuprow:
    """
    lst_corr_cat_cat_table_cramers_v = []
    lst_corr_cat_cat_table_tschuprow = []
    for pred_1 in cat_preds:
        for pred_2 in cat_preds:
            dict_corr_cat_cat_table_cramers_v = {
                gb.CAT_1: pred_1,
                gb.CAT_2: pred_2,
                gb.CORR: cm.cat_correlation(df[pred_1], df[pred_2]),
                gb.CAT_1_URL: df_mor_categorical.loc[
                    df_mor_categorical[gb.PREDICTOR] == pred_1, gb.PLOT_LINK_MOR
                ].values[0],
                gb.CAT_2_URL: df_mor_categorical.loc[
                    df_mor_categorical[gb.PREDICTOR] == pred_2, gb.PLOT_LINK_MOR
                ].values[0],
            }
            lst_corr_cat_cat_table_cramers_v.append(dict_corr_cat_cat_table_cramers_v)

            dict_corr_cat_cat_table_tschuprow = {
                gb.CAT_1: pred_1,
                gb.CAT_2: pred_2,
                gb.CORR: cm.cat_correlation(df[pred_1], df[pred_2], tschuprow=True),
                gb.CAT_1_URL: df_mor_categorical.loc[
                    df_mor_categorical[gb.PREDICTOR] == pred_1, gb.PLOT_LINK_MOR
                ].values[0],
                gb.CAT_2_URL: df_mor_categorical.loc[
                    df_mor_categorical[gb.PREDICTOR] == pred_2, gb.PLOT_LINK_MOR
                ].values[0],
            }
            lst_corr_cat_cat_table_tschuprow.append(dict_corr_cat_cat_table_tschuprow)

    df_corr_cat_cat_table_cramers_v = pd.DataFrame(
        lst_corr_cat_cat_table_cramers_v,
        columns=[
            gb.CAT_1,
            gb.CAT_2,
            gb.CORR,
            gb.CAT_1_URL,
            gb.CAT_2_URL,
        ],
    )

    df_corr_cat_cat_table_tschuprow = pd.DataFrame(
        lst_corr_cat_cat_table_tschuprow,
        columns=[
            gb.CAT_1,
            gb.CAT_2,
            gb.CORR,
            gb.CAT_1_URL,
            gb.CAT_2_URL,
        ],
    )

    # Sorting the values in correlation table in descending order
    df_corr_cat_cat_table_cramers_v = df_corr_cat_cat_table_cramers_v.sort_values(
        by=[gb.CORR], ascending=False
    )
    df_corr_cat_cat_table_tschuprow = df_corr_cat_cat_table_tschuprow.sort_values(
        by=[gb.CORR], ascending=False
    )

    # plot the cont/cont correlation heatmap of cramer
    cat_cat_corr_heatmap_cramers_v = cm.corr_heatmap_plots(
        df_corr_cat_cat_table_cramers_v,
        gb.CAT_1,
        gb.CAT_2,
        gb.CRAMER,
        gb.CORR,
    )

    # plot the cont/cont correlation heatmap of tschuprow
    cat_cat_corr_heatmap_tschuprow = cm.corr_heatmap_plots(
        df_corr_cat_cat_table_tschuprow,
        gb.CAT_1,
        gb.CAT_2,
        gb.TSCHUPROW,
        gb.CORR,
    )

    # Writing the matrix into HTML
    with open(gb.HTML_FILE, "a") as f:
        heading = "Categorical/Categorical Correlations"
        f.write(
            f"<h1 style='text-align:center; font-size:50px; font-weight:bold;'>{heading}</h1>"
        )
        f.write(cat_cat_corr_heatmap_cramers_v)
        f.write(cat_cat_corr_heatmap_tschuprow)

    # Adding the condition to remove the self correlation values from the correlation table
    df_corr_cat_cat_table_cramers_v = df_corr_cat_cat_table_cramers_v[
        df_corr_cat_cat_table_cramers_v[gb.CAT_1]
        != df_corr_cat_cat_table_cramers_v[gb.CAT_2]
    ]

    df_corr_cat_cat_table_tschuprow = df_corr_cat_cat_table_tschuprow[
        df_corr_cat_cat_table_tschuprow[gb.CAT_1]
        != df_corr_cat_cat_table_tschuprow[gb.CAT_2]
    ]

    return df_corr_cat_cat_table_cramers_v, df_corr_cat_cat_table_tschuprow


def correlation_cat_cont_table(
    cm, df, cat_preds, cont_preds, df_mor_categorical, df_mor_continuous
):
    """
    Function to generate correlation matrix for categorical vs continuous correlation
    :param cm: correlation metrics object
    :param df: Pandas Dataframe
    :param cat_preds: Categorical Predictors list
    :param cont_preds: Continuous Predictors list
    :param df_mor_categorical: Mean of Response dataframe for categorical predictors
    :param df_mor_continuous: Mean of Response dataframe for continuous predictors
    :return:
    """
    lst_corr_cat_cont_table = []
    for cat_pred in cat_preds:
        for cont_pred in cont_preds:
            dict_corr_cat_cont_table = {
                gb.CAT: cat_pred,
                gb.CONT: cont_pred,
                gb.CORR: cm.cat_cont_correlation_ratio(df[cat_pred], df[cont_pred]),
                gb.CAT_URL: df_mor_categorical.loc[
                    df_mor_categorical[gb.PREDICTOR] == cat_pred, gb.PLOT_LINK_MOR
                ].values[0],
                gb.CONT_URL: df_mor_continuous.loc[
                    df_mor_continuous[gb.PREDICTOR] == cont_pred, gb.PLOT_LINK_MOR
                ].values[0],
            }
            lst_corr_cat_cont_table.append(dict_corr_cat_cont_table)

    df_corr_cat_cont_table = pd.DataFrame(
        lst_corr_cat_cont_table,
        columns=[
            gb.CAT,
            gb.CONT,
            gb.CORR,
            gb.CAT_URL,
            gb.CONT_URL,
        ],
    )

    df_corr_cat_cont_table = df_corr_cat_cont_table.sort_values(
        by=[gb.CORR], ascending=False
    )

    # plot the cont/cont correlation heatmap
    cat_cont_corr_heatmap = cm.corr_heatmap_plots(
        df_corr_cat_cont_table,
        gb.CAT,
        gb.CONT,
        gb.RATIO,
        gb.CORR,
    )

    # Writing the matrix into HTML
    with open(gb.HTML_FILE, "a") as f:
        heading = "Categorical/Continuous Correlations"
        f.write(
            f"<h1 style='text-align:center; font-size:50px; font-weight:bold;'>{heading}</h1>"
        )
        f.write(cat_cont_corr_heatmap)

    return df_corr_cat_cont_table


def correlation_metrics(df, predictors, df_mor_categorical, df_mor_continuous):
    """
    Function to generate correlation matrix for categorical and continuous predictors
    :param df: Pandas Dataframe
    :param predictors: Predictors List
    :param df_mor_categorical: Mean of Response dataframe for categorical predictors
    :param df_mor_continuous: Mean of Response dataframe for continuous predictors
    :return:  df_corr_cat_cat_table_cramers_v, df_corr_cat_cat_table_tschuprow,
            df_corr_cont_cont_table, df_corr_cat_cont_table,
    """
    df_corr_cat_cat_table_cramers_v = pd.DataFrame()
    df_corr_cat_cat_table_tschuprow = pd.DataFrame()
    df_corr_cont_cont_table = pd.DataFrame()
    df_corr_cat_cont_table = pd.DataFrame()

    cm = CorrelationMetrics()
    cat_preds, cont_preds = return_categorical_continuous_predictor_list(df, predictors)

    if len(cont_preds) > 0:
        # Getting the correlation coefficients of cont/cont predictors
        df_corr_cont_cont_table = correlation_cont_cont_table(
            cm, df, cont_preds, df_mor_continuous
        )

        # Save the cont_cont correlation table to HTML
        save_dataframe_to_HTML(
            df_corr_cont_cont_table,
            gb.CONT_1_URL,
            gb.CONT_2_URL,
            gb.CORRELATION_PEARSON_CAPTION,
        )

    if len(cat_preds) > 0:
        # Getting the correlation coefficients of cat/cat predictors
        (
            df_corr_cat_cat_table_cramers_v,
            df_corr_cat_cat_table_tschuprow,
        ) = correlation_cat_cat_table(cm, df, cat_preds, df_mor_categorical)

        # Save the cont_cont correlation table to HTML
        save_dataframe_to_HTML(
            df_corr_cat_cat_table_cramers_v,
            gb.CAT_1_URL,
            gb.CAT_2_URL,
            gb.CORRELATION_CRAMER_CAPTION,
        )
        save_dataframe_to_HTML(
            df_corr_cat_cat_table_tschuprow,
            gb.CAT_1_URL,
            gb.CAT_2_URL,
            gb.CORRELATION_TSCHUPROW_CAPTION,
        )

    if len(cat_preds) > 0 and len(cont_preds) > 0:
        # Getting the correlation coefficients of cat/cont predictors
        df_corr_cat_cont_table = correlation_cat_cont_table(
            cm, df, cat_preds, cont_preds, df_mor_categorical, df_mor_continuous
        )

        # Save the cat/cont correlation into HTML table
        save_dataframe_to_HTML(
            df_corr_cat_cont_table,
            gb.CAT_URL,
            gb.CONT_URL,
            gb.CORRELATION_RATIO_TABLE,
        )

    return (
        df_corr_cat_cat_table_cramers_v,
        df_corr_cat_cat_table_tschuprow,
        df_corr_cont_cont_table,
        df_corr_cat_cont_table,
    )


def brute_force_metrics(
    df,
    predictors,
    response,
    path_brute_force_plot,
    df_cramer,
    df_tschuprow,
    df_pearson,
    df_correlation_ratio,
):
    """
    Function to generate brute force metrics table and plots
    :param df: Pandas Dataframe
    :param predictors: Predictors List
    :param response: Response Text
    :param path_brute_force_plot: Brute Force plots URL
    :return: None
    """
    bf_mor = BruteForceMeanOfResponse()
    cat_preds, cont_preds = return_categorical_continuous_predictor_list(df, predictors)

    # For cat_cat brute force combination
    if len(cat_preds) > 1:
        lst_cat_cat_bf_table = []
        for pred_1 in cat_preds:
            for pred_2 in cat_preds:
                if pred_1 != pred_2:
                    lst_cat_cat_bf_table.append(
                        bf_mor.brute_force_cat_cat_mean_of_response(
                            df, pred_1, pred_2, response, path_brute_force_plot
                        )
                    )

        df_cat_cat_brute_force_table = pd.DataFrame(
            lst_cat_cat_bf_table,
            columns=[
                gb.CAT_1,
                gb.CAT_2,
                gb.DIFF_MEAN_RESP_RANKING,
                gb.DIFF_MEAN_RESP_WEIGHTED_RANKING,
                gb.LINK,
            ],
        )

        df_cat_cat_brute_force_table = df_cat_cat_brute_force_table.sort_values(
            by=[gb.DIFF_MEAN_RESP_WEIGHTED_RANKING], ascending=False
        )

        # Merging the Brute Force Dataframes with Correlation Matrix
        df_cat_cat_brute_force_merged = pd.merge(
            pd.merge(
                df_cat_cat_brute_force_table,
                df_cramer[[gb.CAT_1, gb.CAT_2, gb.CORR]],
                on=[gb.CAT_1, gb.CAT_2],
            ),
            df_tschuprow[[gb.CAT_1, gb.CAT_2, gb.CORR]],
            on=[gb.CAT_1, gb.CAT_2],
        )

        df_cat_cat_brute_force_merged = df_cat_cat_brute_force_merged.rename(
            columns={f"{gb.CORR}_x": gb.CRAMER, f"{gb.CORR}_y": gb.TSCHUPROW}
        )

        # Getting the absolute values of cramer and tschuprow correlation
        df_cat_cat_brute_force_merged[gb.ABS_CRAMER] = df_cat_cat_brute_force_merged[
            gb.CRAMER
        ].abs()
        df_cat_cat_brute_force_merged[gb.ABS_TSCHUPROW] = df_cat_cat_brute_force_merged[
            gb.TSCHUPROW
        ].abs()

        # Save the cat/cat correlation into HTML table
        save_brute_force_dataframe_to_HTML(
            df_cat_cat_brute_force_merged, gb.LINK, gb.CAT_CAT_BRUTE_FORCE_CAPTION
        )

    #  For cont_cont brute force combination
    if len(cont_preds) > 1:
        lst_cont_cont_bf_table = []
        for pred_1 in cont_preds:
            for pred_2 in cont_preds:
                if pred_1 != pred_2:
                    lst_cont_cont_bf_table.append(
                        bf_mor.brute_force_cont_cont_mean_of_response(
                            df, pred_1, pred_2, response, path_brute_force_plot
                        )
                    )

        df_cont_cont_brute_force_table = pd.DataFrame(
            lst_cont_cont_bf_table,
            columns=[
                gb.CONT_1,
                gb.CONT_2,
                gb.DIFF_MEAN_RESP_RANKING,
                gb.DIFF_MEAN_RESP_WEIGHTED_RANKING,
                gb.LINK,
            ],
        )

        df_cont_cont_brute_force_table = df_cont_cont_brute_force_table.sort_values(
            by=[gb.DIFF_MEAN_RESP_WEIGHTED_RANKING], ascending=False
        )

        # Merging the cont_cont brute force table with pearson corrlation values
        df_cont_cont_brute_force_merged = pd.merge(
            df_cont_cont_brute_force_table,
            df_pearson[[gb.CONT_1, gb.CONT_2, gb.CORR]],
            on=[gb.CONT_1, gb.CONT_2],
        )

        # Renaming the correlation column to match the final brute force table
        df_cont_cont_brute_force_merged = df_cont_cont_brute_force_merged.rename(
            columns={f"{gb.CORR}": gb.PEARSON}
        )

        # Getting the absolute values of pearson correlation
        df_cont_cont_brute_force_merged[
            gb.ABS_PEARSON
        ] = df_cont_cont_brute_force_merged[gb.PEARSON].abs()

        # Save the cont/cont correlation into HTML table
        save_brute_force_dataframe_to_HTML(
            df_cont_cont_brute_force_merged, gb.LINK, gb.CONT_CONT_BRUTE_FORCE_CAPTION
        )

    #  For cat_cont brute force combination
    if len(cat_preds) > 0 and len(cont_preds) > 0:
        lst_cat_cont_bf_table = []
        for cat_pred in cat_preds:
            for cont_pred in cont_preds:
                lst_cat_cont_bf_table.append(
                    bf_mor.brute_force_cat_cont_mean_of_response(
                        df, cat_pred, cont_pred, response, path_brute_force_plot
                    )
                )

        df_cat_cont_brute_force_table = pd.DataFrame(
            lst_cat_cont_bf_table,
            columns=[
                gb.CAT,
                gb.CONT,
                gb.DIFF_MEAN_RESP_RANKING,
                gb.DIFF_MEAN_RESP_WEIGHTED_RANKING,
                gb.LINK,
            ],
        )

        df_cat_cont_brute_force_table = df_cat_cont_brute_force_table.sort_values(
            by=[gb.DIFF_MEAN_RESP_WEIGHTED_RANKING], ascending=False
        )

        # Merging the cat_cont brute force table with correlation ratio values
        df_cat_cont_brute_force_merged = pd.merge(
            df_cat_cont_brute_force_table,
            df_correlation_ratio[[gb.CAT, gb.CONT, gb.CORR]],
            on=[gb.CAT, gb.CONT],
        )

        # Renaming the correlation column to match the final brute force table
        df_cat_cont_brute_force_merged = df_cat_cont_brute_force_merged.rename(
            columns={f"{gb.CORR}": gb.CORR_RATIO}
        )

        # Getting the absolute values of correlation ratio
        df_cat_cont_brute_force_merged[
            gb.ABS_CORR_RATIO
        ] = df_cat_cont_brute_force_merged[gb.CORR_RATIO].abs()

        # Save the cat/cont correlation into HTML table
        save_brute_force_dataframe_to_HTML(
            df_cat_cont_brute_force_merged, gb.LINK, gb.CAT_CONT_BRUTE_FORCE_CAPTION
        )

    return


def mean_of_response_plots_weighted_unweighted_response_metrics(
    df, predictors, response, path
):
    """
    Function to generate mean of response plots, weighted and unweighted responses
    :param df: Pandas Dataframe
    :param predictors: predictors list
    :param response: response text
    :param path: Mean of response plots path
    :return: df_summary_statistics_mean_of_response_categorical,
        df_summary_statistics_mean_of_response_continuous,
    """

    lst_summary_statistics_mean_of_response_categorical = []
    lst_summary_statistics_mean_of_response_continuous = []
    mean_of_response = MeanOfResponse()
    (
        categorical_predictors_list,
        continuous_predictors_list,
    ) = return_categorical_continuous_predictor_list(df, predictors)
    response_type = return_response_type(df[response])

    for predictor in categorical_predictors_list:
        if response_type == gb.CONTINUOUS_TYPE_RESP:
            lst_summary_statistics_mean_of_response_categorical.append(
                mean_of_response.mean_of_response_cat_pred_cont_resp(
                    df, predictor, response, path
                )
            )

        elif response_type == gb.BOOLEAN_TYPE_RESP:
            lst_summary_statistics_mean_of_response_categorical.append(
                mean_of_response.mean_of_response_cat_pred_cat_resp(
                    df, predictor, response, path
                )
            )

    for predictor in continuous_predictors_list:
        if response_type == gb.CONTINUOUS_TYPE_RESP:
            lst_summary_statistics_mean_of_response_continuous.append(
                mean_of_response.mean_of_response_cont_pred_cont_resp(
                    df, predictor, response, path
                )
            )

        elif response_type == gb.BOOLEAN_TYPE_RESP:
            lst_summary_statistics_mean_of_response_continuous.append(
                mean_of_response.mean_of_response_cont_pred_cat_resp(
                    df, predictor, response, path
                )
            )

    df_summary_statistics_mean_of_response_categorical = pd.DataFrame(
        lst_summary_statistics_mean_of_response_categorical,
        columns=[
            gb.PREDICTOR,
            gb.WEIGHTED_MEAN_SQUARED_DIFF,
            gb.MEAN_SQUARED_DIFF,
            gb.PLOT_LINK_MOR,
        ],
    )

    df_summary_statistics_mean_of_response_continuous = pd.DataFrame(
        lst_summary_statistics_mean_of_response_continuous,
        columns=[
            gb.PREDICTOR,
            gb.WEIGHTED_MEAN_SQUARED_DIFF,
            gb.MEAN_SQUARED_DIFF,
            gb.PLOT_LINK_MOR,
        ],
    )

    return (
        df_summary_statistics_mean_of_response_categorical,
        df_summary_statistics_mean_of_response_continuous,
    )


def resp_vs_pred_plots(df, predictors, response, path):
    """
    This function generates a dataframe with predictor and its plot with response
    :param df:
    :param predictors:
    :param response:
    :param path:
    :return:
    """
    predictor_vs_response_plots = PredictorVsResponsePlots()
    lst_resp_vs_pred_plots_categorical = []
    lst_resp_vs_pred_plots_continuous = []
    (
        categorical_predictors_list,
        continuous_predictors_list,
    ) = return_categorical_continuous_predictor_list(df, predictors)
    response_type = return_response_type(df[response])

    for predictor in categorical_predictors_list:
        if response_type == gb.CONTINUOUS_TYPE_RESP:
            lst_resp_vs_pred_plots_categorical.append(
                predictor_vs_response_plots.cont_response_cat_predictor(
                    df, predictor, response, path
                )
            )

        elif response_type == gb.BOOLEAN_TYPE_RESP:
            lst_resp_vs_pred_plots_categorical.append(
                predictor_vs_response_plots.cat_response_cat_predictor(
                    df, predictor, response, path
                )
            )

    for predictor in continuous_predictors_list:
        if response_type == gb.CONTINUOUS_TYPE_RESP:
            lst_resp_vs_pred_plots_continuous.append(
                predictor_vs_response_plots.cont_response_cont_predictor(
                    df, predictor, response, path
                )
            )

        elif response_type == gb.BOOLEAN_TYPE_RESP:
            lst_resp_vs_pred_plots_continuous.append(
                predictor_vs_response_plots.cat_response_cont_predictor(
                    df, predictor, response, path
                )
            )

    df_resp_vs_pred_plots_categorical = pd.DataFrame(
        lst_resp_vs_pred_plots_categorical,
        columns=[gb.PREDICTOR, gb.PLOT_LINK],
    )

    df_resp_vs_pred_plots_continuous = pd.DataFrame(
        lst_resp_vs_pred_plots_continuous,
        columns=[gb.PREDICTOR, gb.PLOT_LINK],
    )

    return df_resp_vs_pred_plots_categorical, df_resp_vs_pred_plots_continuous


def get_p_t_value(df, predictors, response):
    """
    Function to get the dataframe with p_value and t_value of continuous predictors
    :param df:
    :param predictors:
    :param response:
    :return:
    """
    (
        categorical_predictors_list,
        continuous_predictors_list,
    ) = return_categorical_continuous_predictor_list(df, predictors)
    response_type = return_response_type(df[response])
    lst_summary_statistics_p_t_values_continuous = []

    for predictor in continuous_predictors_list:
        if response_type == gb.CONTINUOUS_TYPE_RESP:
            lst_summary_statistics_p_t_values_continuous.append(
                create_linear_regression(df, predictor, response)
            )
        elif response_type == gb.BOOLEAN_TYPE_RESP:
            lst_summary_statistics_p_t_values_continuous.append(
                create_logistic_regression(df, predictor, response)
            )

    df_summary_statistics_p_t_values_continuous = pd.DataFrame(
        lst_summary_statistics_p_t_values_continuous,
        columns=[gb.PREDICTOR, gb.P_VALUE, gb.T_VALUE],
    )

    return df_summary_statistics_p_t_values_continuous


def process_dataframes(dataset, df, predictors, response):
    """
    Driver function to process the dataframes, merge and save the dataframes into the HTML page
    :param dataset: Dataset name string
    :param df: Pandas Dataframe
    :param predictors: List of predictors
    :param response: Response String
    :return: None
    """
    # Getting the path of mean of response plots and predictor vs response plots
    path_mean_of_response_plot = create_mean_of_response_plot_folder(dataset)
    path_predictor_vs_response_plot = create_resp_pred_plot_folder(dataset)
    path_brute_force_plot = create_brute_force_plot_folder(dataset)

    # Get the list of categorical and continuous predictors
    (
        categorical_predictors_list,
        continuous_predictors_list,
    ) = return_categorical_continuous_predictor_list(df, predictors)

    # Get the MOR plots, weighted and unweighted mean of response metrics for categorical and continuous predictors
    (
        df_mor_categorical,
        df_mor_continuous,
    ) = mean_of_response_plots_weighted_unweighted_response_metrics(
        df, predictors, response, path_mean_of_response_plot
    )

    # Get the response vs predictors plot URLs for categorical and continuous predictors
    df_resp_vs_pred_categorical, df_resp_vs_pred_continuous = resp_vs_pred_plots(
        df, predictors, response, path_predictor_vs_response_plot
    )

    # Random Forest variable importance for continuous predictors
    df_rf_continuous_predictors_importance = random_forest_variable_importance_ranking(
        df, continuous_predictors_list, response
    )

    # Getting p_value and t_value for continuous predictors
    df_summary_statistics_p_t_values_continuous = get_p_t_value(
        df, predictors, response
    )

    # Merging the continuous predictors dataframes
    df_mor_continuous.set_index(gb.PREDICTOR, inplace=True)
    df_resp_vs_pred_continuous.set_index(gb.PREDICTOR, inplace=True)
    df_rf_continuous_predictors_importance.set_index(gb.PREDICTOR, inplace=True)
    df_summary_statistics_p_t_values_continuous.set_index(gb.PREDICTOR, inplace=True)

    df_continuous_merged = df_mor_continuous.join(
        [
            df_resp_vs_pred_continuous,
            df_rf_continuous_predictors_importance,
            df_summary_statistics_p_t_values_continuous,
        ],
        on=None,
    )
    df_continuous_merged.reset_index(inplace=True)
    df_mor_continuous.reset_index(inplace=True)

    # Merging the categorical predictors dataframes
    df_mor_categorical.set_index(gb.PREDICTOR, inplace=True)
    df_resp_vs_pred_categorical.set_index(gb.PREDICTOR, inplace=True)
    df_categorical_merged = df_mor_categorical.join(
        [df_resp_vs_pred_categorical], on=None
    )
    df_categorical_merged.reset_index(inplace=True)
    df_mor_categorical.reset_index(inplace=True)

    save_dataframe_to_HTML(
        df_continuous_merged,
        gb.PLOT_LINK_MOR,
        gb.PLOT_LINK,
        caption=gb.CONTINUOUS_PREDICTORS_CAPTION,
    )
    save_dataframe_to_HTML(
        df_categorical_merged,
        gb.PLOT_LINK_MOR,
        gb.PLOT_LINK,
        caption=gb.CATEGORICAL_PREDICTORS_CAPTION,
    )

    (
        df_corr_cat_cat_table_cramers_v,
        df_corr_cat_cat_table_tschuprow,
        df_corr_cont_cont_table,
        df_corr_cat_cont_table,
    ) = correlation_metrics(df, predictors, df_mor_categorical, df_mor_continuous)

    brute_force_metrics(
        df,
        predictors,
        response,
        path_brute_force_plot,
        df_corr_cat_cat_table_cramers_v,
        df_corr_cat_cat_table_tschuprow,
        df_corr_cont_cont_table,
        df_corr_cat_cont_table,
    )

    return


def machine_learning_pipelines(dataset, df, predictors, response, heading):

    # Sort DataFrame based on 'game_date' in descending order
    df.sort_values(by="game_date", ascending=False, inplace=True)

    # Split the predictors and response
    X = df[predictors]
    y = df[response]

    # Categorical Columns
    categorical_columns = []
    for predictor in predictors:
        col_type = return_predictor_type(df[predictor])
        if col_type == gb.CATEGORICAL_TYPE_PRED:
            categorical_columns.append(predictor)

    # Creating one hot encoder for categorical columns
    onehot_encoder = OneHotEncoder(sparse=False)
    for column in categorical_columns:
        encoded_columns = pd.DataFrame(onehot_encoder.fit_transform(X[[column]]))
        X = pd.concat([X, encoded_columns], axis=1)
        X.drop(column, axis=1, inplace=True)

    # Calculate split index position
    split_index = math.ceil(len(df) * 0.8)  # 80% for training, 20% for testing

    # Building the Machine Learning Models
    print(
        f"Running Machine Learning Pipeline for {dataset} dataset with features {predictors}"
    )

    X.columns = X.columns.astype(str)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Creating Pipeline for transformation and model creation
    pipeline_SVC = Pipeline(
        [("scaler", StandardScaler()), ("svc", SVC(probability=True))]
    )
    pipeline_LR = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
    pipeline_KNN = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=35))]
    )

    pipeline_RFC = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rfc", RandomForestClassifier(n_estimators=100)),
        ]
    )

    pipeline_DT = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("dtc", DecisionTreeClassifier(max_depth=5)),
        ]
    )

    pipelines = [pipeline_SVC, pipeline_LR, pipeline_KNN, pipeline_RFC, pipeline_DT]

    # List to store ROC Curve data for each model
    roc_data = []

    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    with open(gb.HTML_FILE, "a") as f:
        f.write(
            f"<h1 style='text-align:center; font-size:50px; font-weight:bold;'>{heading}</h1>"
        )

    for i, model in enumerate(pipelines):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        roc_data.append((pipelines[i].steps[-1][0].upper(), fpr, tpr, auc_score))

        # Precision, Recall, and F1 Score
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        with open(gb.HTML_FILE, "a") as f:
            f.write(f"<h2>{pipelines[i]} Metrics</h2>")
            f.write(f"<p>Precision: {precision}</p>")
            f.write(f"<p>Recall: {recall}</p>")
            f.write(f"<p>F1 Score: {f1}</p>")
            f.write(f"<p>Accuracy: {model.score(X_test, y_test)}</p>")
            f.write(f"<p>AUC Score: {auc_score}</p>")

    # Create the Plotly figure
    fig = go.Figure()

    # Add ROC curves as traces to the figure
    for name, fpr, tpr, auc_score in roc_data:
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name="{} (AUC = {:.2f})".format(name, auc_score),
            )
        )

    # Customize the plot layout
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.5, y=-0.2),
        xaxis=dict(range=[0, 1], constrain="domain"),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
        hovermode="closest",
    )

    # Add the ROC Curve into the final html
    with open(gb.HTML_FILE, "a") as f:
        f.write(pio.to_html(fig, include_plotlyjs="cdn"))

    return


def main():
    # Loading the dataset, predictors list and response variable
    dataset, df, predictors, response = load_dataset()

    # Writing the name of Dataset on top of HTML File
    with open(gb.HTML_FILE, "a") as f:
        heading = dataset.upper() + " DATASET"
        f.write(
            f"<h1 style='text-align:center; font-size:50px; font-weight:bold;'>{heading}</h1>"
        )

    # Creating a copy of original dataframe for Machine Learning Pipeline
    df_original = df
    df = df.drop("game_date", axis=1)

    # Function to process all the dataframes and generate plots for feature engineering
    process_dataframes(dataset, df, predictors, response)

    # Machine Learning Pipelines
    machine_learning_pipelines(
        dataset, df_original, predictors, response, "Model Evaluation with all features"
    )

    # Selected best features after feature Engineering
    predictors_1 = [
        "pitchers_strikeout_to_walk_ratio_difference",
        "pitchers_opponent_batting_average_difference",
        "pitchers_strikeout_rate_difference",
        "pitchers_dice_difference",
        "team_gross_production_average_difference",
        "team_walk_to_strikeout_ratio_difference",
        "overcast",
    ]

    machine_learning_pipelines(
        dataset,
        df_original,
        predictors_1,
        response,
        f"Model Evaluation with selected best features {predictors_1}",
    )

    return


if __name__ == "__main__":
    sys.exit(main())
