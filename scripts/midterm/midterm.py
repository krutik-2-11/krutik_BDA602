import os
import sys

import pandas as pd
import statsmodels.api
from correlation_metrics import CorrelationMetrics
from data_loader import TestDatasets
from mean_of_response import MeanOfResponse
from predictor_vs_response_plots import PredictorVsResponsePlots
from sklearn.ensemble import RandomForestRegressor

# Declaring the globals here
CONTINUOUS_TYPE_PRED = "continuous predictor"
CATEGORICAL_TYPE_PRED = "categorical predictor"
BOOLEAN_TYPE_RESP = "boolean response"
CONTINUOUS_TYPE_RESP = "continuous response"
PATH_RESP_PRED = "response_predictor_plots"
PATH_MEAN_OF_RESPONSE = "mean_of_response_plots"
PLOT_LINK_MOR = "plot_link_mor"
PLOT_LINK = "plot_link"
PREDICTOR = "predictor"
WEIGHTED_MEAN_SQUARED_DIFF = "weighted_mean_squared_diff"
MEAN_SQUARED_DIFF = "mean_squared_diff"
CATEGORICAL_PREDICTORS_CAPTION = "Categorical Predictors"
CONTINUOUS_PREDICTORS_CAPTION = "Continuous Predictors"
IMPORTANCE = "importance"
P_VALUE = "p_value"
T_VALUE = "t_value"
CONT_1 = "cont_1"
CONT_2 = "cont_2"
CAT_1 = "cat_1"
CAT_2 = "cat_2"
CORR = "corr"
CONT_1_URL = "cont_1_url"
CONT_2_URL = "cont_2_url"
CAT_1_URL = "cat_1_url"
CAT_2_URL = "cat_2_url"
CORRELATION_PEARSON_CAPTION = "Correlation Pearson Table"
CORRELATION_TSCHUPROW_CAPTION = "Correlation Tschuprow Table"
CORRELATION_CRAMER_CAPTION = "Correlation Cramer Table"
PEARSON = "Pearson's"
TSCHUPROW = "Tschuprow"
CRAMER = "Cramer"


def load_dataset():
    """
    Function to load the dataset with the user prompt.
    Returns dataset name, dataframe, predictors list and response
    :return: dataset, df, predictors, response
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

    return dataset, df, predictors, response


def create_resp_pred_plot_folder(dataset):
    """
    Function to create a folder for response predictor plots
    :param dataset:
    :return:
    """
    path = f"{PATH_RESP_PRED}"
    if not os.path.exists(path):
        os.mkdir(path)
    path = f"{path}/{dataset}"
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def create_mean_of_response_plot_folder(dataset):
    """
    Function to create a folder for mean of response plots
    :param dataset:
    :return:
    """
    path = f"{PATH_MEAN_OF_RESPONSE}"
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
    :param path:
    :return:
    """
    url = "file://" + os.path.abspath(path)
    return f'<a href="{url}">Plot</a>'


def save_dataframe_to_HTML(df, plot_link_mor, plot_link, caption):
    """
    Function to convert dataframe into HTML
    :param df:
    :param caption:
    :return:
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
        .set_caption(caption)
    )

    # Generate an HTML table from the styled DataFrame
    html_table = styled_table.to_html()

    with open("my_table.html", "a") as f:
        f.write(html_table)


def return_predictor_type(column):
    """
    Function to categorize the predictor into categorical or continuous type
    :param column:
    :return: string "categorical" or "continuous"
    """
    if column.dtype in ["object", "bool"]:
        return CATEGORICAL_TYPE_PRED
    else:
        return CONTINUOUS_TYPE_PRED


def return_response_type(column):
    """
    Function to categorize the response into boolean or continuous type
    :param column:
    :return: string "continuous" or "boolean"
    """
    if len(column.unique()) > 2:
        return CONTINUOUS_TYPE_RESP
    else:
        return BOOLEAN_TYPE_RESP


def return_categorical_continuous_predictor_list(df, predictors):
    """
    Function to return the list of categorical and continuous predictors
    :param df:
    :param predictors:
    :return: categorical_predictors_list, continuous_predictors_list
    """
    categorical_predictors_list = []
    continuous_predictors_list = []
    for predictor in predictors:
        predictor_type = return_predictor_type(df[predictor])
        if predictor_type == CATEGORICAL_TYPE_PRED:
            categorical_predictors_list.append(predictor)
        elif predictor_type == CONTINUOUS_TYPE_PRED:
            continuous_predictors_list.append(predictor)

    return categorical_predictors_list, continuous_predictors_list


def random_forest_variable_importance_ranking(df, continuous_predictors, response):

    X = df[continuous_predictors]
    y = df[response]

    # create a random forest regressor
    rf = RandomForestRegressor(random_state=42)

    # Fit the model into the data
    rf.fit(X, y)

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame(
        {PREDICTOR: continuous_predictors, IMPORTANCE: importances}
    )

    # Sort the dataframe by feature importance in descending order
    feature_importances = feature_importances.sort_values(
        IMPORTANCE, ascending=False
    ).reset_index(drop=True)

    return feature_importances


def create_linear_regression(df, predictor, response):
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

    summary_dict[PREDICTOR] = predictor
    summary_dict[P_VALUE] = p_value
    summary_dict[T_VALUE] = t_value

    return summary_dict


def create_logistic_regression(df, predictor, response):
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

    summary_dict[PREDICTOR] = predictor
    summary_dict[P_VALUE] = p_value
    summary_dict[T_VALUE] = t_value

    return summary_dict


def correlation_cont_cont_table(cm, df, cont_preds, df_mor_continuous):
    """
    Function to generate correlation matrix plot and table for cont/cont predictors
    :param cm:
    :param df:
    :param cont_preds:
    :param df_mor_continuous:
    :return: df_corr_cont_cont_table
    """
    lst_corr_cont_cont_table = []
    for pred_1 in cont_preds:
        for pred_2 in cont_preds:
            dict_corr_cont_cont_table = {
                CONT_1: pred_1,
                CONT_2: pred_2,
                CORR: cm.cont_cont_correlation(df[pred_1], df[pred_2]),
                CONT_1_URL: df_mor_continuous.loc[
                    df_mor_continuous[PREDICTOR] == pred_1, PLOT_LINK_MOR
                ].values[0],
                CONT_2_URL: df_mor_continuous.loc[
                    df_mor_continuous[PREDICTOR] == pred_2, PLOT_LINK_MOR
                ].values[0],
            }
            lst_corr_cont_cont_table.append(dict_corr_cont_cont_table)

    df_corr_cont_cont_table = pd.DataFrame(
        lst_corr_cont_cont_table,
        columns=[
            CONT_1,
            CONT_2,
            CORR,
            CONT_1_URL,
            CONT_2_URL,
        ],
    )

    df_corr_cont_cont_table = df_corr_cont_cont_table.sort_values(
        by=[CORR], ascending=False
    )
    # plot the cont/cont correlation heatmap
    cont_cont_corr_heatmap = cm.corr_heatmap_plots(
        df_corr_cont_cont_table,
        CONT_1,
        CONTINUOUS_TYPE_PRED,
        CONT_2,
        CONTINUOUS_TYPE_PRED,
        PEARSON,
        CORR,
    )

    # Writing the matrix into HTML
    with open("my_table.html", "a") as f:
        f.write(cont_cont_corr_heatmap)

    # Adding the condition to remove the self correlation values from the correlation table
    df_corr_cont_cont_table = df_corr_cont_cont_table[
        df_corr_cont_cont_table[CONT_1] != df_corr_cont_cont_table[CONT_2]
    ]

    return df_corr_cont_cont_table


def correlation_cat_cat_table(cm, df, cat_preds, df_mor_categorical):
    lst_corr_cat_cat_table_cramers_v = []
    lst_corr_cat_cat_table_tschuprow = []
    for pred_1 in cat_preds:
        for pred_2 in cat_preds:
            dict_corr_cat_cat_table_cramers_v = {
                CAT_1: pred_1,
                CAT_2: pred_2,
                CORR: cm.cat_correlation(df[pred_1], df[pred_2]),
                CAT_1_URL: df_mor_categorical.loc[
                    df_mor_categorical[PREDICTOR] == pred_1, PLOT_LINK_MOR
                ].values[0],
                CAT_2_URL: df_mor_categorical.loc[
                    df_mor_categorical[PREDICTOR] == pred_2, PLOT_LINK_MOR
                ].values[0],
            }
            lst_corr_cat_cat_table_cramers_v.append(dict_corr_cat_cat_table_cramers_v)

            dict_corr_cat_cat_table_tschuprow = {
                CAT_1: pred_1,
                CAT_2: pred_2,
                CORR: cm.cat_correlation(df[pred_1], df[pred_2], tschuprow=True),
                CAT_1_URL: df_mor_categorical.loc[
                    df_mor_categorical[PREDICTOR] == pred_1, PLOT_LINK_MOR
                ].values[0],
                CAT_2_URL: df_mor_categorical.loc[
                    df_mor_categorical[PREDICTOR] == pred_2, PLOT_LINK_MOR
                ].values[0],
            }
            lst_corr_cat_cat_table_tschuprow.append(dict_corr_cat_cat_table_tschuprow)

    df_corr_cat_cat_table_cramers_v = pd.DataFrame(
        lst_corr_cat_cat_table_cramers_v,
        columns=[
            CAT_1,
            CAT_2,
            CORR,
            CAT_1_URL,
            CAT_2_URL,
        ],
    )

    df_corr_cat_cat_table_tschuprow = pd.DataFrame(
        lst_corr_cat_cat_table_tschuprow,
        columns=[
            CAT_1,
            CAT_2,
            CORR,
            CAT_1_URL,
            CAT_2_URL,
        ],
    )

    # Sorting the values in correlation table in descending order
    df_corr_cat_cat_table_cramers_v = df_corr_cat_cat_table_cramers_v.sort_values(
        by=[CORR], ascending=False
    )
    df_corr_cat_cat_table_tschuprow = df_corr_cat_cat_table_tschuprow.sort_values(
        by=[CORR], ascending=False
    )

    # plot the cont/cont correlation heatmap of cramer
    cat_cat_corr_heatmap_cramers_v = cm.corr_heatmap_plots(
        df_corr_cat_cat_table_cramers_v,
        CAT_1,
        CATEGORICAL_TYPE_PRED,
        CAT_2,
        CATEGORICAL_TYPE_PRED,
        CRAMER,
        CORR,
    )

    # plot the cont/cont correlation heatmap of tschuprow
    cat_cat_corr_heatmap_tschuprow = cm.corr_heatmap_plots(
        df_corr_cat_cat_table_tschuprow,
        CAT_1,
        CATEGORICAL_TYPE_PRED,
        CAT_2,
        CATEGORICAL_TYPE_PRED,
        TSCHUPROW,
        CORR,
    )

    # Writing the matrix into HTML
    with open("my_table.html", "a") as f:
        f.write(cat_cat_corr_heatmap_cramers_v)
        f.write(cat_cat_corr_heatmap_tschuprow)

    # Adding the condition to remove the self correlation values from the correlation table
    df_corr_cat_cat_table_cramers_v = df_corr_cat_cat_table_cramers_v[
        df_corr_cat_cat_table_cramers_v[CAT_1] != df_corr_cat_cat_table_cramers_v[CAT_2]
    ]

    df_corr_cat_cat_table_tschuprow = df_corr_cat_cat_table_tschuprow[
        df_corr_cat_cat_table_tschuprow[CAT_1] != df_corr_cat_cat_table_tschuprow[CAT_2]
    ]

    return df_corr_cat_cat_table_cramers_v, df_corr_cat_cat_table_tschuprow


def correlation_metrics(
    df, predictors, response, df_mor_categorical, df_mor_continuous
):
    cm = CorrelationMetrics()
    cat_preds, cont_preds = return_categorical_continuous_predictor_list(df, predictors)

    # Getting the correlation coefficients of cont/cont predictors
    df_corr_cont_cont_table = correlation_cont_cont_table(
        cm, df, cont_preds, df_mor_continuous
    )

    # Save the cont_cont correlation table to HTML
    save_dataframe_to_HTML(
        df_corr_cont_cont_table, CONT_1_URL, CONT_2_URL, CORRELATION_PEARSON_CAPTION
    )

    # Getting the correlation coefficients of cat/cat predictors
    (
        df_corr_cat_cat_table_cramers_v,
        df_corr_cat_cat_table_tschuprow,
    ) = correlation_cat_cat_table(cm, df, cat_preds, df_mor_categorical)

    # Save the cont_cont correlation table to HTML
    save_dataframe_to_HTML(
        df_corr_cat_cat_table_cramers_v,
        CAT_1_URL,
        CAT_2_URL,
        CORRELATION_CRAMER_CAPTION,
    )
    save_dataframe_to_HTML(
        df_corr_cat_cat_table_tschuprow,
        CAT_1_URL,
        CAT_2_URL,
        CORRELATION_TSCHUPROW_CAPTION,
    )

    return


def mean_of_response_plots_weighted_unweighted_response_metrics(
    df, predictors, response, path
):
    """
    Function to generate mean of response plots, weighted and unweighted responses
    :param df:
    :param response:
    :param predictors:
    :param path:
    :return:
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
        if response_type == CONTINUOUS_TYPE_RESP:
            lst_summary_statistics_mean_of_response_categorical.append(
                mean_of_response.mean_of_response_cat_pred_cont_resp(
                    df, predictor, response, path
                )
            )

        elif response_type == BOOLEAN_TYPE_RESP:
            lst_summary_statistics_mean_of_response_categorical.append(
                mean_of_response.mean_of_response_cat_pred_cat_resp(
                    df, predictor, response, path
                )
            )

    for predictor in continuous_predictors_list:
        if response_type == CONTINUOUS_TYPE_RESP:
            lst_summary_statistics_mean_of_response_continuous.append(
                mean_of_response.mean_of_response_cont_pred_cont_resp(
                    df, predictor, response, path
                )
            )

        elif response_type == BOOLEAN_TYPE_RESP:
            lst_summary_statistics_mean_of_response_continuous.append(
                mean_of_response.mean_of_response_cont_pred_cat_resp(
                    df, predictor, response, path
                )
            )

    df_summary_statistics_mean_of_response_categorical = pd.DataFrame(
        lst_summary_statistics_mean_of_response_categorical,
        columns=[
            PREDICTOR,
            WEIGHTED_MEAN_SQUARED_DIFF,
            MEAN_SQUARED_DIFF,
            PLOT_LINK_MOR,
        ],
    )

    df_summary_statistics_mean_of_response_continuous = pd.DataFrame(
        lst_summary_statistics_mean_of_response_continuous,
        columns=[
            PREDICTOR,
            WEIGHTED_MEAN_SQUARED_DIFF,
            MEAN_SQUARED_DIFF,
            PLOT_LINK_MOR,
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
        if response_type == CONTINUOUS_TYPE_RESP:
            lst_resp_vs_pred_plots_categorical.append(
                predictor_vs_response_plots.cont_resp_cat_predictor(
                    df, predictor, response, path
                )
            )

        elif response_type == BOOLEAN_TYPE_RESP:
            lst_resp_vs_pred_plots_categorical.append(
                predictor_vs_response_plots.cat_response_cat_predictor(
                    df, predictor, response, path
                )
            )

    for predictor in continuous_predictors_list:
        if response_type == CONTINUOUS_TYPE_RESP:
            lst_resp_vs_pred_plots_continuous.append(
                predictor_vs_response_plots.cat_resp_cont_predictor(
                    df, predictor, response, path
                )
            )

        elif response_type == BOOLEAN_TYPE_RESP:
            lst_resp_vs_pred_plots_continuous.append(
                predictor_vs_response_plots.cat_resp_cont_predictor(
                    df, predictor, response, path
                )
            )

    df_resp_vs_pred_plots_categorical = pd.DataFrame(
        lst_resp_vs_pred_plots_categorical,
        columns=[PREDICTOR, PLOT_LINK],
    )

    df_resp_vs_pred_plots_continuous = pd.DataFrame(
        lst_resp_vs_pred_plots_continuous,
        columns=[PREDICTOR, PLOT_LINK],
    )

    return df_resp_vs_pred_plots_categorical, df_resp_vs_pred_plots_continuous


def get_p_t_value(df, predictors, response):
    (
        categorical_predictors_list,
        continuous_predictors_list,
    ) = return_categorical_continuous_predictor_list(df, predictors)
    response_type = return_response_type(df[response])
    lst_summary_statistics_p_t_values_continuous = []

    for predictor in continuous_predictors_list:
        if response_type == CONTINUOUS_TYPE_RESP:
            lst_summary_statistics_p_t_values_continuous.append(
                create_linear_regression(df, predictor, response)
            )
        elif response_type == BOOLEAN_TYPE_RESP:
            lst_summary_statistics_p_t_values_continuous.append(
                create_logistic_regression(df, predictor, response)
            )

    df_summary_statistics_p_t_values_continuous = pd.DataFrame(
        lst_summary_statistics_p_t_values_continuous,
        columns=[PREDICTOR, P_VALUE, T_VALUE],
    )

    return df_summary_statistics_p_t_values_continuous


def process_dataframes(dataset, df, predictors, response):
    # Getting the path of mean of response plots and predictor vs response plots
    path_mean_of_response_plot = create_mean_of_response_plot_folder(dataset)
    path_predictor_vs_response_plot = create_resp_pred_plot_folder(dataset)

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
    df_mor_continuous.set_index(PREDICTOR, inplace=True)
    df_resp_vs_pred_continuous.set_index(PREDICTOR, inplace=True)
    df_rf_continuous_predictors_importance.set_index(PREDICTOR, inplace=True)
    df_summary_statistics_p_t_values_continuous.set_index(PREDICTOR, inplace=True)

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
    df_mor_categorical.set_index(PREDICTOR, inplace=True)
    df_resp_vs_pred_categorical.set_index(PREDICTOR, inplace=True)
    df_categorical_merged = df_mor_categorical.join(
        [df_resp_vs_pred_categorical], on=None
    )
    df_categorical_merged.reset_index(inplace=True)
    df_mor_categorical.reset_index(inplace=True)

    save_dataframe_to_HTML(
        df_continuous_merged,
        PLOT_LINK_MOR,
        PLOT_LINK,
        caption=CONTINUOUS_PREDICTORS_CAPTION,
    )
    save_dataframe_to_HTML(
        df_categorical_merged,
        PLOT_LINK_MOR,
        PLOT_LINK,
        caption=CATEGORICAL_PREDICTORS_CAPTION,
    )

    correlation_metrics(df, predictors, response, df_mor_categorical, df_mor_continuous)

    return


def main():
    dataset, df, predictors, response = load_dataset()
    process_dataframes(dataset, df, predictors, response)
    return


if __name__ == "__main__":
    sys.exit(main())
