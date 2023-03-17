import os
import sys

import numpy
import pandas as pd
import statsmodels.api
from data_loader import TestDatasets
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go

PATH_RESP_PRED = "response_predictor_plots"
PATH_REGRESSION = "regression_plots"


def cont_resp_cat_predictor(df, predictor, response, path):
    group_labels = df[predictor].unique()
    hist_data = []

    for label in group_labels:
        hist_data.append(numpy.array(df[df[predictor] == label][response]))

    # Create distribution plot
    fig_1 = ff.create_distplot(hist_data, group_labels)
    fig_1.update_layout(
        title=f"Continuous Response {response} by Categorical Predictor {predictor}",
        xaxis_title=f"Response {response}",
        yaxis_title="Distribution",
    )
    # fig_1.show()

    fig_1.write_html(
        file=f"{path}/cont_response_{response}_cat_predictor_{predictor}_dist_plot.html",
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, len(curr_group)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title=f"Continuous Response {response} by Categorical Predictor {predictor}",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    # fig_2.show()

    fig_2.write_html(
        file=f"{path}/cont_response_{response}_cat_predictor_{predictor}_violin_plot.html",
        include_plotlyjs="cdn",
    )

    return


def cat_resp_cont_predictor(df, predictor, response, path):
    group_labels = df[response].unique().astype(str)
    hist_data = []

    for label in group_labels:
        hist_data.append(numpy.array(df[df[response].astype(str) == label][predictor]))

    # Create distribution plot
    fig_1 = ff.create_distplot(hist_data, group_labels)
    fig_1.update_layout(
        title=f"Continuous Response {response} by Categorical Predictor {predictor}",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    # fig_1.show()

    fig_1.write_html(
        file=f"{path}/cont_response_{response}_cat_predictor_{predictor}_dist_plot.html",
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, len(curr_hist)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title=f"Continuous Response {response} by Categorical Predictor {predictor}",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    # fig_2.show()

    fig_2.write_html(
        file=f"{path}/cont_response_{response}_cat_predictor_{predictor}_violin_plot.html",
        include_plotlyjs="cdn",
    )

    return


def cat_response_cat_predictor(df, predictor, response, path):
    # Pivot the data set to create a frequency table
    pivoted_data = df.pivot_table(index=response, columns=predictor, aggfunc="size")

    # Create the heatmap trace
    heatmap = go.Heatmap(
        x=pivoted_data.columns, y=pivoted_data.index, z=pivoted_data, colorscale="Blues"
    )

    # Define the layout of the plot
    layout = go.Layout(
        title=f"Categorical Response {response} by Categorical Predictor {predictor}",
        xaxis=dict(title=predictor),
        yaxis=dict(title=response),
    )

    fig = go.Figure(data=[heatmap], layout=layout)

    # fig.show()

    fig.write_html(
        file=f"{path}/cat_response_{response}_cat_predictor_{predictor}_heat_plot.html",
        include_plotlyjs="cdn",
    )

    return


def cont_response_cont_predictor(df, predictor, response, path):
    x = df[predictor]
    y = df[response]

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Continuous Response {response} by Continuous Predictor {predictor}",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    # fig.show()

    fig.write_html(
        file=f"{path}/cont_response_{response}_cont_predictor_{predictor}_scatter_plot.html",
        include_plotlyjs="cdn",
    )

    return


def return_column_type(column, predictor_response):
    if predictor_response == "response":
        if len(column.unique()) > 2:
            return "continuous"
        else:
            return "boolean"

    elif predictor_response == "predictor":
        if column.dtype in ["object", "bool"]:
            return "categorical"
        else:
            return "continuous"


def create_resp_pred_plot_folder(dataset):
    path = f"../{PATH_RESP_PRED}"
    if not os.path.exists(path):
        os.mkdir(path)
    path = f"{path}/{dataset}"
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def create_linear_regression(df, predictor, response):
    summary_dict = {}
    # numpy array for predictor column
    X = df[predictor]

    # numpy array for response column
    y = df[response]

    feature_name = predictor
    predict = statsmodels.api.add_constant(X)
    linear_regression_model = statsmodels.api.OLS(y, predict)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {feature_name}")
    # print(linear_regression_model_fitted.summary())

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    print(f"p_value = {p_value}")
    print(f"t_score = {t_value}")

    summary_dict["predictor"] = predictor
    summary_dict["p_value"] = p_value
    summary_dict["t_value"] = t_value

    print(summary_dict)

    return summary_dict


def create_logistic_regression(df, predictor, response):
    summary_dict = {}

    # numpy array for predictor column
    X = df[predictor]

    # numpy array for response column
    y = df[response]

    feature_name = predictor
    predict = statsmodels.api.add_constant(X)
    logistic_regression_model = statsmodels.api.Logit(y, predict)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    print(f"Variable: {feature_name}")
    # print(logistic_regression_model_fitted.summary())

    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    print(f"p_value = {p_value}")
    print(f"t_score = {t_value}")

    summary_dict["predictor"] = predictor
    summary_dict["p_value"] = p_value
    summary_dict["t_value"] = t_value

    print(summary_dict)
    return summary_dict


def main():
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

    path = create_resp_pred_plot_folder(dataset)
    response_type = return_column_type(df[response], "response")

    lst_summary_statistics = []

    for predictor in predictors:
        predictor_type = return_column_type(df[predictor], "predictor")

        if response_type == "boolean":
            if predictor_type == "categorical":
                cat_response_cat_predictor(df, predictor, response, path)
            elif predictor_type == "continuous":
                cat_resp_cont_predictor(df, predictor, response, path)
                lst_summary_statistics.append(
                    create_logistic_regression(df, predictor, response)
                )

        elif response_type == "continuous":
            if predictor_type == "categorical":
                cont_resp_cat_predictor(df, predictor, response, path)
            elif predictor_type == "continuous":
                cont_response_cont_predictor(df, predictor, response, path)
                lst_summary_statistics.append(
                    create_linear_regression(df, predictor, response)
                )

    df_summary_statistics = pd.DataFrame(
        lst_summary_statistics, columns=["predictor", "p_value", "t_value"]
    )
    print(df_summary_statistics)


if __name__ == "__main__":
    sys.exit(main())
