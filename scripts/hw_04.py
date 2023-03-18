import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api
from data_loader import TestDatasets
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go

"""
A lot of source code has been used from Dr. Julien's lecture materials. This includes:
Linear Regression, Logistic Regression p_value and t_value calculations
Plots for cont response vs cont predictors
Plots for cont response vs cat predictors
Plots for cat response vs cont predictors
Plots for cat response vs cat predictors
https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/4/2
https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/4/4
https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/4/5
https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1

"""

PATH_RESP_PRED = "response_predictor_plots"
PATH_MEAN_OF_RESPONSE = "mean_of_response_plots"


def cont_resp_cat_predictor(df, predictor, response, path):
    group_labels = df[predictor].unique()
    hist_data = []

    for label in group_labels:
        hist_data.append(np.array(df[df[predictor] == label][response]))

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
                x=np.repeat(curr_group, len(curr_group)),
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
        hist_data.append(np.array(df[df[response].astype(str) == label][predictor]))

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
                x=np.repeat(curr_group, len(curr_hist)),
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


def create_mean_of_response_plot_folder(dataset):
    path = f"../{PATH_MEAN_OF_RESPONSE}"
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

    predict = statsmodels.api.add_constant(X)
    linear_regression_model = statsmodels.api.OLS(y, predict)
    linear_regression_model_fitted = linear_regression_model.fit()

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    summary_dict["predictor"] = predictor
    summary_dict["p_value"] = p_value
    summary_dict["t_value"] = t_value

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

    summary_dict["predictor"] = predictor
    summary_dict["p_value"] = p_value
    summary_dict["t_value"] = t_value

    return summary_dict


# Mean of response plot for continuous predictor & categorical response
def mean_of_response_cont_pred_cat_resp(df, predictor, response, path):

    bin_count = 10

    # creating the array of predictors for the overall population
    pred_vals_total = np.array(df[predictor])

    # using np.histogram function to get the bins and population in each bin (for bar plot)
    population, bins = np.histogram(
        pred_vals_total,
        bins=bin_count,
        range=(np.min(pred_vals_total), np.max(pred_vals_total)),
    )

    # taking the average of bins to get the midpoints
    bins_mod = 0.5 * (bins[:-1] + bins[1:])

    # getting the predictor values for boolean value "True"
    df_true = df.loc[df[response] == bool(1)]

    # creating the array of predictor values for boolean value "True"
    pred_vals_true = np.array(df_true[predictor])

    # getting the count of predictor values in each bin corresponding to boolean value "True"
    population_true, _ = np.histogram(pred_vals_true, bins=bins)

    # getting the response ratio in each bin
    population_response = population_true / population

    # overall mean response value for "true" responses in the entire population
    true_value_response_rate = len(df_true) / len(df)

    # creating the array of "true" class mean response for entire population to build the plot
    true_class_response_rate_population_arr = np.array(
        [true_value_response_rate] * bin_count
    )

    # since the response is boolean, so bin_mean is (number of positives in bin)/(size of bin)
    # which is equal to population_response calculated above
    bin_mean = population_response
    population_mean = true_class_response_rate_population_arr

    # calculating mean squared difference
    squared_diff = (bin_mean - population_mean) ** 2
    mean_squared_diff = np.nansum(squared_diff) / bin_count

    # Calculating the weighted mean squared difference
    bin_population_ratio = population / np.nansum(population)
    weighted_squared_diff = squared_diff * bin_population_ratio
    weighted_mean_squared_diff = np.nansum(weighted_squared_diff) / bin_count

    # bar plot for overall population
    fig = go.Figure(
        data=go.Bar(
            x=bins_mod,
            y=population,
            name=predictor,
            marker=dict(color="blue"),
        )
    )

    # scatter plot for mean of response for a class within each bin
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=population_response,
            yaxis="y2",
            name="Response",
            marker=dict(color="red"),
        )
    )

    # scatter plot for mean of response for the "true" value in entire population
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=true_class_response_rate_population_arr,
            yaxis="y2",
            mode="lines",
            name="Population Overall Response",
        )
    )

    fig.update_layout(
        title_text=response + " vs " + predictor + " Mean of Response Rate Plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Frequency in Each Bin"),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")

    # fig.show()

    # Saving the plots into an HTML file with dynamic path

    fig.write_html(
        file=f"{path}/{response}_VS_{predictor}_MOR.html",
        include_plotlyjs="cdn",
    )
    summary_dict = {}
    summary_dict["predictor"] = predictor
    summary_dict["mean_squared_diff"] = mean_squared_diff
    summary_dict["weighted_mean_squared_diff"] = weighted_mean_squared_diff

    return summary_dict


# Mean of response plot for categorical predictor & categorical response
def mean_of_response_cat_pred_cat_resp(df, predictor, response, path):

    # Here bins will be our categories
    bins_mod = df[predictor].unique()
    bin_count = len(bins_mod)

    # counting the total population and boolean True population in each bin_mod or each predictor category
    total_population_bins_mod = []
    total_true_population_bins_mod = []
    for category in bins_mod:
        df_temp_population = df.loc[df[predictor] == category]
        df_temp_population_true = df_temp_population.loc[
            df_temp_population[response] == bool(1)
        ]
        population_count = len(df_temp_population)
        true_population_count = len(df_temp_population_true)
        total_population_bins_mod.append(population_count)
        total_true_population_bins_mod.append(true_population_count)

    total_population_bins_mod = np.array(total_population_bins_mod)
    total_true_population_bins_mod = np.array(total_true_population_bins_mod)

    population_response = total_true_population_bins_mod / total_population_bins_mod

    # getting the predictor values for boolean value "True"
    df_true = df.loc[df[response] == bool(1)]
    true_value_response_rate = len(df_true) / len(df)

    # creating the array of "true" class mean response for entire population to build the plot
    true_class_response_rate_population_arr = np.array(
        [true_value_response_rate] * bin_count
    )

    # since the response is boolean, so bin_mean is (number of positives in bin)/(size of bin) which is
    # equal to population_response calculated above
    bin_mean = population_response
    population_mean = true_class_response_rate_population_arr

    # calculating mean squared difference
    squared_diff = (bin_mean - population_mean) ** 2
    mean_squared_diff = np.nansum(squared_diff) / bin_count

    # Calculating the weighted mean squared difference
    bin_population_ratio = total_population_bins_mod / np.nansum(
        total_population_bins_mod
    )
    weighted_squared_diff = squared_diff * bin_population_ratio
    weighted_mean_squared_diff = np.nansum(weighted_squared_diff) / bin_count

    # bar plot for overall population
    fig = go.Figure(
        data=go.Bar(
            x=bins_mod,
            y=total_population_bins_mod,
            name=predictor,
            marker=dict(color="blue"),
        )
    )

    # scatter plot for mean of response for a class within each bin
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=population_response,
            yaxis="y2",
            name="Response",
            marker=dict(color="red"),
        )
    )

    # scatter plot for mean of response for the "true" value in entire population
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=true_class_response_rate_population_arr,
            yaxis="y2",
            mode="lines",
            name="Overall Population Response",
        )
    )

    fig.update_layout(
        title_text=response + " vs " + predictor + " Mean of Response Rate Plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Frequency in Each Bin"),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")

    # fig.show()

    # Saving the plots into an HTML file with dynamic path

    fig.write_html(
        file=f"{path}/{response}_VS_{predictor}_MOR.html",
        include_plotlyjs="cdn",
    )

    summary_dict = {}
    summary_dict["predictor"] = predictor
    summary_dict["mean_squared_diff"] = mean_squared_diff
    summary_dict["weighted_mean_squared_diff"] = weighted_mean_squared_diff

    return summary_dict


# Mean of response plot for continuous predictor & continuous response
def mean_of_response_cont_pred_cont_resp(df, predictor, response, path):

    bin_count = 10
    bins = pd.cut(df[predictor], bin_count)
    bins_mod = bins.apply(lambda x: x.mid)
    bins_mod = np.array(bins_mod.unique())
    bins_mod = np.sort(bins_mod)
    result = df.groupby(bins)[response].agg(["sum", "count"])

    total_population = np.sum(result["count"])
    total_sum = np.sum(result["sum"])
    bin_population = np.array(result["count"])
    bin_sum = np.array(result["sum"])

    bin_response = bin_sum / bin_population
    population_response = total_sum / total_population

    population_response_arr = np.array([population_response] * bin_count)

    squared_diff = (bin_response - population_response_arr) ** 2
    mean_squared_diff = np.nansum(squared_diff) / bin_count

    bin_population_ratio = bin_population / total_population
    weighted_squared_diff = squared_diff * bin_population_ratio
    weighted_mean_squared_diff = np.nansum(weighted_squared_diff) / bin_count

    # bar plot for overall population
    fig = go.Figure(
        data=go.Bar(
            x=bins_mod,
            y=bin_population,
            name=predictor,
            marker=dict(color="blue"),
        )
    )

    # scatter plot for mean of response for a class within each bin
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=bin_response,
            yaxis="y2",
            name="Response",
            marker=dict(color="red"),
        )
    )

    # scatter plot for mean of response for the "true" value in entire population
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=population_response_arr,
            yaxis="y2",
            mode="lines",
            name="Population Overall Response",
        )
    )

    fig.update_layout(
        title_text=response + " vs " + predictor + " Mean of Response Rate Plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Frequency in Each Bin"),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")

    # fig.show()

    # Saving the plots into an HTML file with dynamic path

    fig.write_html(
        file=f"{path}/{response}_VS_{predictor}_MOR.html",
        include_plotlyjs="cdn",
    )

    summary_dict = {}
    summary_dict["predictor"] = predictor
    summary_dict["mean_squared_diff"] = mean_squared_diff
    summary_dict["weighted_mean_squared_diff"] = weighted_mean_squared_diff

    return summary_dict


# Mean of response plot for categorical predictor & continuous response
def mean_of_response_cat_pred_cont_resp(df, predictor, response, path):
    # Here bins will be our categories
    bins_mod = df[predictor].unique()
    bin_count = len(bins_mod)

    bin_population = []
    bin_sum_values = []
    for category in bins_mod:
        df_temp = df.loc[df[predictor] == category]
        bin_population.append(len(df_temp))
        bin_sum_values.append(df_temp[response].sum())

    bin_population = np.array(bin_population)
    bin_sum_values = np.array(bin_sum_values)

    total_population = len(df)
    total_sum = df[response].sum()
    bin_response = bin_sum_values / bin_population
    population_response = total_sum / total_population
    population_response_arr = np.array([population_response] * bin_count)

    squared_diff = (bin_response - population_response_arr) ** 2
    mean_squared_diff = np.nansum(squared_diff) / bin_count

    bin_population_ratio = bin_population / total_population
    weighted_squared_diff = squared_diff * bin_population_ratio
    weighted_mean_squared_diff = np.nansum(weighted_squared_diff) / bin_count

    # bar plot for overall population
    fig = go.Figure(
        data=go.Bar(
            x=bins_mod,
            y=bin_population,
            name=predictor,
            marker=dict(color="blue"),
        )
    )

    # scatter plot for mean of response for a class within each bin
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=bin_response,
            yaxis="y2",
            name="Response",
            marker=dict(color="red"),
        )
    )

    # scatter plot for mean of response for the "true" value in entire population
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=population_response_arr,
            yaxis="y2",
            mode="lines",
            name="Population Overall Response",
        )
    )

    fig.update_layout(
        title_text=response + " vs " + predictor + " Mean of Response Rate Plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Frequency in Each Bin"),
            side="left",
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")

    # fig.show()

    # Saving the plots into an HTML file with dynamic path

    fig.write_html(
        file=f"{path}/{response}_VS_{predictor}_MOR.html",
        include_plotlyjs="cdn",
    )

    summary_dict = {}
    summary_dict["predictor"] = predictor
    summary_dict["mean_squared_diff"] = mean_squared_diff
    summary_dict["weighted_mean_squared_diff"] = weighted_mean_squared_diff

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
    path_mean_of_response = create_mean_of_response_plot_folder(dataset)
    response_type = return_column_type(df[response], "response")

    lst_summary_statistics_p_t_value = []
    lst_summary_statistics_mean_of_response = []

    for predictor in predictors:
        predictor_type = return_column_type(df[predictor], "predictor")

        if response_type == "boolean":
            if predictor_type == "categorical":
                cat_response_cat_predictor(df, predictor, response, path)
                lst_summary_statistics_mean_of_response.append(
                    mean_of_response_cat_pred_cat_resp(
                        df, predictor, response, path_mean_of_response
                    )
                )

            elif predictor_type == "continuous":
                cat_resp_cont_predictor(df, predictor, response, path)
                lst_summary_statistics_p_t_value.append(
                    create_logistic_regression(df, predictor, response)
                )
                lst_summary_statistics_mean_of_response.append(
                    mean_of_response_cont_pred_cat_resp(
                        df, predictor, response, path_mean_of_response
                    )
                )

        elif response_type == "continuous":
            if predictor_type == "categorical":
                cont_resp_cat_predictor(df, predictor, response, path)

                lst_summary_statistics_mean_of_response.append(
                    mean_of_response_cat_pred_cont_resp(
                        df, predictor, response, path_mean_of_response
                    )
                )

            elif predictor_type == "continuous":
                cont_response_cont_predictor(df, predictor, response, path)
                lst_summary_statistics_p_t_value.append(
                    create_linear_regression(df, predictor, response)
                )
                lst_summary_statistics_mean_of_response.append(
                    mean_of_response_cont_pred_cont_resp(
                        df, predictor, response, path_mean_of_response
                    )
                )

    df_summary_statistics_p_t_value = pd.DataFrame(
        lst_summary_statistics_p_t_value, columns=["predictor", "p_value", "t_value"]
    )

    # Order the results in increasing order of p_value and decreasing order of t_value
    df_summary_statistics_p_t_value.sort_values(
        by=["t_value", "p_value"], ascending=[False, True], inplace=True
    )
    print("*" * 50)
    print(
        f"****************Summary Statistics of Continuous Predictors of dataset {dataset}**********************"
    )
    print(df_summary_statistics_p_t_value)
    print("*" * 50)

    df_summary_statistics_mean_of_response = pd.DataFrame(
        lst_summary_statistics_mean_of_response,
        columns=["predictor", "weighted_mean_squared_diff", "mean_squared_diff"],
    )

    # Order the results in decreasing order of weighted_mean_Squared_error and mean_squared_error
    df_summary_statistics_mean_of_response.sort_values(
        by=["weighted_mean_squared_diff", "mean_squared_diff"],
        ascending=[False, False],
        inplace=True,
    )

    print("*" * 50)
    print(
        f"****************Summary Statistics of Mean of Response of dataset {dataset}**********************"
    )
    print(df_summary_statistics_mean_of_response)
    print("*" * 50)


if __name__ == "__main__":
    sys.exit(main())
