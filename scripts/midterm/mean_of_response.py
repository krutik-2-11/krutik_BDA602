import globals as gb
import numpy as np
import pandas as pd
from plotly import graph_objects as go


class MeanOfResponse:
    # Mean of response plot for continuous predictor & categorical response
    def mean_of_response_cont_pred_cat_resp(self, df, predictor, response, path):

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
        weighted_mean_squared_diff = np.nansum(weighted_squared_diff)

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

        plot_link_mor = f"{path}/{response}_VS_{predictor}_MOR.html"

        fig.write_html(
            file=plot_link_mor,
            include_plotlyjs="cdn",
        )

        summary_dict = {
            gb.PREDICTOR: predictor,
            gb.MEAN_SQUARED_DIFF: mean_squared_diff,
            gb.WEIGHTED_MEAN_SQUARED_DIFF: weighted_mean_squared_diff,
            gb.PLOT_LINK_MOR: plot_link_mor,
        }

        return summary_dict

    # Mean of response plot for categorical predictor & categorical response
    def mean_of_response_cat_pred_cat_resp(self, df, predictor, response, path):

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
        weighted_mean_squared_diff = np.nansum(weighted_squared_diff)

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

        plot_link_mor = f"{path}/{response}_VS_{predictor}_MOR.html"
        fig.write_html(
            file=plot_link_mor,
            include_plotlyjs="cdn",
        )

        summary_dict = {
            gb.PREDICTOR: predictor,
            gb.MEAN_SQUARED_DIFF: mean_squared_diff,
            gb.WEIGHTED_MEAN_SQUARED_DIFF: weighted_mean_squared_diff,
            gb.PLOT_LINK_MOR: plot_link_mor,
        }

        return summary_dict

    # Mean of response plot for continuous predictor & continuous response
    def mean_of_response_cont_pred_cont_resp(self, df, predictor, response, path):

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
        weighted_mean_squared_diff = np.nansum(weighted_squared_diff)

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

        plot_link_mor = f"{path}/{response}_VS_{predictor}_MOR.html"

        fig.write_html(
            file=plot_link_mor,
            include_plotlyjs="cdn",
        )

        summary_dict = {
            gb.PREDICTOR: predictor,
            gb.MEAN_SQUARED_DIFF: mean_squared_diff,
            gb.WEIGHTED_MEAN_SQUARED_DIFF: weighted_mean_squared_diff,
            gb.PLOT_LINK_MOR: plot_link_mor,
        }

        return summary_dict

    # Mean of response plot for categorical predictor & continuous response
    def mean_of_response_cat_pred_cont_resp(self, df, predictor, response, path):
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
        weighted_mean_squared_diff = np.nansum(weighted_squared_diff)

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

        plot_link_mor = f"{path}/{response}_VS_{predictor}_MOR.html"

        fig.write_html(
            file=plot_link_mor,
            include_plotlyjs="cdn",
        )

        summary_dict = {
            gb.PREDICTOR: predictor,
            gb.MEAN_SQUARED_DIFF: mean_squared_diff,
            gb.WEIGHTED_MEAN_SQUARED_DIFF: weighted_mean_squared_diff,
            gb.PLOT_LINK_MOR: plot_link_mor,
        }

        return summary_dict
