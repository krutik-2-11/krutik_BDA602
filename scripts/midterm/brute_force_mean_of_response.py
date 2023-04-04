import numpy as np
import pandas as pd
from plotly import graph_objects as go

DIFF_MEAN_RESP_RANKING = "diff_mean_resp_ranking"
DIFF_MEAN_RESP_WEIGHTED_RANKING = "diff_mean_resp_weighted_ranking"
BF_UNWEIGHTED = "bf_unweighted"
BF_WEIGHTED = "bf_weighted"
MEAN_SIZE = "mean_size"
LINK = "link"
CAT_1 = "cat_1"
CAT_2 = "cat_2"
CONT_1 = "cont_1"
CONT_2 = "cont_2"
CAT = "cat"
CONT = "cont"


class BruteForceMeanOfResponse:
    def brute_force_cat_cat_mean_of_response(
        self, df, cat_pred_1, cat_pred_2, response, path
    ):
        df_temp = df

        df_temp = (
            df_temp[[cat_pred_1, cat_pred_2, response]]
            .groupby([cat_pred_1, cat_pred_2])
            .agg(["mean", "size"])
            .reset_index()
        )

        df_temp.columns = df_temp.columns.to_flat_index().map("".join)

        response_mean = response + "mean"
        response_size = response + "size"

        df_temp[BF_UNWEIGHTED] = (
            df_temp[response_mean]
            .to_frame()
            .apply(lambda x: (df[response].mean() - x) ** 2)
        )

        df_temp[BF_WEIGHTED] = df_temp.apply(
            lambda x: (x[response_size] / df_temp[response_size].sum())
            * x[BF_UNWEIGHTED],
            axis=1,
        )

        df_temp[MEAN_SIZE] = df_temp.apply(
            lambda x: "{:.3f} (pop:{})".format(x[response_mean], x[response_size]),
            axis=1,
        )

        # Create the heatmap trace
        heatmap = go.Heatmap(
            x=np.array(df_temp[cat_pred_1]),
            y=np.array(df_temp[cat_pred_2]),
            z=np.array(df_temp[response_mean]),
            text=np.array(df_temp[MEAN_SIZE]),
            colorscale="Blues",
            texttemplate="%{text}",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title=f"{cat_pred_1} vs {cat_pred_2} (Bin Averages)",
            xaxis=dict(title=cat_pred_1),
            yaxis=dict(title=cat_pred_2),
        )

        fig = go.Figure(data=[heatmap], layout=layout)

        # fig.show()

        plot_link_brute_force = (
            f"{path}/{cat_pred_1}_vs_{cat_pred_2}_brute_force_heat_plot.html"
        )
        fig.write_html(
            file=plot_link_brute_force,
            include_plotlyjs="cdn",
        )

        summary_dict = {
            CAT_1: cat_pred_1,
            CAT_2: cat_pred_2,
            DIFF_MEAN_RESP_RANKING: df_temp[BF_UNWEIGHTED].sum()
            / (df[cat_pred_1].nunique() * df[cat_pred_2].nunique()),
            DIFF_MEAN_RESP_WEIGHTED_RANKING: df_temp[BF_WEIGHTED].sum(),
            LINK: plot_link_brute_force,
        }

        return summary_dict

    def brute_force_cont_cont_mean_of_response(
        self, df, cont_pred_1, cont_pred_2, response, path
    ):
        df_temp = df
        cont_pred_1_bins = cont_pred_1 + "_bins"
        cont_pred_2_bins = cont_pred_2 + "_bins"

        df_temp[cont_pred_1_bins] = (
            pd.cut(df_temp[cont_pred_1], bins=10, right=True)
        ).apply(lambda x: x.mid)
        df_temp[cont_pred_2_bins] = (
            pd.cut(df_temp[cont_pred_2], bins=10, right=True)
        ).apply(lambda x: x.mid)

        df_temp = (
            df_temp[
                [cont_pred_1, cont_pred_2, cont_pred_1_bins, cont_pred_2_bins, response]
            ]
            .groupby([cont_pred_1_bins, cont_pred_2_bins])
            .agg(["mean", "size"])
            .reset_index()
        )

        df_temp.columns = df_temp.columns.to_flat_index().map("".join)

        response_mean = response + "mean"
        response_size = response + "size"

        df_temp[BF_UNWEIGHTED] = (
            df_temp[response_mean]
            .to_frame()
            .apply(lambda x: (df[response].mean() - x) ** 2)
        )

        df_temp[BF_WEIGHTED] = df_temp.apply(
            lambda x: (x[response_size] / df_temp[response_size].sum())
            * x[BF_UNWEIGHTED],
            axis=1,
        )

        df_temp[MEAN_SIZE] = df_temp.apply(
            lambda x: "{:.3f} (pop:{})".format(x[response_mean], x[response_size]),
            axis=1,
        )

        # Create the heatmap trace
        heatmap = go.Heatmap(
            x=np.array(df_temp[cont_pred_1_bins]),
            y=np.array(df_temp[cont_pred_2_bins]),
            z=np.array(df_temp[response_mean]),
            text=np.array(df_temp[MEAN_SIZE]),
            colorscale="Blues",
            texttemplate="%{text}",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title=f"{cont_pred_1} vs {cont_pred_2} (Bin Averages)",
            xaxis=dict(title=cont_pred_1),
            yaxis=dict(title=cont_pred_2),
        )

        fig = go.Figure(data=[heatmap], layout=layout)

        # fig.show()

        plot_link_brute_force = (
            f"{path}/{cont_pred_1}_vs_{cont_pred_2}_brute_force_heat_plot.html"
        )
        fig.write_html(
            file=plot_link_brute_force,
            include_plotlyjs="cdn",
        )

        summary_dict = {
            CONT_1: cont_pred_1,
            CONT_2: cont_pred_2,
            DIFF_MEAN_RESP_RANKING: df_temp[BF_UNWEIGHTED].sum() / len(df_temp),
            DIFF_MEAN_RESP_WEIGHTED_RANKING: df_temp[BF_WEIGHTED].sum(),
            LINK: plot_link_brute_force,
        }

        return summary_dict

    def brute_force_cat_cont_mean_of_response(
        self, df, cat_pred, cont_pred, response, path
    ):
        df_temp = df
        cont_pred_bins = cont_pred + "_bins"

        df_temp[cont_pred_bins] = (
            pd.cut(df_temp[cont_pred], bins=10, right=True)
        ).apply(lambda x: x.mid)

        df_temp = (
            df_temp[[cat_pred, cont_pred, cont_pred_bins, response]]
            .groupby([cat_pred, cont_pred_bins])
            .agg(["mean", "size"])
            .reset_index()
        )

        df_temp.columns = df_temp.columns.to_flat_index().map("".join)

        response_mean = response + "mean"
        response_size = response + "size"

        df_temp[BF_UNWEIGHTED] = (
            df_temp[response_mean]
            .to_frame()
            .apply(lambda x: (df[response].mean() - x) ** 2)
        )

        df_temp[BF_WEIGHTED] = df_temp.apply(
            lambda x: (x[response_size] / df_temp[response_size].sum())
            * x[BF_UNWEIGHTED],
            axis=1,
        )

        df_temp[MEAN_SIZE] = df_temp.apply(
            lambda x: "{:.3f} (pop:{})".format(x[response_mean], x[response_size]),
            axis=1,
        )

        # Create the heatmap trace
        heatmap = go.Heatmap(
            x=np.array(df_temp[cat_pred]),
            y=np.array(df_temp[cont_pred_bins]),
            z=np.array(df_temp[response_mean]),
            text=np.array(df_temp[MEAN_SIZE]),
            colorscale="Blues",
            texttemplate="%{text}",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title=f"{cat_pred} vs {cont_pred} (Bin Averages)",
            xaxis=dict(title=cat_pred),
            yaxis=dict(title=cont_pred),
        )

        fig = go.Figure(data=[heatmap], layout=layout)

        # fig.show()

        plot_link_brute_force = (
            f"{path}/{cat_pred}_vs_{cont_pred}_brute_force_heat_plot.html"
        )
        fig.write_html(
            file=plot_link_brute_force,
            include_plotlyjs="cdn",
        )

        summary_dict = {
            CAT: cat_pred,
            CONT: cont_pred,
            DIFF_MEAN_RESP_RANKING: df_temp[BF_UNWEIGHTED].sum() / len(df_temp),
            DIFF_MEAN_RESP_WEIGHTED_RANKING: df_temp[BF_WEIGHTED].sum(),
            LINK: plot_link_brute_force,
        }

        return summary_dict
