import numpy as np
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go


class PredictorVsResponsePlots:
    def cont_response_cat_predictor(self, df, predictor, response, path):
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

        plot_link = f"{path}/cont_response_{response}_cat_predictor_{predictor}_violin_plot.html"
        fig_2.write_html(
            file=plot_link,
            include_plotlyjs="cdn",
        )

        summary_dict = {"predictor": predictor, "plot_link": plot_link}
        return summary_dict

    def cat_response_cont_predictor(self, df, predictor, response, path):
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

        plot_link = f"{path}/cont_response_{response}_cat_predictor_{predictor}_violin_plot.html"
        fig_2.write_html(
            file=plot_link,
            include_plotlyjs="cdn",
        )

        summary_dict = {"predictor": predictor, "plot_link": plot_link}
        return summary_dict

    def cat_response_cat_predictor(self, df, predictor, response, path):
        # Pivot the data set to create a frequency table
        pivoted_data = df.pivot_table(index=response, columns=predictor, aggfunc="size")

        # Create the heatmap trace
        heatmap = go.Heatmap(
            x=pivoted_data.columns,
            y=pivoted_data.index,
            z=pivoted_data,
            colorscale="Blues",
        )

        # Define the layout of the plot
        layout = go.Layout(
            title=f"Categorical Response {response} by Categorical Predictor {predictor}",
            xaxis=dict(title=predictor),
            yaxis=dict(title=response),
        )

        fig = go.Figure(data=[heatmap], layout=layout)

        # fig.show()

        plot_link = (
            f"{path}/cat_response_{response}_cat_predictor_{predictor}_heat_plot.html"
        )
        fig.write_html(
            file=plot_link,
            include_plotlyjs="cdn",
        )

        summary_dict = {"predictor": predictor, "plot_link": plot_link}
        return summary_dict

    def cont_response_cont_predictor(self, df, predictor, response, path):
        x = df[predictor]
        y = df[response]

        fig = px.scatter(x=x, y=y, trendline="ols")
        fig.update_layout(
            title=f"Continuous Response {response} by Continuous Predictor {predictor}",
            xaxis_title="Predictor",
            yaxis_title="Response",
        )
        # fig.show()

        plot_link = f"{path}/cont_response_{response}_cont_predictor_{predictor}_scatter_plot.html"
        fig.write_html(
            file=plot_link,
            include_plotlyjs="cdn",
        )

        summary_dict = {"predictor": predictor, "plot_link": plot_link}
        return summary_dict
