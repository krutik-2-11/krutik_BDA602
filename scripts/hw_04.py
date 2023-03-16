import random
import sys
from typing import List, Tuple

import numpy
import pandas
import seaborn
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn import datasets


class TestDatasets:
    def __init__(self):
        self.seaborn_data_sets = ["mpg", "tips", "titanic"]
        self.sklearn_data_sets = ["diabetes", "breast_cancer"]
        self.all_data_sets = self.seaborn_data_sets + self.sklearn_data_sets

    TITANIC_PREDICTORS = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "embarked",
        "parch",
        "fare",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
        "class",
    ]

    def get_all_available_datasets(self) -> List[str]:
        return self.all_data_sets

    def get_test_data_set(
        self, data_set_name: str = None
    ) -> Tuple[pandas.DataFrame, List[str], str]:
        """Function to load a few test data sets

        :param:
        data_set_name : string, optional
            Data set to load

        :return:
        data_set : :class:`pandas.DataFrame`
            Tabular data, possibly with some preprocessing applied.
        predictors :list[str]
            List of predictor variables
        response: str
            Response variable
        """

        if data_set_name is None:
            data_set_name = random.choice(self.all_data_sets)
        else:
            if data_set_name not in self.all_data_sets:
                raise Exception(f"Data set choice not valid: {data_set_name}")

        if data_set_name in self.seaborn_data_sets:
            if data_set_name == "mpg":
                data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
                predictors = [
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "origin",
                ]
                response = "mpg"
            elif data_set_name == "tips":
                data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
                predictors = [
                    "total_bill",
                    "sex",
                    "smoker",
                    "day",
                    "time",
                    "size",
                ]
                response = "tip"
            elif data_set_name in ["titanic", "titanic_2"]:
                data_set = seaborn.load_dataset(name="titanic").dropna()
                data_set["alone"] = data_set["alone"].astype(str)
                data_set["class"] = data_set["class"].astype(str)
                data_set["deck"] = data_set["deck"].astype(str)
                data_set["pclass"] = data_set["pclass"].astype(str)
                predictors = self.TITANIC_PREDICTORS
                if data_set_name == "titanic":
                    response = "survived"
                elif data_set_name == "titanic_2":
                    response = "alive"
        elif data_set_name in self.sklearn_data_sets:
            if data_set_name == "boston":
                data = datasets.load_boston()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
                data_set["CHAS"] = data_set["CHAS"].astype(str)
            elif data_set_name == "diabetes":
                data = datasets.load_diabetes()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            elif data_set_name == "breast_cancer":
                data = datasets.load_breast_cancer()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["target"] = data.target
            predictors = data.feature_names
            response = "target"

        # Change category dtype to string
        for predictor in predictors:
            if data_set[predictor].dtype in ["category"]:
                data_set[predictor] = data_set[predictor].astype(str)

        print(f"Data set selected: {data_set_name}")
        data_set.reset_index(drop=True, inplace=True)
        return data_set, predictors, response


def cont_resp_cat_predictor(df, predictor, response):
    group_labels = df[predictor].unique()
    hist_data = []

    for label in group_labels:
        hist_data.append(numpy.array(df[df[predictor] == label][response]))

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels)
    fig_1.update_layout(
        title=f"Continuous Response {response} by Categorical Predictor {predictor}",
        xaxis_title=f"Response {response}",
        yaxis_title="Distribution",
    )
    # fig_1.show()

    fig_1.write_html(
        file=f"../plots/lecture_6_cont_response_{response}_cat_predictor_{predictor}_dist_plot.html",
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
        file=f"../plots/lecture_6_cont_response_{response}_cat_predictor_{predictor}_violin_plot.html",
        include_plotlyjs="cdn",
    )

    return


def cat_resp_cont_predictor(df, predictor, response):
    group_labels = df[response].unique().astype(str)
    hist_data = []

    for label in group_labels:
        hist_data.append(numpy.array(df[df[response].astype(str) == label][predictor]))

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels)
    fig_1.update_layout(
        title=f"Continuous Response {response} by Categorical Predictor {predictor}",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    # fig_1.show()

    fig_1.write_html(
        file=f"../plots/lecture_6_cont_response_{response}_cat_predictor_{predictor}_dist_plot.html",
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
        file=f"../plots/lecture_6_cont_response_{response}_cat_predictor_{predictor}_violin_plot.html",
        include_plotlyjs="cdn",
    )

    return


def cat_response_cat_predictor(df, predictor, response):
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

    # Create the figure and add the trace
    fig = go.Figure(data=[heatmap], layout=layout)

    # Show the plot
    # fig.show()

    fig.write_html(
        file=f"../plots/lecture_6_cat_response_{response}_cat_predictor_{predictor}_violin_plot.html",
        include_plotlyjs="cdn",
    )

    return


def cont_response_cont_predictor(df, predictor, response):
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
        file=f"../plots/lecture_6_cont_response_{response}_cont_predictor_{predictor}_scatter_plot.html",
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
        if column.dtype == "object":
            return "categorical"
        else:
            return "continuous"


def main():
    test_datasets = TestDatasets()
    dataset_dict = {}
    for test in test_datasets.get_all_available_datasets():
        df, predictors, response = test_datasets.get_test_data_set(data_set_name=test)
        dataset_dict[test] = [df, predictors, response]

    df = dataset_dict["titanic"][0]
    predictors = dataset_dict["titanic"][1]
    response = dataset_dict["titanic"][2]

    response_type = return_column_type(df[response], "response")
    for predictor in predictors:
        predictor_type = return_column_type(df[predictor], "predictor")
        if response_type == "boolean" and predictor_type == "categorical":
            cat_response_cat_predictor(df, predictor, response)
        elif response_type == "boolean" and predictor_type == "continuous":
            cat_resp_cont_predictor(df, predictor, response)
        elif response_type == "continuous" and predictor_type == "categorical":
            cont_resp_cat_predictor(df, predictor, response)
        elif response_type == "continuous" and predictor_type == "continuous":
            cont_response_cont_predictor(df, predictor, response)


if __name__ == "__main__":
    sys.exit(main())
