import os.path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

"""
References:
https://www.kaggle.com/code/sixteenpython/machine-learning-with-iris-dataset
https://www.kaggle.com/code/tcvieira/simple-random-forest-iris-dataset/notebook
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
https://plotly.com/python/multiple-axes/

"""


def ingest_data(csv_url):
    data_source = pd.read_csv(
        csv_url,
        names=["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"],
    )

    return data_source


def quartiles(arr):
    # using quantile function of numpy to get the first, second (median) and third quartile
    return np.quantile(arr, 0.25), np.quantile(arr, 0.5), np.quantile(arr, 0.75)


def summary_statistic(arr, predictor):
    # getting mean, median, max and quantiles of predictors
    print(f"mean {predictor} = {np.mean(arr)}")
    print(f"min {predictor} = {np.min(arr)}")
    print(f"max {predictor} = {np.max(arr)}")
    print(f"Quartiles {predictor} = {quartiles(arr)}")


def create_plots(df):

    # Checking if the plots folder is already available, if not creating the one
    path = "../plots"
    path_exist = os.path.exists(path)

    if not path_exist:
        os.mkdir(path)

    # Histogram Distribution Plot for Sepal Length
    fig = px.histogram(
        df,
        x="sepal_len",
        color="class",
        marginal="violin",
        labels={"sepal_len": "Sepal Length (cm)", "class": "Species of Iris"},
        title="Sepal Length Distribution for different species",
    )
    fig.show()
    fig.write_html(
        file=path + "/SepalLengthHistogramViolin.html", include_plotlyjs="cdn"
    )

    # Histogram Distribution Plot for Sepal Width
    fig = px.histogram(
        df,
        x="sepal_wid",
        color="class",
        marginal="violin",
        labels={"sepal_wid": "Sepal Width (cm)", "class": "Species of Iris"},
        title="Sepal Width Distribution for different species",
    )
    fig.show()
    fig.write_html(
        file=path + "/SepalWidthHistogramViolin.html", include_plotlyjs="cdn"
    )

    # Histogram Distribution Plot for Petal Length
    fig = px.histogram(
        df,
        x="petal_len",
        color="class",
        marginal="violin",
        labels={"petal_len": "Petal Length (cm)", "class": "Species of Iris"},
        title="Petal Length Distribution for different species",
    )
    fig.show()
    fig.write_html(
        file=path + "/PetalLengthHistogramViolin.html", include_plotlyjs="cdn"
    )

    # Histogram Distribution Plot for Petal Width
    fig = px.histogram(
        df,
        x="petal_wid",
        color="class",
        marginal="violin",
        labels={"petal_wid": "Petal Width (cm)", "class": "Species of Iris"},
        title="Petal Width Distribution for different species",
    )
    fig.show()
    fig.write_html(
        file=path + "/PetalWidthHistogramViolin.html", include_plotlyjs="cdn"
    )

    # Matrix Plot
    fig = px.scatter_matrix(
        df,
        dimensions=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
        color="class",
        symbol="class",
        title="Scatter matrix of iris data set",
        labels={col: col.replace("_", " ") for col in df.columns},
    )  # remove underscore
    fig.show()
    fig.write_html(file=path + "/ScatterMatrix.html", include_plotlyjs="cdn")


def machine_learning_pipelines(df):
    # Building the Machine Learning Models
    X = df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]]
    y = df["class"]
    print("Independent Variables")
    print(X)
    print("Target Variable")
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )  # 70% training and 30% test
    # Creating Pipeline for transformation and model creation
    pipeline_SVC = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    pipeline_LR = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression())])
    pipeline_KNN = Pipeline(
        [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=8))]
    )
    pipeline_RFC = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rfc", RandomForestClassifier(n_estimators=100)),
        ]
    )
    pipelines = [pipeline_SVC, pipeline_LR, pipeline_KNN, pipeline_RFC]
    for pipe in pipelines:
        pipe.fit(X_train, y_train)
    for i, model in enumerate(pipelines):
        print("{} Test Accuracy: {}".format(pipelines[i], model.score(X_test, y_test)))


def make_mean_of_response_plots(predictor, species, iris):

    predictor_map = {
        "sepal_len": "Sepal Length",
        "sepal_wid": "Sepal Width",
        "petal_len": "Petal Length",
        "petal_wid": "Petal Width",
    }

    species_map = {
        "Iris-setosa": "Iris Setosa",
        "Iris-versicolor": "Iris Versicolor",
        "Iris-virginica": "Iris Virginica",
    }

    iris_predictor_species = iris[[predictor, "class"]]

    # creating the array of predictors for the overall population i.e., all three species
    a = np.array(iris_predictor_species[predictor])

    # using np.histogram function to get the bins and population in each bin (for bar plot)
    population, bins = np.histogram(a, bins=10, range=(np.min(a), np.max(a)))

    # taking the average of bins to get the midpoints
    bins_mod = 0.5 * (bins[:-1] + bins[1:])

    # getting the predictor values for a specific class
    iris_predictor_subgroup = iris_predictor_species.loc[
        iris_predictor_species["class"] == species
    ]
    # creating the array of predictor values for a specific class
    b = np.array(iris_predictor_subgroup[predictor])
    population_iris_class, _ = np.histogram(b, bins=bins)

    # getting the response value range is [0,1]
    population_response = population_iris_class / population

    # overall class mean response value; it is 0.33 as we have around 50 records for each class
    species_class_response_rate = len(iris.loc[iris["class"] == species]) / len(iris)

    # creating the array of class mean response to build the plot
    species_class_response_rate_arr = np.array(
        [species_class_response_rate] * len(bins_mod)
    )

    # bar plot for overall population
    fig = go.Figure(
        data=go.Bar(
            x=bins_mod,
            y=population,
            name=predictor_map[predictor],
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

    # scatter plot for mean of response for the class in entire population
    fig.add_trace(
        go.Scatter(
            x=bins_mod,
            y=species_class_response_rate_arr,
            yaxis="y2",
            mode="lines",
            name=species_map[species] + " Overall Response",
        )
    )

    fig.update_layout(
        title_text=species_map[species]
        + " vs "
        + predictor_map[predictor]
        + " Mean of Response Rate Plot",
        legend=dict(orientation="v"),
        yaxis=dict(
            title=dict(text="Frequency in Each Bin"),
            side="left",
            range=[0, 50],
        ),
        yaxis2=dict(
            title=dict(text="Response"),
            side="right",
            range=[-0.1, 1.2],
            overlaying="y",
            tickmode="auto",
        ),
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Predictor Bins")

    fig.show()

    # Saving the plots into an HTML file with dynamic path
    path = "../plots"
    path_exist = os.path.exists(path)

    if not path_exist:
        os.mkdir(path)

    fig.write_html(
        file=path + f"/{species_map[species]}vs{predictor_map[predictor]}_MOR.html",
        include_plotlyjs="cdn",
    )


def main():

    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = ingest_data(csv_url)

    sepal_length_arr = np.array(df["sepal_len"])
    sepal_width_arr = np.array(df["sepal_wid"])
    petal_length_arr = np.array(df["petal_len"])
    petal_width_arr = np.array(df["petal_wid"])

    # calling functions for summary statistic
    summary_statistic(sepal_length_arr, "Sepal Length")
    summary_statistic(sepal_width_arr, "Sepal Width")
    summary_statistic(petal_length_arr, "Petal Length")
    summary_statistic(petal_width_arr, "Petal Width")

    # Plotting the graphs
    create_plots(df)

    # Building the machine learning pipelines
    machine_learning_pipelines(df)

    # Create mean of response plots
    predictors = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    for specimen in species:
        for predictor in predictors:
            make_mean_of_response_plots(predictor, specimen, df)


if __name__ == "__main__":
    sys.exit(main())
