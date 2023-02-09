import sys

import numpy as np
import pandas as pd
import plotly.express as px
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

"""


def ingest_data():
    data_source = pd.read_csv(
        "../data/iris.data",
        names=["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"],
    )
    return data_source


def quartiles(arr):
    return np.quantile(arr, 0.25), np.quantile(arr, 0.5), np.quantile(arr, 0.75)


def summary_statistic(arr, predictor):
    print(f"mean {predictor} = {np.mean(arr)}")
    print(f"min {predictor} = {np.min(arr)}")
    print(f"max {predictor} = {np.max(arr)}")
    print(f"Quartiles {predictor} = {quartiles(arr)}")


def create_plots(df):
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


def main():
    df = ingest_data()
    print(df)
    # printing summary statistic
    sepal_length_arr = np.array(df["sepal_len"])
    sepal_width_arr = np.array(df["sepal_wid"])
    petal_length_arr = np.array(df["petal_len"])
    petal_width_arr = np.array(df["petal_wid"])
    summary_statistic(sepal_length_arr, "Sepal Length")
    summary_statistic(sepal_width_arr, "Sepal Width")
    summary_statistic(petal_length_arr, "Petal Length")
    summary_statistic(petal_width_arr, "Petal Width")
    # Plotting the graphs
    create_plots(df)
    # Building the machine learning pipelines
    machine_learning_pipelines(df)


if __name__ == "__main__":
    sys.exit(main())
