import numpy as np
import pandas as pd
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
    data_source = pd.read_csv("../data/iris.data")
    data_source.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    return data_source


def mean(arr):
    return np.mean(arr)


def min(arr):
    return np.min(arr)


def max(arr):
    return np.max(arr)


def quartiles(arr):
    return np.quantile(arr, 0.25), np.quantile(arr, 0.5), np.quantile(arr, 0.75)


if __name__ == "__main__":
    df = ingest_data()
    print(df)

    sepal_length_arr = np.array(df["sepal_len"])
    sepal_width_arr = np.array(df["sepal_wid"])
    petal_length_arr = np.array(df["petal_len"])
    petal_width_arr = np.array(df["petal_wid"])

    # # Printing the summary statistic mean
    # print("mean sepal length = ", mean(sepal_length_arr))
    # print("mean sepal width = ", mean(sepal_width_arr))
    # print("mean petal length = ", mean(petal_length_arr))
    # print("mean petal width = ", mean(petal_width_arr))
    #
    # # Printing the summary statistic min
    # print("min sepal length = ", min(sepal_length_arr))
    # print("min sepal width = ", min(sepal_width_arr))
    # print("min petal length = ", min(petal_length_arr))
    # print("min petal width = ", min(petal_width_arr))
    #
    # # Printing the summary statistic max
    # print("max sepal length = ", max(sepal_length_arr))
    # print("max sepal width = ", max(sepal_width_arr))
    # print("max petal length = ", max(petal_length_arr))
    # print("max petal width = ", max(petal_width_arr))
    #
    # # Printing the summary statistic quartiles
    # print("quartiles sepal length = ", quartiles(sepal_length_arr))
    # print("quartiles sepal width = ", quartiles(sepal_width_arr))
    # print("quartiles petal length = ", quartiles(petal_length_arr))
    # print("quartiles petal width = ", quartiles(petal_width_arr))
    #
    # print("Correlation = ")
    # print(df.corr())
    #
    # # Histogram Distribution Plot for Sepal Length
    # fig = px.histogram(
    #     df,
    #     x="sepal_len",
    #     color="class",
    #     marginal="violin",
    #     labels={"sepal_len": "Sepal Length (cm)", "class": "Species of Iris"},
    #     title="Sepal Length Distribution for different species",
    # )
    # fig.show()
    #
    # # Histogram Distribution Plot for Sepal Width
    # fig = px.histogram(
    #     df,
    #     x="sepal_wid",
    #     color="class",
    #     marginal="violin",
    #     labels={"sepal_wid": "Sepal Width (cm)", "class": "Species of Iris"},
    #     title="Sepal Width Distribution for different species",
    # )
    # fig.show()
    #
    # # Histogram Distribution Plot for Petal Length
    # fig = px.histogram(
    #     df,
    #     x="petal_len",
    #     color="class",
    #     marginal="violin",
    #     labels={"petal_len": "Petal Length (cm)", "class": "Species of Iris"},
    #     title="Petal Length Distribution for different species",
    # )
    # fig.show()
    #
    # # Histogram Distribution Plot for Petal Width
    # fig = px.histogram(
    #     df,
    #     x="petal_wid",
    #     color="class",
    #     marginal="violin",
    #     labels={"petal_wid": "Petal Width (cm)", "class": "Species of Iris"},
    #     title="Petal Width Distribution for different species",
    # )
    # fig.show()
    #
    # # Matrix Plot
    # fig = px.scatter_matrix(
    #     df,
    #     dimensions=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
    #     color="class",
    #     symbol="class",
    #     title="Scatter matrix of iris data set",
    #     labels={col: col.replace("_", " ") for col in df.columns},
    # )  # remove underscore
    # fig.show()

    # Building the Machine Learning Models

    # Using the StandardScaler Transformer

    # data = [sepal_length_arr, sepal_width_arr, petal_length_arr, petal_width_arr]
    X = df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]]

    # scaler = StandardScaler()
    # print(scaler.fit(data))
    # print(scaler.mean_)
    #
    # print("Transformed Data")
    #
    # transformed_data = scaler.transform(data)
    # print(transformed_data)

    # Fitting the transformed data against random forest classifier
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


"""

    # Create a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: Random Forest Classifier", metrics.accuracy_score(y_test, y_pred))


    # Logistic Regression Classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: Logistic Regression Classifier", metrics.accuracy_score(y_test, y_pred))

    #K Nearest Neighbour Classifier
    clf = KNeighborsClassifier(n_neighbors=8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: K Nearest Neighbour Classifier", metrics.accuracy_score(y_test, y_pred))

    # Support Vector Classifier
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy: Support Vector Classifier", metrics.accuracy_score(y_test, y_pred))
"""
