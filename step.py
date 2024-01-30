# TODO: fix the mess of imports
from operator import index

from dice_ml import Data
import data

from models.model_interface import ModelInterface
from models.pytorch_wrapper import PyTorchModel
from models.pytorch_models.dnn_basic import BaselineDNN
from models.pytorch_models.logreg import LogisticRegression as LogisticRegressionPT
from sklearn.linear_model import LogisticRegression


from data.data_interface import DataInterface
from data.synthetic_data import create_synthetic_data
import pandas as pd
from recourse.recourse_interface import RecourseInterface
from recourse.step_lib import StEP, StEPRecourse
from typing import Optional
import metrics.recourse_metrics as recourse_metrics
import metrics.classifier_metrics as classifier_metrics
import numpy as np
import time

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.stats as st

# binary should be full pipeline data->model->recourse->eval


def main(data_interface: DataInterface, model_choice="BaselineDNN"):
    print("-------------")
    start = time.time()
    print(data_interface.get_data()["EDUCATION"].value_counts())
    print(data_interface.get_data()["SEX"].value_counts())
    print(data_interface.get_data().columns)
    data_interface.encode_data()
    data_interface.scale_data("Standard")
    print(data_interface.get_data()["EDUCATION"].value_counts())
    print(data_interface.get_data().columns)
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data()
    if model_choice == "BaselineDNN":
        model = PyTorchModel(BaselineDNN(feats_train.shape[1]), validation_features=feats_valid,
                             validation_labels=labels_valid, batch_size=1, epochs=10, lr=0.002)
    elif model_choice == "LogisticRegressionPT":
        model = PyTorchModel(LogisticRegressionPT(feats_train.shape[1]), validation_features=feats_valid,
                             validation_labels=labels_valid, batch_size=1, epochs=10, lr=0.002)
    elif model_choice == "LogisticRegressionSK":
        model = LogisticRegression(class_weight="balanced")
    else:
        raise Exception("Invalid model choice")
    mi = ModelInterface(model)
    mi.fit(feats_train, labels_train)
    for score in classifier_metrics.run_classifier_tests(labels_test, mi.predict(feats_test)):
        print(score)

    start1 = time.time()
    num_clusters = 3
    max_iter = 50
    step_rec = StEPRecourse(mi, data_interface, num_clusters, max_iter, confidence_threshold=0.7,
                            directions_rescaler="constant step size", step_size=1.0)
    preds = pd.Series(mi.predict(feats_test), index=feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index].head(100)

    recourse_results = neg_data.apply(
        lambda row: generate_recourse_results(row, step_rec, mi), axis=1)
    df_results = pd.DataFrame.from_dict(
        dict(zip(recourse_results.index, recourse_results.values))).T
    df_results.rename(columns={0: "l2_path_len", 1: "l2_prox", 2: "l2_path_steps",
                      3: "poi_id", 4: "failures", 5: "diversity"}, inplace=True)
    df_results = df_results.explode(df_results.columns.values.tolist())
    # print(df_results.groupby('poi_id').mean()[['l2_path_len', 'l2_prox', 'l2_path_steps']])
    end = time.time()
    print("Model training took", start1-start, "seconds")
    print("Recourse took", end-start1, "seconds")
    print("Total it took", end-start, "seconds")
    return (df_results.mean(axis=0))

    end = time.time()
    print("Model training took", start1-start, "seconds")
    print("Recourse took", end-start1, "seconds")
    print("Total it took", end-start, "seconds")
    sns.histplot(df_results, x='l2_prox')
    plt.show()


def generate_recourse_results(poi, recourse_interface, model_interface):
    poi = poi.to_frame().T
    paths = recourse_interface.get_paths(poi)
    cfs = recourse_interface.get_counterfactuals_from_paths(paths)
    """print(cfs)
    print("________________________")"""
    

    l2_path_len = []
    l2_prox = []
    l2_path_steps = []
    failures = []
    """print(poi)"""
    for i, p in enumerate(paths):
        """print("________________________")
        print(p)"""
        if model_interface.predict(p[-1]) == 1:
            l2_path_len.append(recourse_metrics.compute_norm_path(p, 2))
            l2_prox.append(recourse_metrics.compute_norm(poi, p[-1], ord=2))
            l2_path_steps.append(len(p[1:]))
            failures.append(0)
        else:
            failures.append(1)
    return [l2_path_len, l2_prox, l2_path_steps,
            list(poi.index.values)*len(paths), failures,
            [recourse_metrics.compute_diversity(cfs)]*len(paths)]


if __name__ == "__main__":
    """df = create_synthetic_data(5000)
    cols = list(df.columns)
    label_column = cols[-1]
    continuous_features = cols[:3]
    ordinal_features = [cols[4]]
    categorical_features = cols[3:5]
    immutable_features =  [cols[3]]
    main(df, [])"""

    continuous_features = ["LIMIT_BAL", "AGE"]+[f"PAY_{i}" for i in range(
        1, 7)]+[f"BILL_AMT{i}" for i in range(1, 7)]+[f"PAY_AMT{i}" for i in range(1, 7)]
    ordinal_features = ["EDUCATION"]
    ordinal_features_order = {"EDUCATION": [5,4,3, 2, 1]}
    unidirection_features = [[], ["EDUCATION"]]
    categorical_features = ["SEX", "MARRIAGE","EDUCATION"]
    immutable_features = ["SEX", "MARRIAGE","AGE","EDUCATION"]
    label_column = "default payment next month"
    positive_label = 0
    file_path = (
        "data/datasets/default of credit card clients.xls")
    df = pd.read_excel(file_path, header=1)
    print(df["SEX"].value_counts())
    df.loc[df['EDUCATION'].isin([0,6]), "EDUCATION"] = 5
    df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)

    for pay_column in [f"PAY_{i}" for i in range(1, 7)]:
        df.loc[df[pay_column] < 0, pay_column] = 0
    print(df.head(10))
    print(df[label_column].value_counts())
    print(df.head(10))
    print(df["EDUCATION"].value_counts())
    print(df["SEX"].value_counts())
    print(df[df.isnull().any(axis=1)])
    """df_minor = df[df[label_column]==1]
    df_major = df[df[label_column]==0].sample(len(df_minor))
    df = pd.concat([df_minor, df_major])
    print(df.head(10))
    print(df[label_column].value_counts())"""

    num_trails = 1
    results = []
    for i in range(num_trails):
        di = DataInterface(df, None, continuous_features, ordinal_features,
                           categorical_features, immutable_features, label_column,
                           pos_label=positive_label, file_header_row=1, dropped_columns=["ID"], 
                           unidirection_features=unidirection_features, ordinal_features_order=ordinal_features_order)
        results.append(
            main(di, model_choice="LogisticRegressionSK").to_frame().T)
    print(results)
    results = pd.concat(results, ignore_index=True)
    print(results)
    print(results.mean(axis=0))
