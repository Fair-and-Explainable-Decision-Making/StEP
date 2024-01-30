from data import data_interface
from models import model_interface
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
import multiprocessing
from data.dataset_utils import get_dataset_interface_by_name
from models.model_utils import get_model_interface_by_name
from recourse.recourse_utils import get_recourse_interface_by_name


def run_experiments(arguments):
    data_interface = get_dataset_interface_by_name(arguments["dataset name"])
    if arguments["dataset encoded"]:
        data_interface.encode_data()
    if arguments["dataset scaler"]:
        data_interface.scale_data(arguments["dataset scaler"])
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data(
        arguments["dataset valid-test split"][0], arguments["dataset valid-test split"][1])
    model_interface = get_model_interface_by_name(arguments["model name"], feats_train=feats_train, feats_valid=feats_valid, labels_valid=labels_valid,
                                                  batch_size=arguments["model batch size"],  epochs=arguments["model epochs"],  lr=arguments["model learning rate"])
    model_interface.fit(feats_train, labels_train)
    preds = pd.Series(model_interface.predict(feats_test), index=feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index]
    for score in classifier_metrics.run_classifier_tests(labels_test, model_interface.predict(feats_test)):
        print(score)

    recourse_results = {}
    for recourse_name, recourse_args in arguments["recourse methods"].items():
        recourse_interface = get_recourse_interface_by_name(recourse_name, model_interface, data_interface, recourse_args)
        recourse_results = neg_data.apply(
            lambda row: generate_recourse_results(row, recourse_interface, model_interface, data_interface), axis=1)
        df_results = pd.DataFrame.from_dict(
            dict(zip(recourse_results.index, recourse_results.values))).T
        df_results.rename(columns={0: "l2_path_len", 1: "l2_prox", 2: "l2_path_steps",
                        3: "poi_id", 4: "failures", 5: "diversity"}, inplace=True)
        df_results = df_results.explode(df_results.columns.values.tolist())
        recourse_results[recourse_name] = df_results
    # print(df_results.groupby('poi_id').mean()[['l2_path_len', 'l2_prox', 'l2_path_steps']])
    return recourse_results

def generate_recourse_results(poi, recourse_interface, model_interface):
    poi = poi.to_frame().T
    cfs = recourse_interface.get_counterfactuals(poi)
    paths = recourse_interface.get_paths(poi)

    l2_path_len = []
    l2_prox = []
    l2_path_steps = []
    failures = []
    for i, p in enumerate(paths):
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
    run_experiments()
    #use df.to_latex
