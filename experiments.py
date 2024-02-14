from data.data_interface import DataInterface
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
from recourse.utils.recourse_utils import get_recourse_interface_by_name
import joblib
from joblib import Parallel, delayed


def run_experiments_trials(arguments):
    start_trials = time.time()
    p_results = Parallel(n_jobs=5)(delayed(run_experiments_one_trial)(arguments,trial) for trial in range(arguments["trials"]))
    print(p_results)
    recourse_methods = list(p_results[0].keys())
    recourse_results_trials_dict = {}
    for recourse_method in recourse_methods:
        recourse_results_trials_dict[recourse_method] = []
        for result_i in p_results:
            recourse_results_trials_dict[recourse_method].append(result_i[recourse_method][0])
    print(recourse_results_trials_dict)
    agg_recourse_results = {}
    recourse_results = {}
    for recourse_name, results in recourse_results_trials_dict.items():
        results = pd.concat(results, ignore_index=True)
        recourse_results[recourse_name]=results
        results = results.mean(axis=0)
        agg_recourse_results[recourse_name]=results
    end_trials = time.time()
    print("All trials took", end_trials-start_trials, "seconds.")
    return agg_recourse_results, recourse_results

def run_experiments_one_trial(arguments,trial_num = 0):
    recourse_results_dict = {}
    start_trial = time.time()
    data_interface = get_dataset_interface_by_name(arguments["dataset name"])
    if arguments["dataset encoded"]:
        data_interface.encode_data()
    if arguments["dataset scaler"]:
        data_interface.scale_data(arguments["dataset scaler"])
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data(
        arguments["dataset valid-test split"][0], arguments["dataset valid-test split"][1],random_state=trial_num)
    model_args = arguments["base model"].copy()
    model_args["feats_train"], model_args["feats_valid"], model_args["labels_valid"] = feats_train, feats_valid, labels_valid
    model_interface = get_model_interface_by_name(**model_args)
    
    model_interface.fit(feats_train, labels_train)
    preds = pd.Series(model_interface.predict(feats_test), index=feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index]
    #TODO: change this to output to a text file or something similar
    for score in classifier_metrics.run_classifier_tests(labels_test, model_interface.predict(feats_test)):
        print(score)

    done_training_trial = time.time()
    print("Model training took", done_training_trial-start_trial, "seconds")
    for recourse_name, recourse_args in arguments["recourse methods"].items():
        recourse_args['random seed'] = trial_num
        start_recourse = time.time()
        if recourse_name not in recourse_results_dict:
            recourse_results_dict[recourse_name] = []
        recourse_interface = get_recourse_interface_by_name(recourse_name, model_interface, data_interface, **recourse_args)
        k_directions = recourse_args['k_directions']
        recourse_results = neg_data.apply(
            lambda row: generate_recourse_results(row, recourse_interface, model_interface,k_directions), axis=1)
        df_results = pd.DataFrame.from_dict(
            dict(zip(recourse_results.index, recourse_results.values))).T
        df_results.rename(columns={0: "l2_path_len", 1: "l2_prox", 2: "l2_path_steps",
                        3: "poi_id", 4: "failures", 5: "diversity"}, inplace=True)
        df_results = df_results.explode(df_results.columns.values.tolist())
        recourse_results_dict[recourse_name].append(df_results.mean(axis=0).to_frame().T)
        end_recourse = time.time()
        print(recourse_name, "recourse took", end_recourse-start_recourse, "seconds for", len(neg_data), "samples.")
    end_trial = time.time()
    print("Trial", trial_num,"took", end_trial-start_trial, "seconds.")
    return recourse_results_dict
    

def generate_recourse_results(poi, recourse_interface, model_interface,k_directions):
    poi = poi.to_frame().T
    paths = recourse_interface.get_paths(poi)
    cfs = recourse_interface.get_counterfactuals_from_paths(paths)
    
    l2_path_len = []
    l2_prox = []
    l2_path_steps = []
    failures = []
    if len(paths) < 1:
        failures= [1]*k_directions
        l2_path_len = [np.nan]*k_directions
        l2_prox = [np.nan]*k_directions
        l2_path_steps = [np.nan]*k_directions
    else:
        for i, p in enumerate(paths):
            if not p or model_interface.predict(p[-1]) == 0:
                failures.append(1)
                l2_path_len.append(np.nan)
                l2_prox.append(np.nan)
                l2_path_steps.append(np.nan)
            elif model_interface.predict(p[-1]) == 1 and p:
                l2_path_len.append(recourse_metrics.compute_norm_path(p, 2))
                l2_prox.append(recourse_metrics.compute_norm(poi, p[-1], ord=2))
                l2_path_steps.append(len(p[1:]))
                failures.append(0)
            else:
                failures.append(1)
                l2_path_len.append(np.nan)
                l2_prox.append(np.nan)
                l2_path_steps.append(np.nan)
    return [l2_path_len, l2_prox, l2_path_steps,
            list(poi.index.values)*len(paths), failures,
            [recourse_metrics.compute_diversity(cfs)]*len(paths)]


if __name__ == "__main__":
    #argument = your dict
    #TODO: give names for util files that can go in here
    arguments = {
        "trials": 10,
        "dataset name": "credit default",
        "dataset encoded" : "OneHot",
        "dataset scaler" : "Standard",
        "dataset valid-test split" : [0.15,0.15],
        "base model" : {"name":"LogisticRegressionSK"},
        "recourse methods" : {"StEP": {'k_directions':3, 'max_iterations':50, 'confidence_threshold':0.7,
                            'directions_rescaler': "constant step size", 'step_size': 1.0}}
    }
    agg_recourse_results, recourse_results = run_experiments_trials(arguments)
    print(agg_recourse_results)
    print("---------")
    print(recourse_results)
    #TODO: use df.to_latex
