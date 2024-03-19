import csv
import pandas as pd
from typing import Tuple

from sklearn.linear_model import LogisticRegression

import metrics.recourse_metrics as recourse_metrics
import metrics.classifier_metrics as classifier_metrics
import numpy as np
import time
from data.dataset_utils import get_dataset_interface_by_name
from models.model_utils import get_model_interface_by_name
from recourse.utils.recourse_utils import get_recourse_interface_by_name
from joblib import Parallel, delayed
import torch
import random
import pathlib
import json

def run_experiments_trials(arguments: dict) -> Tuple[dict, dict]:
    start_trials = time.time()
    p_results = Parallel(n_jobs=arguments["n jobs"])(delayed(run_experiments_one_trial)(
        arguments, trial) for trial in range(arguments["trials"]))
    print(p_results)
    recourse_methods = list(p_results[0][0].keys())
    recourse_results_trials_dict = {}
    pf_recourse_results_trials_dict = {}
    base_model_results = []
    for result_i in p_results:
            base_model_results.append(result_i[1])
    base_model_results = pd.concat(base_model_results, ignore_index=True)
    base_model_results_agg_err = (base_model_results.std(axis=0)/(np.sqrt(len(base_model_results)))).to_frame().T
    base_model_results_agg = (base_model_results.mean(axis=0)).to_frame().T
    if arguments["save results"]:
            base_model_dir = "results/{}/{}/{}".format(arguments["experiment name"],
                arguments["dataset name"], arguments["base model"]["name"]).replace(" ", "")
            pathlib.Path(base_model_dir).mkdir(parents=True, exist_ok=True) 
            base_model_results.to_csv(base_model_dir+"/"+arguments["base model"]["name"]+'_results.csv', index=False)
            base_model_results_agg.to_csv(base_model_dir+"/"+arguments["base model"]["name"]+'_agg.csv', index=False)
            base_model_results_agg_err.to_csv(base_model_dir+"/"+arguments["base model"]["name"]+'_agg_err.csv', index=False)
    for recourse_method in recourse_methods:
        recourse_results_trials_dict[recourse_method] = []
        pf_recourse_results_trials_dict[recourse_method] = []
        for result_i in p_results:
            recourse_results_trials_dict[recourse_method].append(
                result_i[0][recourse_method][0])
            pf_recourse_results_trials_dict[recourse_method].append(
                result_i[2][recourse_method][0])
    print(recourse_results_trials_dict)
    save_results_dict(arguments,recourse_results_trials_dict)
    save_results_dict(arguments,pf_recourse_results_trials_dict,f_name_prefix="partial_failed_")
    end_trials = time.time()
    print("All trials took", end_trials-start_trials, "seconds.")

    """return agg_recourse_results, agg_recourse_results_err, recourse_results,\
            base_model_results_agg, base_model_results_agg_err, base_model_results"""

def save_results_dict(arguments, results_dict, f_name_prefix = ""):
    agg_recourse_results = {}
    recourse_results = {}
    agg_recourse_results_err = {}
    for recourse_name, results in results_dict.items():
        results = pd.concat(results, ignore_index=True)
        results["recourse_name"] = recourse_name
        recourse_results[recourse_name] = results.copy()
        agg_recourse_results_err[recourse_name] = (results.std(axis=0)/(np.sqrt(len(results)))).to_frame().T
        agg_recourse_results[recourse_name] = (results.mean(axis=0)).to_frame().T
        #TODO: go into folder
        if arguments["save results"]:
            recourse_results[recourse_name]["recourse_name"] = recourse_name
            agg_recourse_results_err[recourse_name]["recourse_name"] = recourse_name
            agg_recourse_results[recourse_name]["recourse_name"] = recourse_name
            recourse_dir = "results/{}/{}/{}/{}".format(arguments["experiment name"],
                arguments["dataset name"], arguments["base model"]["name"],recourse_name).replace(" ", "")
            pathlib.Path(recourse_dir).mkdir(parents=True, exist_ok=True) 
            recourse_results[recourse_name].to_csv(recourse_dir+'/'+f_name_prefix+'all_trials_results.csv', index=False)
            agg_recourse_results[recourse_name].to_csv(recourse_dir+'/'+f_name_prefix+'agg_all_trials_results.csv', index=False)
            agg_recourse_results_err[recourse_name].to_csv(recourse_dir+'/'+f_name_prefix+'agg_all_trials_err.csv', index=False)

def run_experiments_one_trial(arguments, trial_num=0):
    file_str = "{}_{}_dataset_{}_model_{}_trialnum".format(arguments["experiment name"],
                    arguments["dataset name"], arguments["base model"]["name"],str(trial_num)).replace(" ", "")
    np.random.seed(trial_num)
    torch.manual_seed(trial_num)
    random.seed(trial_num)
    
    start_trial = time.time()
    data_interface = get_dataset_interface_by_name(arguments["dataset name"])
    if arguments["dataset encoded"]:
        data_interface.encode_data()
    if arguments["dataset scaler"]:
        data_interface.scale_data(arguments["dataset scaler"])
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data(
        arguments["dataset valid-test split"][0], arguments["dataset valid-test split"][1], random_state=trial_num)
    model_args = arguments["base model"].copy()
    model_args["feats_train"], model_args["feats_valid"], model_args["labels_valid"] = feats_train, feats_valid, labels_valid
    model_args['random seed'] = trial_num 
    model_interface = get_model_interface_by_name(**model_args) 
    if model_args["load model"]:
        model_interface.load_model("models/saved models/"+file_str)
    else:
        model_interface.fit(feats_train, labels_train)
    if model_args["save model"]:
        model_interface.save_model("models/saved models/"+file_str)
    preds = pd.Series(model_interface.predict(
        feats_test), index=feats_test.index)
    if 'max num neg samples' in arguments.keys():
        n_neg_samples = arguments['max num neg samples']
    else:
        n_neg_samples = 100
        arguments['max num neg samples'] = n_neg_samples
    neg_data = feats_test.loc[preds[preds != 1].index].head(n_neg_samples)
    actual_n_neg_samples = neg_data.shape[0]
    print(neg_data.shape)
    # TODO: change this to output to a text file or something similar
    base_model_results = classifier_metrics.run_classifier_tests(labels_test, model_interface.predict(feats_test))

    done_training_trial = time.time()
    print("Model training took", done_training_trial-start_trial, "seconds")
    recourse_results_dict = {}
    recourse_partialfail_results_dict = {}
    if "only base model" not in arguments: 
        arguments["only base model"] = False
    if not arguments["only base model"]:
        for recourse_name, recourse_args in arguments["recourse methods"].items():
            recourse_args['random seed'] = trial_num
            start_recourse = time.time()
            if recourse_name not in recourse_results_dict:
                recourse_results_dict[recourse_name] = []
            if recourse_name not in recourse_partialfail_results_dict:
                recourse_partialfail_results_dict[recourse_name] = []
            recourse_interface = get_recourse_interface_by_name(
                recourse_name, model_interface, data_interface, **recourse_args)
            k_directions = recourse_args['k_directions']
            recourse_dir = "results/{}/{}/{}/{}/trials".format(arguments["experiment name"],
                arguments["dataset name"], arguments["base model"]["name"],recourse_name).replace(" ", "")
            pathlib.Path(recourse_dir).mkdir(parents=True, exist_ok=True) 
            results_file_str = recourse_dir+"/paths_trial_{}".format(trial_num).replace(" ", "")
            global path_df_li 
            path_df_li = []
            
            recourse_results = neg_data.apply(
                lambda row: generate_recourse_results(row, recourse_interface, model_interface, k_directions,recourse_args), axis=1)
            print(recourse_results)
            path_df = pd.concat(path_df_li, ignore_index=True)
            print(neg_data)
            print(path_df)
            
            path_df.to_csv(results_file_str+'.csv', index=False)
            #TODO: go into folder
            df_results = pd.DataFrame.from_dict(
                dict(zip(recourse_results.index, recourse_results.values))).T
            df_results.rename(columns={0: "l2_path_len", 1: "l2_prox", 2: "l2_path_steps",
                                    3: "poi_id", 4: "path_failures", 5: "any_path_failed", 6: "diversity"}, inplace=True)
            print(trial_num)
            print(df_results)
            print(df_results[df_results.stack().str.len().lt(k_directions).any(level=0)])
            df_results = df_results.explode(df_results.columns.values.tolist())
            #df_results.to_csv(recourse_dir+"/no_agg_trial_{}".format(trial_num).replace(" ", "")+'.csv', index=False)
            end_recourse = time.time()
            df_results["time"] = end_recourse - start_recourse
            df_results["n_neg_samples"] = actual_n_neg_samples
            #df_agg = df_results.copy().groupby('poi_id').filter(lambda x: len(x) < k_directions)
            #df_agg = df_agg[df_agg['failures'].apply(lambda x: sum(x) == k_directions)]
            
            df_agg = df_results.copy().loc[df_results['any_path_failed'] == 0]
            df_min = df_agg.copy().groupby("poi_id").min().add_prefix('min_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            df_max = df_agg.copy().groupby("poi_id").max().add_prefix('max_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            #TODO variance group by
            cols_to_drop = ['poi_id','min_diversity','max_diversity','min_time','max_time','min_any_path_failed','max_any_path_failed','min_n_neg_samples','max_n_neg_samples']
            df_agg = df_agg.mean(axis=0).to_frame().T
            df_agg = pd.concat([df_agg, df_min, df_max], axis=1).drop(columns=cols_to_drop)
            df_agg["any_path_failed"] = df_results["any_path_failed"].mean()
            recourse_results_dict[recourse_name].append(df_agg)
            
            df_min = df_results.copy().groupby("poi_id").min().add_prefix('min_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            df_max = df_results.copy().groupby("poi_id").max().add_prefix('max_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            df_agg_partialfail = df_results.copy().mean(axis=0).to_frame().T
            df_agg_partialfail = pd.concat([df_agg_partialfail, df_min, df_max], axis=1).drop(columns=cols_to_drop)
            recourse_partialfail_results_dict[recourse_name].append(df_agg_partialfail)
            print(recourse_name, "recourse took", end_recourse -
                start_recourse, "seconds for", len(neg_data), "samples.")
            recourse_dir = "results/{}/{}/{}/{}/trials".format(arguments["experiment name"],
                arguments["dataset name"], arguments["base model"]["name"],recourse_name).replace(" ", "")
            results_file_str = recourse_dir+"/agg_results_trial_{}".format(trial_num).replace(" ", "")
            df_agg.to_csv(results_file_str+'.csv', index=False)
            results_file_str = recourse_dir+"/partial_failed_agg_results_trial_{}".format(trial_num).replace(" ", "")
            df_agg_partialfail.to_csv(results_file_str+'.csv', index=False)
    end_trial = time.time()
    return recourse_results_dict, base_model_results, recourse_partialfail_results_dict


def generate_recourse_results(poi, recourse_interface, model_interface, k_directions, recourse_args):
    poi = poi.to_frame().T
    paths = recourse_interface.get_paths(poi)
    cfs = recourse_interface.get_counterfactuals_from_paths(paths)

    l2_path_len = []
    l2_prox = []
    l2_path_steps = []
    failures = []
    poi_failed = 0
    if len(paths) != k_directions or paths is None:
        failures = [1]*k_directions
        l2_path_len = [np.nan]*k_directions
        l2_prox = [np.nan]*k_directions
        l2_path_steps = [np.nan]*k_directions
        for i, p in enumerate(paths):
            p_save = poi.copy()
            p_save["poi_index"] = poi.index.values[0]
            p_save["path_num"] = i
            p_save["path_order"] = 0            
            p_save["failure"] = 1
            path_df_li.append(p_save)
            poi_failed = 1
    else:
        for i, p in enumerate(paths):
            if model_interface.predict(p[-1],confidence_threshold=recourse_args["confidence_threshold"]) == 0:
                failures.append(1)
                l2_path_len.append(np.nan)
                l2_prox.append(np.nan)
                l2_path_steps.append(np.nan)
                p_save = pd.concat(p.copy(), ignore_index=True)
                p_save["poi_index"] = poi.index.values[0]
                p_save["path_num"] = i
                p_save["path_order"] = list(range(len(p)))
                p_save["failure"] = 1
                poi_failed = 1
            elif model_interface.predict(p[-1],confidence_threshold=recourse_args["confidence_threshold"]) == 1 and p:
                l2_path_len.append(recourse_metrics.compute_norm_path(p, 2))
                l2_prox.append(
                    recourse_metrics.compute_norm(poi, p[-1], ord=2))
                l2_path_steps.append(len(p[1:]))
                failures.append(0)
                p_save = pd.concat(p.copy(), ignore_index=True)
                p_save["poi_index"] = poi.index.values[0]
                p_save["path_num"] = i
                p_save["path_order"] = list(range(len(p)))
                p_save["failure"] = 0
            else:
                failures.append(1)
                l2_path_len.append(np.nan)
                l2_prox.append(np.nan)
                l2_path_steps.append(np.nan)
                p_save = poi.copy().reset_index(drop=True)
                p_save["poi_index"] = poi.index.values[0]
                p_save["path_num"] = i
                p_save["path_order"] = 0
                p_save["failure"] = 1
                poi_failed = 1
            
            #TODO: save cfs into csv, put into seperate folder
            path_df_li.append(p_save)
    return [l2_path_len, l2_prox, l2_path_steps,
            list(poi.index.values)*k_directions, failures, [poi_failed]*k_directions,
            [recourse_metrics.compute_diversity(cfs)]*k_directions]


if __name__ == "__main__":
    # argument = your dict
    # TODO: give names for util files that can go in here
    """
    Argument values
    - "n jobs": any postive int, generally no more than the number of cores you have
    - "trials": any positive int
    - "dataset name": one of "credit default", "give credit", or "adult census"
    - "dataset encoded": just "OneHot" or None,
    - "dataset scaler" : "Standard", "MinMax, or None
    - "dataset valid-test split" : List[float, float] where both values summed is < 1.0
    - "base model":
        - "name": "BaselineDNN", "LogisticRegressionPT", "LogisticRegressionSK", or "RandomForestSK"
        - "batch_size": Number of samples for each batch (PyTorch only)
        - "epochs": Number of epochs for training (PyTorch only)
        - "lr": Learning rate (PyTorch only)
    - "recourse methods": dictionary of keys=recourse method name and values=dict of recourse arguments
        - "StEP": dict of StEP arguments ('k_directions', 'max_iterations', 'confidence_threshold',
                            'directions_rescaler','step_size')
        - "DiCE": dict of DiCE arguments ('backend','k_directions','confidence_threshold')
        - "FACE": dict of FACE arguments ('k_directions', 'direction_threshold', 'confidence_threshold','weight_bias')
        - Note, not all of the recorse methods need to be in the dict. 
            E.g., you can just have, 
                "recourse methods" : {"StEP": {'k_directions':3, 'max_iterations':50, 'confidence_threshold':0.7,
                'directions_rescaler': "constant step size", 'step_size': 1.0}}
            for your recourse methods.
    """
    #TODO: renaming saving
    
    for k in range(3,6):
        for step_size in [0.1, 0.25, 0.5, 0.75, 1.0]:
            for conf_thres in [0.5, 0.55, 0.6, 0.65, 0.7]:
                expnam = f"results/{k}Clust_{str(step_size)}StepSize_{str(conf_thres)}ConfThres"

                arguments = {
                    "n jobs": 10,
                    "trials": 10,
                    "dataset name": "adult census",
                    "dataset encoded": "OneHot",
                    "dataset scaler": "Standard",
                    "dataset valid-test split": [0.15, 0.15],
                    "base model": {"name": "LogisticRegressionSK", "load model": False, "save model": False},
                    "recourse methods": {"StEP": {'k_directions':k, 'max_iterations':50, 'confidence_threshold':conf_thres,
                            'directions_rescaler': "constant step size", 'step_size': step_size}},
                    "save results": True,
                    "save experiment": True,
                    "experiment name": expnam
                }
                all_results = run_experiments_trials(arguments)

    df_li = []
    for k in range(3,6):
        for step_size in [0.1, 0.25, 0.5, 0.75, 1.0]:
            for conf_thres in [0.5, 0.55, 0.6, 0.65, 0.7]:
                expnam = f"results/{k}Clust_{str(step_size)}StepSize_{str(conf_thres)}ConfThres"
                print(expnam)
                fname = "/adultcensus/LogisticRegressionSK/StEP/partial_failed_agg_all_trials_results.csv"
                csv_loc = expnam+fname
                df = pd.read_csv(csv_loc)
                df['k'] = k
                df['step_size'] = step_size
                df['conf_thres'] = conf_thres
                df_li.append(df)
    df = pd.concat(df_li,ignore_index=True)
    cols = df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    df = df[cols]
    print(df)
    df.to_csv('results/adult_hyperparam.csv')  
