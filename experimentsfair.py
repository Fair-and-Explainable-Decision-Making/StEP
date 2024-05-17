import csv
from turtle import color
from networkx import is_empty
import pandas as pd
from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors

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
        n_neg_samples = 1000
        arguments['max num neg samples'] = n_neg_samples

    prot_attr = data_interface.get_processed_prot_feats()
    print(prot_attr)
    
    if data_interface.get_processed_prot_feats():
        prot_attr = data_interface.get_processed_prot_feats()[0]
        neg_data = feats_test.loc[preds[preds != 1].index]

        min_group_samples = neg_data[prot_attr].value_counts().min()
        if n_neg_samples//2 < min_group_samples:
            min_group_samples = n_neg_samples//2
        prot_attr_vals = neg_data[prot_attr].unique()
        print()
        df_g = []
        for g in prot_attr_vals:
            df_g.append(neg_data.loc[neg_data[prot_attr] == g].head(min_group_samples))
        neg_data = pd.concat(df_g)
        arguments['max num neg samples'] = min_group_samples
    print(neg_data[prot_attr].value_counts())
    
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
                lambda row: generate_recourse_results(row, recourse_interface, model_interface, k_directions,recourse_args,prot_attr,feats_train), axis=1)
            print(recourse_results)
            path_df = pd.concat(path_df_li, ignore_index=True)
            print(neg_data)
            print(path_df)
            
            path_df.to_csv(results_file_str+'.csv', index=False)
            #TODO: go into folder
            df_results = pd.DataFrame.from_dict(
                dict(zip(recourse_results.index, recourse_results.values))).T
            df_results.rename(columns={0: "l2_path_len", 1: "l2_prox", 2: "l1_prox",
                                    3: "poi_id", 4: "path_failures", 5: "any_path_failed", 6: "diversity", 7: prot_attr, 8: "density"}, inplace=True)
            print(trial_num)
            print(df_results)
            print(df_results[df_results.stack().str.len().lt(k_directions).any(level=0)])
            df_results = df_results.explode(df_results.columns.values.tolist())
            print(df_results)
            #df_results.to_csv(recourse_dir+"/no_agg_trial_{}".format(trial_num).replace(" ", "")+'.csv', index=False)
            end_recourse = time.time()
            df_results["time"] = end_recourse - start_recourse
            df_results["n_neg_samples"] = actual_n_neg_samples
            #df_agg = df_results.copy().groupby('poi_id').filter(lambda x: len(x) < k_directions)
            #df_agg = df_agg[df_agg['failures'].apply(lambda x: sum(x) == k_directions)]
            
            df_agg = df_results.copy().loc[df_results['any_path_failed'] == 0]
            df_agg = df_agg.drop(columns=['path_failures',"any_path_failed"])
            df_min = df_agg.copy().groupby("poi_id").min().reset_index().drop(columns=['poi_id'])
            #print(df_min)
            df_max = df_agg.copy().max().to_frame().T.drop(columns=['poi_id',prot_attr])

            df_agg_g = df_min.copy().groupby(prot_attr).mean().reset_index()
            
            df_agg_g_0 = df_agg_g.copy().div(df_max.iloc[0]).loc[df_agg_g[prot_attr]==0.0].add_prefix("0_").reset_index()
            df_agg_g_1 = df_agg_g.copy().div(df_max.iloc[0]).loc[df_agg_g[prot_attr]==1.0].add_prefix("1_").reset_index()
            #TODO variance group by
            #cols_to_drop = ['poi_id','min_diversity','max_diversity','min_time','max_time','min_any_path_failed','max_any_path_failed','min_n_neg_samples','max_n_neg_samples']
            df_agg = df_min.mean(axis=0).to_frame().T
            
            #df_agg = pd.concat([df_agg, df_min, df_max], axis=1).drop(columns=cols_to_drop)
            #print(df_agg)
            
            df_agg_temp = pd.concat([df_agg, df_agg_g_0,df_agg_g_1], axis=1)
            #print(df_agg)
            egalitarian = df_agg_temp[["0_l2_prox", "0_l1_prox", "0_density", "1_l2_prox", "1_l1_prox", "1_density"]].max(axis=1)
            utilitarian = df_agg_temp[["0_l2_prox", "0_l1_prox", "0_density", "1_l2_prox", "1_l1_prox", "1_density"]].mean(axis=1)

            df_agg_g_0 = df_agg_g.copy().loc[df_agg_g[prot_attr]==0.0].add_prefix("0_").reset_index()
            df_agg_g_1 = df_agg_g.copy().loc[df_agg_g[prot_attr]==1.0].add_prefix("1_").reset_index()
            df_agg = pd.concat([df_agg, df_agg_g_0,df_agg_g_1], axis=1)
            df_agg["egalitarian"]= egalitarian
            df_agg["utilitarian"]= utilitarian
            df_agg["any_path_failed"] = df_results["any_path_failed"].mean()
            
            
            recourse_results_dict[recourse_name].append(df_agg)
            
            df_agg = df_results.copy().drop(columns=['path_failures',"any_path_failed"])
            df_min = df_agg.copy().groupby("poi_id").min().reset_index().drop(columns=['poi_id'])
            df_max = df_agg.copy().max().to_frame().T.drop(columns=['poi_id',prot_attr])
            df_agg_g = df_min.copy().groupby(prot_attr).mean().reset_index()
            df_agg_g_0 = df_agg_g.copy().div(df_max.iloc[0]).loc[df_agg_g[prot_attr]==0.0].add_prefix("0_").reset_index()
            df_agg_g_1 = df_agg_g.copy().div(df_max.iloc[0]).loc[df_agg_g[prot_attr]==1.0].add_prefix("1_").reset_index()
            df_agg_partialfail = df_min.copy().mean(axis=0).to_frame().T
            
            #df_agg_partialfail = pd.concat([df_agg_partialfail, df_min, df_max], axis=1).drop(columns=cols_to_drop)
            df_agg_partialfail_temp = pd.concat([df_agg_partialfail, df_agg_g_0,df_agg_g_1], axis=1)
            egalitarian = df_agg_partialfail_temp[["0_l2_prox", "0_l1_prox", "0_density", "1_l2_prox", "1_l1_prox", "1_density"]].max(axis=1)
            utilitarian = df_agg_partialfail_temp[["0_l2_prox", "0_l1_prox", "0_density", "1_l2_prox", "1_l1_prox", "1_density"]].mean(axis=1)
            df_agg_g_0 = df_agg_g.copy().loc[df_agg_g[prot_attr]==0.0].add_prefix("0_").reset_index()
            df_agg_g_1 = df_agg_g.copy().loc[df_agg_g[prot_attr]==1.0].add_prefix("1_").reset_index()
            df_agg_partialfail = pd.concat([df_agg_partialfail, df_agg_g_0,df_agg_g_1], axis=1)
            df_agg_partialfail["egalitarian"]= egalitarian
            df_agg_partialfail["utilitarian"]= utilitarian

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


def generate_recourse_results(poi, recourse_interface, model_interface, k_directions, recourse_args, prot_attr, feats_train):
    poi = poi.to_frame().T
    paths = recourse_interface.get_paths(poi)
    cfs = recourse_interface.get_counterfactuals_from_paths(paths)

    knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn.fit(feats_train)
    
    l2_path_len = []
    l2_prox = []
    l2_path_steps = []
    failures = []
    densities = []
    poi_failed = 0
    prot_attr_val = poi[prot_attr].tolist()
    
    if len(paths) != k_directions or paths is None:
        failures = [1]*k_directions
        l2_path_len = [np.nan]*k_directions
        l2_prox = [np.nan]*k_directions
        l2_path_steps = [np.nan]*k_directions
        densities = [np.nan]*k_directions
        poi_failed = 1
        for i in range(k_directions):
            p_save = poi.copy()
            p_save["poi_index"] = poi.index.values[0]
            p_save["path_num"] = i
            p_save["path_order"] = 0            
            p_save["failure"] = 1
            
            path_df_li.append(p_save)
            
    else:
        for i, p in enumerate(paths):
            if model_interface.predict(p[-1],confidence_threshold=recourse_args["confidence_threshold"]) == 0:
                failures.append(1)
                l2_path_len.append(np.nan)
                l2_prox.append(np.nan)
                l2_path_steps.append(np.nan)
                densities.append(np.nan)
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
                l2_path_steps.append(recourse_metrics.compute_norm(poi, p[-1], ord=1))
                failures.append(0)
                densities.append(np.mean(knn.kneighbors(p[-1], 5, return_distance=True)[0][0]))
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
                densities.append(np.nan)
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
            [recourse_metrics.compute_diversity(cfs)]*k_directions, prot_attr_val*k_directions, densities]


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
    """arguments = {
        "n jobs": 10,
        "trials": 10,
        "dataset name": "credit default",
        "dataset encoded": "OneHot",
        "dataset scaler": "Standard",
        "dataset valid-test split": [0.15, 0.15],
        "base model": {"name": "LogisticRegressionSK", "load model": False, "save model": False},
        "recourse methods": {"DiCE": {'k_directions':3, 'backend':'sklearn', 'confidence_threshold':0.5}},
        "save results": True,
        "save experiment": True,
        "experiment name": "fair"
    }
    all_results = run_experiments_trials(arguments)

    arguments = {
        "n jobs": 10,
        "trials": 10,
        "dataset name": "credit default",
        "dataset encoded": "OneHot",
        "dataset scaler": "Standard",
        "dataset valid-test split": [0.15, 0.15],
        "base model": {"name": "LogisticRegressionSK", "load model": False, "save model": False},
        "recourse methods": {"FACE": {'k_directions':1, 'direction_threshold':3.0, 'confidence_threshold':0.5,
                                'weight_bias':2.024,'max_iterations':50}},
        "save results": True,
        "save experiment": True,
        "experiment name": "fair"
    }
    all_results = run_experiments_trials(arguments)

    arguments = {
        "n jobs": 10,
        "trials": 10,
        "dataset name": "credit default",
        "dataset encoded": "OneHot",
        "dataset scaler": "Standard",
        "dataset valid-test split": [0.15, 0.15],
        "base model": {"name": "LogisticRegressionSK", "load model": False, "save model": False},
        "recourse methods": {"WachterAgnosticL2": {'k_directions':1, 'confidence_threshold':0.5}},
        "save results": True,
        "save experiment": True,
        "experiment name": "fair"
    }
    all_results = run_experiments_trials(arguments)

    arguments = {
        "n jobs": 10,
        "trials": 10,
        "dataset name": "credit default",
        "dataset encoded": "OneHot",
        "dataset scaler": "Standard",
        "dataset valid-test split": [0.15, 0.15],
        "base model": {"name": "LogisticRegressionSK", "load model": False, "save model": False},
        "recourse methods": {"WachterAgnosticSparse": {'k_directions':1, 'confidence_threshold':0.5}},
        "save results": True,
        "save experiment": True,
        "experiment name": "fair"
    }
    all_results = run_experiments_trials(arguments)"""

    for d in ["creditdefault"]:
        df_li = []
        for rc in ["WachterAgnosticL2","WachterAgnosticSparse", "DiCE","FACE"]:
            expnam = f"results/fair"
            print(expnam)
            fname = f"/{d}/LogisticRegressionSK/{rc}/partial_failed_agg_all_trials_results.csv"
            csv_loc = expnam+fname
            df = pd.read_csv(csv_loc)
            df = df[["l2_prox","l1_prox","density","0_l2_prox","0_l1_prox","0_density","1_l2_prox","1_l1_prox","1_density","recourse_name"]]
            df_li.append(df)
        df = pd.concat(df_li,ignore_index=True)
        df_max = df[["l2_prox","l1_prox","density"]].copy().max().to_frame().T
        print(df_max) 
        df_max = pd.concat([df_max.add_prefix("0_"),df_max.add_prefix("1_")],axis=1)
        print(df_max) 
        df_welf = df.copy()[["0_l2_prox","0_l1_prox","0_density","1_l2_prox","1_l1_prox","1_density"]]
        print(df_welf)
        df_welf = df_welf/df_welf.max()
        print(df_welf)
        df["utilitarian"]= df_welf.mean(axis=1)
        df["egalitarian"]= df_welf.max(axis=1)
        print(df)
        df.to_csv(f'results/fair_{d}.csv')
        df_disc = df.copy()[["0_l2_prox","0_l1_prox","0_density"]].rename(columns = lambda x: x.strip('0_')) - df.copy()[["1_l2_prox","1_l1_prox","1_density"]].rename(columns = lambda x: x.strip('1_'))
        print(df_disc)
        df_disc.to_csv(f'results/fair_{d}_indirect.csv')

        df_welf = df.copy()[["0_l2_prox","0_l1_prox","0_density","1_l2_prox","1_l1_prox","1_density"]]
        df_welf = df_welf- df_welf.min()
        df_disc = df_welf.copy()[["0_l2_prox","0_l1_prox","0_density"]].rename(columns = lambda x: x.strip('0_')) - df_welf.copy()[["1_l2_prox","1_l1_prox","1_density"]].rename(columns = lambda x: x.strip('1_'))
        print(df_disc)
        df_disc.to_csv(f'results/fair_{d}_induced.csv')
