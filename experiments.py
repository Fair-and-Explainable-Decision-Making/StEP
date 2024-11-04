import csv
import enum
from turtle import color
import pandas as pd
from typing import Tuple

from pyparsing import col
from sklearn import base
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
        n_neg_samples = 1000
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
                                    3: "poi_id", 4: "path_successes", 5: "all_path_succeed", 6: "diversity"}, inplace=True)
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
            
            df_agg = df_results.copy().loc[df_results['all_path_succeed'] == 1]
            df_min = df_agg.copy().groupby("poi_id").min().add_prefix('min_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            df_max = df_agg.copy().groupby("poi_id").max().add_prefix('max_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            #TODO variance group by
            cols_to_drop = ['min_diversity','max_diversity','min_time','max_time','min_all_path_succeed','max_all_path_succeed','min_n_neg_samples','max_n_neg_samples']
            #df_agg = df_agg.mean(axis=0).to_frame().T
            df_agg = df_agg.copy().groupby("poi_id").mean(numeric_only=False).reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            df_agg = pd.concat([df_agg, df_min, df_max], axis=1).drop(columns=cols_to_drop)
            df_agg["all_path_succeed"] = df_results["all_path_succeed"].mean()
            df_agg =  df_agg.rename(columns={"max_path_successes": "pois_with_geq_1_success","min_path_successes": "pois_with_lessthan_k_success"})
            recourse_results_dict[recourse_name].append(df_agg)
            
            df_min = df_results.copy().groupby("poi_id").min().add_prefix('min_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            df_max = df_results.copy().groupby("poi_id").max().add_prefix('max_').reset_index().mean(axis=0).to_frame().T.drop(columns=['poi_id'])
            #df_agg_partialfail = df_results.copy().mean(axis=0).to_frame().T
            df_agg_partialfail = df_results.copy().groupby("poi_id").mean(numeric_only=False).reset_index().mean(axis=0).to_frame().T
            df_agg_partialfail = pd.concat([df_agg_partialfail, df_min, df_max], axis=1).drop(columns=cols_to_drop)
            df_agg_partialfail =  df_agg_partialfail.rename(columns={"max_path_successes": "pois_with_geq_1_success","min_path_successes": "pois_with_lessthan_k_success"})
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
    if poi_failed:
        all_successes = 0
    else:
        all_successes =1
    succeses = 1 - np.array(failures) 
    return [l2_path_len, l2_prox, l2_path_steps,
            list(poi.index.values)*k_directions, succeses, [all_successes]*k_directions,
            [recourse_metrics.compute_diversity(poi,cfs)]*k_directions]

def agg_results_trial_csv(arguments):
    expnam = arguments["experiment name"]
    recourse_dir = "results/{}/{}/{}/{}/trials".format(expnam,
        arguments["dataset name"], arguments["base model"]["name"],list(arguments["recourse methods"].keys())[0]).replace(" ", "")
    
    df_part_li = []
    df_li = []
    for i in range(arguments["trials"]):
        fname_partial = f"/partial_failed_agg_results_trial_{i}.csv"
        csv_loc = recourse_dir+fname_partial
        df_part_li.append(pd.read_csv(csv_loc))
        fname = f"/agg_results_trial_{i}.csv"
        csv_loc = recourse_dir+fname
        df_li.append(pd.read_csv(csv_loc))
    recourse_results_trials_dict = {list(arguments["recourse methods"].keys())[0]: df_li} 
    pf_recourse_results_trials_dict = {list(arguments["recourse methods"].keys())[0]: df_part_li} 
    save_results_dict(arguments,recourse_results_trials_dict)
    save_results_dict(arguments,pf_recourse_results_trials_dict,f_name_prefix="partial_failed_")

def df_to_latex(expnam, cols_to_report=["Success", "Avg Success","L2 Distance","Diversity"]):
    df_li = []
    datasets = ["creditdefault","givecredit","adultcensus"]
    base_models = ["LogisticRegressionSK","RandomForestSK","BaselineDNN"]
    rc = "StEP"
    
    max_err= 0
    csvnames = ["partial_failed_agg_all_trials_results.csv", "partial_failed_agg_all_trials_err.csv"]
    results = []
    for csvname in csvnames:
        df_li = []
        for i_d, d in enumerate(datasets):
            df_li_d = []
            for rc in ["StEP","DiCE","FACE","CCHVAE"]:    
                df_li_b = []
                for i,b in enumerate(base_models):
                    expnam1 = f"results/"+ expnam
                    df_result = pd.read_csv(expnam1 + f"/{d}/{b}/{rc}/"+csvname)
                    cols_to_drop = ["poi_id","min_l2_path_len","min_l2_prox","min_l2_path_steps","max_l2_path_len","max_l2_prox","max_l2_path_steps"]
                    df_new = df_result.copy()
                    df_new = df_new.drop(columns=cols_to_drop)
                    df_new = df_new.rename(columns={"pois_with_geq_1_success": "Success", "path_successes": "Avg Success", "l2_prox": "L2 Distance","diversity": "Diversity", "n_neg_samples":"n Samples","l2_path_len": "Path Length","l2_path_steps":"Path Steps"})

                    if rc == "StEP":
                        df_new["Dataset"] = d
                    else:
                        df_new["Dataset"] = None
                    if i == 0:
                        df_new["Method"] = rc
                        df = df_new[["Dataset","Method"]+cols_to_report]
                    else:
                        df = df_new[cols_to_report]
                    df_new["Dataset"] = d
                    df_new["Method"] = rc
                    df_new["Model"] = b
                    df = df_new[["Dataset","Model","Method"]+cols_to_report]
                    df_li_b.append(df)

                df = pd.concat(df_li_b,ignore_index=True)
                
                df_li_d.append(df)
            df = pd.concat(df_li_d,ignore_index=True)
            df1 = df.pivot_table(index=["Dataset","Model"], columns = ["Method"], aggfunc='sum', sort = False)
            
            df['Dataset'].loc[df['Method'] == "StEP"] = d
            df_li.append(df)
        df = pd.concat(df_li,ignore_index=True)
        
        #df.to_csv(f'results/10trial_latex_ready.csv',float_format="%.2f",index=False)
        df1 = df.pivot_table(index=["Dataset","Method"], columns="Model", values = cols_to_report, aggfunc='sum', sort = False)
        df1 = df1.swaplevel(0, 1, axis=1).sort_index(axis=1)
        cols = ['LogisticRegressionSK', 'RandomForestSK', 'BaselineDNN']
        new_cols = df1.columns.reindex(cols, level=0)
        df1 = df1.reindex(columns=new_cols[0])
        df1 = df1.reindex(cols_to_report, level=1, axis=1)
        print(df1.to_latex())
        print(max_err)
        results.append(df1)
    for r in results:
        print(r.round(decimals=2))
        print(r.round(decimals=2).to_latex())
    print((results[1]/results[0]).max(axis=0).round(decimals=4))
    print((results[1]/results[0]).max(axis=1).round(decimals=4))

def df_to_latex_noise(expnam1,expnam2, cols_to_report=["Success", "Avg Success","L2 Distance","Diversity"]):
    df_li = []
    datasets = ["creditdefault","givecredit","adultcensus"]
    base_models = ["LogisticRegressionSK","RandomForestSK","BaselineDNN"]
    rc = "StEP"
    csvnames = ["partial_failed_agg_all_trials_results.csv", "partial_failed_agg_all_trials_err.csv"]
    results = []
    for csvname in csvnames:
        df_li = []
        for i_d, d in enumerate(datasets):
            df_li_d = []
            for noise in [0.0,0.1,0.3,0.5]:
                noise = round(noise, 1)    
                df_li_b = []
                for i,b in enumerate(base_models):
                    expnam = f"results/"+ expnam1+ str(noise)+expnam2
                    df_result = pd.read_csv(expnam + f"/{d}/{b}/{rc}/"+csvname)
                
                    cols_to_drop = ["poi_id","min_l2_path_len","min_l2_prox","min_l2_path_steps","max_l2_path_len","max_l2_prox","max_l2_path_steps"]
                    df_new = df_result.copy().drop(columns=cols_to_drop)
                    df_new = df_new.rename(columns={"pois_with_geq_1_success": "Success", "path_successes": "Avg Success", "l2_prox": "L2 Distance","diversity": "Diversity", "n_neg_samples":"n Samples","l2_path_len": "Path Length","l2_path_steps":"Path Steps"})
                    
                    if noise == 0.0:
                        df_new["Dataset"] = d
                    else:
                        df_new["Dataset"] = None
                    if i == 0:
                        df_new["Noise"] = noise
                        df = df_new[["Dataset","Noise"]+cols_to_report]
                    else:
                        df = df_new[cols_to_report]
                    df_new["Dataset"] = d
                    df_new["Noise"] = noise
                    df_new["Model"] = b
                    df = df_new[["Dataset","Model","Noise"]+cols_to_report]
                    df_li_b.append(df)

                df = pd.concat(df_li_b,ignore_index=True)
                df_li_d.append(df)
            df = pd.concat(df_li_d,ignore_index=True)
            df1 = df.pivot_table(index=["Dataset","Model"], columns = ["Noise"], aggfunc='sum', sort = False)
            
            df['Dataset'].loc[df['Noise'] == 0.0] = d
            df_li.append(df)
        df = pd.concat(df_li,ignore_index=True)
        
        #df.to_csv(f'results/10trial_latex_ready.csv',float_format="%.2f",index=False)
        print(df)
        df1 = df.pivot_table(index=["Dataset","Noise"], columns="Model", values = cols_to_report, aggfunc='sum', sort = False)
        print(df1)
        df1 = df1.swaplevel(0, 1, axis=1).sort_index(axis=1)
        cols = ['LogisticRegressionSK', 'RandomForestSK', 'BaselineDNN']
        new_cols = df1.columns.reindex(cols, level=0)
        df1 = df1.reindex(columns=new_cols[0])
        df1 = df1.reindex(cols_to_report, level=1, axis=1)
        print(df1)
        print(df1.to_latex())
        results.append(df1)
    for r in results:
        print(r.copy().round(decimals=2))
        print(r.copy().round(decimals=2).to_latex())
    print((results[1]/results[0]).max(axis=0).round(decimals=4))
    print((results[1]/results[0]).max(axis=1).round(decimals=4))

def run_neurips_holistic_experiment(trials=10, n_jobs = 10,k_directions = 3, conf_thres = 0.7):
    base_models = [{"name": "LogisticRegressionSK", "load model": False, "save model": False},
                   {"name": "RandomForestSK", "load model": False, "save model": False},
                   {"name": "BaselineDNN", "load model": False, "save model": False, "batch_size": 1, "epochs":5,"lr":1e-4}]
    
    datasets = ["credit default","give credit","adult census"]
    
    recourse_methods = [{"StEP": {'k_directions':k_directions, 'max_iterations':50, 'confidence_threshold':conf_thres,
                    'directions_rescaler': "constant step size", 'step_size': 1.0, 'noise':0.0}},
          {"DiCE": {'k_directions':k_directions, 'backend':'sklearn', 'confidence_threshold':conf_thres}},
          {"FACE": {'k_directions':k_directions, 'direction_threshold':3.0, 'confidence_threshold':conf_thres,
                    'weight_bias':2.024,'max_iterations':50}},
          {"CCHVAE": {'k_directions':k_directions, 'confidence_threshold':conf_thres, 'max_iterations':50,'train vae': True}}
          ]
    
    for b in base_models:
        for d in datasets:
            for rc in recourse_methods:
                expnam = f"{trials}trials_{k_directions}Clust_{str(conf_thres)}ConfThres_newdiv"
                arguments = {
                    "n jobs": trials,
                    "trials": n_jobs,
                    "dataset name": d,
                    "dataset encoded": "OneHot",
                    "dataset scaler": "Standard",
                    "dataset valid-test split": [0.15, 0.15],
                    "base model": b,
                    "recourse methods": rc,
                    "save results": True,
                    "save experiment": True,
                    "experiment name": expnam
                }
                run_experiments_trials(arguments)

def run_neurips_noise_experiment(trials=10, n_jobs = 10, k_directions = 3, conf_thres = 0.7, noise_range = np.arange(0.0, 0.6, 0.1)):
    base_models = [{"name": "LogisticRegressionSK", "load model": False, "save model": False},
                   {"name": "RandomForestSK", "load model": False, "save model": False},
                   {"name": "BaselineDNN", "load model": False, "save model": False, "batch_size": 1, "epochs":5,"lr":1e-4}]
    
    datasets = ["credit default","give credit","adult census"]
    for b in base_models:
        for d in datasets:
            for noise in noise_range:
                noise = round(noise, 1)
                expnam = f"{trials}trials_{k_directions}Clust_{str(conf_thres)}ConfThres_{str(noise)}noise"
                arguments = {
                    "n jobs": trials,
                    "trials": n_jobs,
                    "dataset name": d,
                    "dataset encoded": "OneHot",
                    "dataset scaler": "Standard",
                    "dataset valid-test split": [0.15, 0.15],
                    "base model": b,
                    "recourse methods": {"StEP": {'k_directions':k_directions, 'max_iterations':50, 'confidence_threshold':conf_thres,
                                        'directions_rescaler': "constant step size", 'step_size': 1.0, 'noise':noise}},
                    "save results": True,
                    "save experiment": True,
                    "experiment name": expnam
                }
                all_results = run_experiments_trials(arguments)                    

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
    run_neurips_holistic_experiment()
    
    