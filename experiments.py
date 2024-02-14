import pandas as pd
from typing import Tuple
import metrics.recourse_metrics as recourse_metrics
import metrics.classifier_metrics as classifier_metrics
import numpy as np
import time
from data.dataset_utils import get_dataset_interface_by_name
from models.model_utils import get_model_interface_by_name
from recourse.utils.recourse_utils import get_recourse_interface_by_name
from joblib import Parallel, delayed


def run_experiments_trials(arguments: dict) -> Tuple[dict, dict]:
    start_trials = time.time()
    p_results = Parallel(n_jobs=arguments["n jobs"])(delayed(run_experiments_one_trial)(
        arguments, trial) for trial in range(arguments["trials"]))
    print(p_results)
    recourse_methods = list(p_results[0][0].keys())
    recourse_results_trials_dict = {}
    base_model_results = []
    for result_i in p_results:
            base_model_results.append(result_i[1])
    base_model_results = pd.concat(base_model_results, ignore_index=True)
    base_model_results_agg_err = (base_model_results.std(axis=0)/(np.sqrt(len(base_model_results)))).to_frame().T
    base_model_results_agg = (base_model_results.mean(axis=0)).to_frame().T
    if arguments["save results"]:
            results_file_str = "results/{}_{}_{}trials".format(
                arguments["base model"]["name"],arguments["dataset name"],str(arguments["trials"])).replace(" ", "")
            base_model_results.to_csv(results_file_str+'_results.csv', index=False)
            base_model_results_agg.to_csv(results_file_str+'_agg.csv', index=False)
            base_model_results_agg_err.to_csv(results_file_str+'_agg_err.csv', index=False)
    for recourse_method in recourse_methods:
        recourse_results_trials_dict[recourse_method] = []
        for result_i in p_results:
            recourse_results_trials_dict[recourse_method].append(
                result_i[0][recourse_method][0])
    print(recourse_results_trials_dict)
    agg_recourse_results = {}
    recourse_results = {}
    agg_recourse_results_err = {}
    for recourse_name, results in recourse_results_trials_dict.items():
        results = pd.concat(results, ignore_index=True).drop(columns=['poi_id'])
        recourse_results[recourse_name] = results
        agg_recourse_results_err[recourse_name] = (results.std(axis=0)/(np.sqrt(len(results)))).to_frame().T
        agg_recourse_results[recourse_name] = (results.mean(axis=0)).to_frame().T
        if arguments["save results"]:
            results_file_str = "results/{}_{}_{}_{}trials".format(
                recourse_name, arguments["dataset name"],
                arguments["base model"]["name"],str(arguments["trials"])).replace(" ", "")
            recourse_results[recourse_name].to_csv(results_file_str+'_results.csv', index=False)
            agg_recourse_results[recourse_name].to_csv(results_file_str+'_agg.csv', index=False)
            agg_recourse_results_err[recourse_name].to_csv(results_file_str+'_agg_err.csv', index=False) 
    end_trials = time.time()
    print("All trials took", end_trials-start_trials, "seconds.")

    return agg_recourse_results, agg_recourse_results_err, recourse_results,\
            base_model_results_agg, base_model_results_agg_err, base_model_results


def run_experiments_one_trial(arguments, trial_num=0):
    file_str = "{}_dataset_{}_model_{}_trialnum".format(arguments["dataset name"],
                    arguments["base model"]["name"],str(trial_num)).replace(" ", "")
    if arguments["save experiment"]:
        arguments["file name"] = file_str
    np.random.seed(trial_num) 
    recourse_results_dict = {}
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
    model_interface = get_model_interface_by_name(**model_args)

    model_interface.fit(feats_train, labels_train)
    preds = pd.Series(model_interface.predict(
        feats_test), index=feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index]
    # TODO: change this to output to a text file or something similar
    base_model_results = classifier_metrics.run_classifier_tests(labels_test, model_interface.predict(feats_test))

    done_training_trial = time.time()
    print("Model training took", done_training_trial-start_trial, "seconds")
    for recourse_name, recourse_args in arguments["recourse methods"].items():
        recourse_args['random seed'] = trial_num
        start_recourse = time.time()
        if recourse_name not in recourse_results_dict:
            recourse_results_dict[recourse_name] = []
        recourse_interface = get_recourse_interface_by_name(
            recourse_name, model_interface, data_interface, **recourse_args)
        k_directions = recourse_args['k_directions']
        recourse_results = neg_data.apply(
            lambda row: generate_recourse_results(row, recourse_interface, model_interface, k_directions), axis=1)
        df_results = pd.DataFrame.from_dict(
            dict(zip(recourse_results.index, recourse_results.values))).T
        df_results.rename(columns={0: "l2_path_len", 1: "l2_prox", 2: "l2_path_steps",
                                   3: "poi_id", 4: "failures", 5: "diversity"}, inplace=True)
        print(df_results)
        df_results = df_results.explode(df_results.columns.values.tolist())
        recourse_results_dict[recourse_name].append(
            df_results.mean(axis=0).to_frame().T)
        end_recourse = time.time()
        print(recourse_name, "recourse took", end_recourse -
              start_recourse, "seconds for", len(neg_data), "samples.")
    end_trial = time.time()
    print("Trial", trial_num, "took", end_trial-start_trial, "seconds.")
    return recourse_results_dict, base_model_results


def generate_recourse_results(poi, recourse_interface, model_interface, k_directions):
    poi = poi.to_frame().T
    paths = recourse_interface.get_paths(poi)
    cfs = recourse_interface.get_counterfactuals_from_paths(paths)

    l2_path_len = []
    l2_prox = []
    l2_path_steps = []
    failures = []
    if len(paths) < 1 or paths is None:
        failures = [1]*k_directions
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
                l2_prox.append(
                    recourse_metrics.compute_norm(poi, p[-1], ord=2))
                l2_path_steps.append(len(p[1:]))
                failures.append(0)
            else:
                failures.append(1)
                l2_path_len.append(np.nan)
                l2_prox.append(np.nan)
                l2_path_steps.append(np.nan)
    return [l2_path_len, l2_prox, l2_path_steps,
            list(poi.index.values)*k_directions, failures,
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
    arguments = {
        "n jobs": 5,
        "trials": 10,
        "dataset name": "credit default",
        "dataset encoded": "OneHot",
        "dataset scaler": "Standard",
        "dataset valid-test split": [0.15, 0.15],
        "base model": {"name": "LogisticRegressionSK"},
        "recourse methods": {"StEP": {'k_directions':3, 'max_iterations':50, 'confidence_threshold':0.7,
                'directions_rescaler': "constant step size", 'step_size': 1.0}},
        "save results": True,
        "save experiment": True
    }
    all_results = run_experiments_trials(arguments)

    for r in all_results:
        print("---------")
        print(r)
    
    # TODO: use df.to_latex
