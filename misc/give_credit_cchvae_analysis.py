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

if __name__ == "__main__":   
    """import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv('results/100trial_adult_avg_fails_per_poi_agg.csv')
    print(df)
    df['conf_thres'] = df['conf_thres'].astype(str)
    df['step_size'] = df['step_size'].astype(str)
    ax = sns.barplot(data=df, x='conf_thres', y='failure', hue='step_size', palette=sns.color_palette("tab10"), alpha=0.7)
    ax.set(xlabel='Conf Threshold', ylabel='Failed Paths per POI')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig("results/failsbar.pdf",bbox_inches='tight')"""
    
    """results_li = []
    for t in range(100):
        for conf_thres in [0.5,0.55,0.60,0.65,0.70]:
            df_li = []
            for k in range(3,4):
                for step_size in [0.25,0.5,0.75]:
                    
                        expnam = f"results/results/100trials_{k}Clust_{str(step_size)}StepSize_{str(conf_thres)}ConfThres"
                        fname = f"/adultcensus/LogisticRegressionSK/StEP/trials/paths_trial_{t}.csv"
                        csv_loc = expnam+fname
                        df = pd.read_csv(csv_loc)
                        cols = df.columns.tolist()
                        cols = cols[-4:]
                        df = df[cols]
                        df['k'] = k
                        df['step_size'] = step_size
                        df['conf_thres'] = conf_thres
                        df_li.append(df)
            df = pd.concat(df_li,ignore_index=True)
            df = df.groupby(['k','step_size','conf_thres','poi_index'])['failure'].max().reset_index()
            print(df)
            df = df.groupby(['poi_index'])['failure'].mean().reset_index(name='shared_index')#value_counts().rename("num_shared_exper_result").reset_index()
            print(df)
            print(df['shared_index'].value_counts())
            df = df['shared_index'].value_counts().reset_index()
            results_li.append(df)
            
    #df.to_csv('results/agg_step_poi_shared_fails.csv')
    df = pd.concat(results_li,ignore_index=True)
    df = df.groupby(['index'])['shared_index'].mean().reset_index()
    df.rename(columns={"index": "avg_result","shared_index": "count"}, inplace= True)
    print(df)
    df.to_csv('results/agg_step_poi_shared_fails.csv')"""


    expnam = f"results/succ10trials_3Clust_0.7ConfThres"
    fname = f"/givecredit/LogisticRegressionSK/CCHVAE/trials/paths_trial_0.csv"
    csv_loc = expnam+fname
    df = pd.read_csv(csv_loc)
    cols = df.columns.tolist()
    #cols = cols[-4:]
    df = df[cols]
    df = df.drop(columns=["path_order","path_num"])
    print(df["failure"].value_counts()/6)
    df0 = df.copy().loc[df["failure"] == 0]
    df1 = df.copy().loc[df["failure"] == 1]
    df0 = df0.groupby(['poi_index']).mean().reset_index()
    df1 = df1.groupby(['poi_index']).mean().reset_index()
    print(df1)
    data_interface = get_dataset_interface_by_name("give credit")
    data_interface.encode_data()
    
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data(.15, .15, random_state=0)
    print(df0["poi_index"].values)
    print(feats_test[feats_test.index.isin(df0["poi_index"].values)])
    df0_log = feats_test.copy()[feats_test.index.isin(df0["poi_index"].values)]
    df1_log = feats_test.copy()[feats_test.index.isin(df1["poi_index"].values)]
    dfall_log = feats_test.copy()[feats_test.index.isin(np.concatenate((df0["poi_index"].values, df1["poi_index"].values), axis=None))]

    with pd.option_context('display.max_columns', 2000):
        print("Feats success poi")
        print(df0_log.describe().round(2))
        print("-----------------------")
        print("Feats failed poi")
        print(df1_log.describe().round(2))
        print("-----------------------")
        print("Feats diff succes-failed poi")
        print(df0_log.describe().round(2)-df1_log.describe().round(2))

    fname = f"/givecredit/RandomForestSK/CCHVAE/trials/paths_trial_0.csv"
    csv_loc = expnam+fname
    df = pd.read_csv(csv_loc)
    cols = df.columns.tolist()
    #cols = cols[-4:]
    df = df[cols]
    df = df.drop(columns=["path_order","path_num"])
    df0 = df.copy()
    df0 = df0.groupby(['poi_index']).mean().reset_index()
    data_interface = get_dataset_interface_by_name("give credit")
    data_interface.encode_data()
    
    
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data(.15, .15, random_state=0)
    df0 = feats_test.copy()[feats_test.index.isin(df0["poi_index"].values)]
    with pd.option_context('display.max_columns', 2000):
        
        print(df0.describe().round(2))
        print("-----------------------")
        print("Feats succes poi")
        print(dfall_log.describe().round(2))
        print("-----------------------")
        print(df0.describe().round(2)-dfall_log.describe().round(2))
        
    
    data_interface = get_dataset_interface_by_name("give credit")
    data_interface.encode_data()
    data_interface.scale_data("Standard")
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data(.15, .15, random_state=0)
    model_args = {"name": "LogisticRegressionSK", "load model": False, "save model": False}
    model_args["feats_train"], model_args["feats_valid"], model_args["labels_valid"] = feats_train, feats_valid, labels_valid
    model_args['random seed'] = 0 
    model_interface = get_model_interface_by_name(**model_args) 
    model_interface.fit(feats_train, labels_train)
    preds = pd.Series(model_interface.predict(
        feats_test), index=feats_test.index)
    probs_log = pd.Series(model_interface.predict_proba(
        feats_test,pos_label_only=True).flatten(), index=feats_test.index)
    feats_pos_log = feats_test.copy()[feats_test.index.isin(probs_log[probs_log>=0.7].index)]
    feats_neg_log = feats_test.copy()[feats_test.index.isin(probs_log[probs_log<0.5].index)]
    
    model_args = {"name": "RandomForestSK", "load model": False, "save model": False}
    model_args["feats_train"], model_args["feats_valid"], model_args["labels_valid"] = feats_train, feats_valid, labels_valid
    model_args['random seed'] = 0 
    model_interface = get_model_interface_by_name(**model_args) 
    model_interface.fit(feats_train, labels_train)
    preds = pd.Series(model_interface.predict(
        feats_test), index=feats_test.index)
    probs_rand = pd.Series(model_interface.predict_proba(
        feats_test,pos_label_only=True).flatten(), index=feats_test.index)
    feats_pos_rand = feats_test.copy()[feats_test.index.isin(probs_rand[probs_rand>=0.7].index)]
    scaler = data_interface.get_scaler()
    cols_to_scale = data_interface.get_scaled_features()
    scaled_feats_pos_log = feats_pos_log.copy()[cols_to_scale]
    scaled_feats_pos_log[cols_to_scale] = scaler.inverse_transform(
        scaled_feats_pos_log)
    scaled_feats_pos_rand = feats_pos_rand.copy()[cols_to_scale]
    scaled_feats_pos_rand[cols_to_scale] = scaler.inverse_transform(
        scaled_feats_pos_rand)
    feats_neg_log = feats_neg_log.copy()[cols_to_scale]
    feats_neg_log[cols_to_scale] = scaler.inverse_transform(
        feats_neg_log)
    
    with pd.option_context('display.max_columns', 2000):
        print("Unscaled Feats with logreg prob >= .7")
        print(scaled_feats_pos_log.describe().round(2))
        print("-----------------------")
        print("Unscaled Feats with randforest prob >= .7")
        print(scaled_feats_pos_rand.describe().round(2))
        print("-----------------------")
        print("Unscaled Feats difference log-rand")
        print(scaled_feats_pos_log.describe().round(2)-scaled_feats_pos_rand.describe().round(2))
        print("-----------------------")
        print("Scaled Feats difference log-rand")
        print(feats_pos_log.describe().round(2)-feats_pos_rand.describe().round(2))

    print()
    print("Log vs Rand Pos probs >= 0.7")
    print("log >= 0.7", len(probs_log[probs_log>=0.7]))
    print("log >= 0.5", len(probs_log[probs_log>=0.5]))
    print(len(probs_log[probs_log>=0.7])/len(probs_log[probs_log>=0.5]))
    print("rand >= 0.7", len(probs_rand[probs_rand>=0.7]))
    print("rand >= 0.5", len(probs_rand[probs_rand>=0.5]))
    print(len(probs_rand[probs_rand>=0.7])/len(probs_rand[probs_rand>=0.5]))

    print(probs_log[probs_log<0.5])
    print(probs_rand[probs_rand<0.5])
    
    print(scaled_feats_pos_log["age"].quantile(np.linspace(.01, 1, 99, 0)))
    print(scaled_feats_pos_log["age"][scaled_feats_pos_log["age"]>=50])
    print(len(scaled_feats_pos_log["age"][scaled_feats_pos_log["age"]>=59])/len(scaled_feats_pos_log))
    print(len(feats_neg_log["age"][feats_neg_log["age"]<59])/len(feats_neg_log))