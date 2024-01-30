# TODO: fix the mess of imports
from operator import index

from dice_ml import Data

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
    data_interface.encode_data()
    data_interface.scale_data(scaling_method="Standard")
    print(data_interface.get_data())
    
    start = time.time()
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
    print(feats_train)
    mi.fit(feats_train, labels_train)
    for score in classifier_metrics.run_classifier_tests(labels_test, mi.predict(feats_test)):
        print(score)
    
    print("filter")
    #print(feats_train.loc[(feats_train[col1] >= 0.95) & (feats_train['Con2'] <= 0.05)])
    special_cluster_feats = feats_train.copy()
    """loc[((feats_train[col1] >= 0.95) & (feats_train['Con2'] <= 0.05))
                                            |  ((feats_train[col1] >= 0.95) & ((feats_train['Con2'] <= 0.5) & (feats_train['Con2'] >= 0.4)))]"""
    special_cluster_labels = labels_train.loc[special_cluster_feats.index]
    special_cluster_data = [special_cluster_feats, special_cluster_labels]
    
    start1 = time.time()
    num_clusters = 1
    max_iter = 50
    step_rec = StEPRecourse(mi, data_interface, num_clusters, max_iter, confidence_threshold=0.9,
                            directions_rescaler="constant step size", step_size=.1,special_cluster_data=special_cluster_data)
    preds = pd.Series(mi.predict(feats_test), index=feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index]
    
    #TODO: do this for cats, ords, and immuts
    col1, col2 = feats_train.columns.values[0], feats_train.columns.values[1]
    special_poi = neg_data.head(1)#pd.DataFrame(np.array([[0.4, 0]]),columns=[col1, col2])
    cfs = step_rec.get_counterfactuals(special_poi)
    paths = step_rec.get_paths(special_poi)
    print("path")
    print(paths[0])
    
    clus, cent = step_rec.get_clusters()
    li = []
    for p_i, p in enumerate(paths):
        for i, row in enumerate(p):
            if i == 0:
                row = np.append(row.values , "poi"+str(p_i))
            elif i== len(paths[0])-1:
                row = np.append(row.values , "cf"+str(p_i))
            else:
                row = np.append(row.values , "path"+str(p_i))
            li.append(row)
    neg_data_r_df = pd.DataFrame(li, columns=[col1, col2, 'data type'])
    li = []
    for row in feats_train.loc[clus.index].values:
        row =np.append(row, "clust")
        li.append(row)
    print(cent)
    for row in cent:
        row =np.append(row , "c_center")
        li.append(row)
    feats_train_clus_r_df = pd.DataFrame(li, columns=[col1, col2, 'data type'])
    print(feats_train_clus_r_df.dtypes)
    df = pd.concat([neg_data_r_df,feats_train_clus_r_df],ignore_index=True)
    df[col1]=df[col1].astype('float')
    df[col2]=df[col2].astype('float')
    print(df)
    df_clus_points = df.loc[(df['data type'] == "clust")]
    df_except_clusp = df.loc[(df['data type'] != "clust")]
    clr = np.array(sns.color_palette("deep"))
    sns.scatterplot(data=df_clus_points, x=col1, y=col2, hue="data type",alpha=0.25,palette=clr)
    sns.scatterplot(data=df_except_clusp, x=col1, y=col2, hue="data type",palette=clr[1:])
    trained_model = mi.get_model()
    b = trained_model.intercept_[0]
    w1, w2 = trained_model.coef_.T
    c = -b/w2
    m = -w1/w2
    xmin, xmax = np.min(feats_train.values[:, 0]), np.max(feats_train.values[:, 0])
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    
    for cf in cfs:
        print("------------")
        print(cf)
        print(mi.predict_proba(cf,pos_label_only=True))
    plt.savefig("test.pdf")
    return
    

def generate_recourse_results(poi,recourse_interface,model_interface):                
    poi = poi.to_frame().T
    cfs = recourse_interface.get_counterfactuals(poi)
    paths = recourse_interface.get_paths(poi)

    l2_path_len = []
    l2_prox = []
    l2_path_steps = []
    failures = []
    for i, p in enumerate(paths):
        if model_interface.predict(p[-1]) == 1:
            l2_path_len.append(recourse_metrics.compute_norm_path(p,2))
            l2_prox.append(recourse_metrics.compute_norm(poi,p[-1],ord=2))
            l2_path_steps.append(len(p[1:]))
            failures.append(0)
        else:
            failures.append(1)
    return [l2_path_len, l2_prox, l2_path_steps,
            list(poi.index.values)*len(paths), failures, 
            [recourse_metrics.compute_diversity(cfs)]*len(paths)]

if __name__ == "__main__":
    df = create_synthetic_data(5000, num_con_feat=2, num_ord_cat_feat = 0,
                          ord_cat_num_unique = 0, num_binary_cat_feat = 0,
                          w = [1.0, -0.5])
    cols = list(df.columns)
    label_column = cols[-1]
    continuous_features = [cols[0],cols[1]]
    ordinal_features = []
    categorical_features = []
    immutable_features =  []
    unidirection_features = [[],[]]
    ordinal_features_order = None
    positive_label = 1
    print(df)
    print(df['Target'].mean())

            
    num_trails= 1
    results = []
    for i in range(num_trails):
        di = DataInterface(df, None, continuous_features, ordinal_features,
                       categorical_features, immutable_features, label_column,
                       pos_label=positive_label, unidirection_features=unidirection_features, ordinal_features_order=ordinal_features_order)
        main(di, model_choice="LogisticRegressionSK")
