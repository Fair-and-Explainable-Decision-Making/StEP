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
from sklearn.manifold import TSNE
import scipy.stats as st

# binary should be full pipeline data->model->recourse->eval


def main(data_interface: DataInterface, model_choice="BaselineDNN"):
    print("-------------")
    start = time.time()
    data_interface.encode_data()
    data_interface.scale_data("Standard")
    print(data_interface.get_data()["EDUCATION"].value_counts())
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
    neg_data = feats_test.loc[preds[preds != 1].index].head(10)
    print("cluster info")
    clus, cent = step_rec.get_clusters()
    print(clus.shape)
    print(clus.value_counts())
    print(cent)
    print(recourse_metrics.compute_diversity(cent))
    pca = TSNE(n_components=2)
    
    neg_data_r = pca.fit_transform(feats_train)
    print(neg_data_r)
    neg_data_r_df = pd.DataFrame(neg_data_r, columns = ['x','y'],index=feats_train.index)
    neg_data_r_df = neg_data_r_df.join(clus).dropna()
    print(neg_data_r_df)
    sns.scatterplot(data=neg_data_r_df, x="x", y="y", hue="datapoint_cluster")
    plt.savefig("clust.pdf")
    plt.show()
    
    
    recourse_results = neg_data.apply(
        lambda row: generate_recourse_results(row, step_rec, mi), axis=1)
    df_results = pd.DataFrame.from_dict(
        dict(zip(recourse_results.index, recourse_results.values))).T
    df_results.rename(columns={0: "l2_path_len", 1: "l2_prox", 2: "l2_path_steps",
                      3: "poi_id", 4: "failures", 5: "diversity"}, inplace=True)
    df_results = df_results.explode(df_results.columns.values.tolist())
    # print(df_results.groupby('poi_id').mean()[['l2_path_len', 'l2_prox', 'l2_path_steps']])
    print(df_results)
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
    #cfs = recourse_interface.get_counterfactuals(poi)
    paths = recourse_interface.get_paths(poi)
    cfs = recourse_interface.get_counterfactuals_from_paths(paths)
    """print(cfs)
    print("________________________")"""
    

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
    """df = create_synthetic_data(5000)
    cols = list(df.columns)
    label_column = cols[-1]
    continuous_features = cols[:3]
    ordinal_features = [cols[4]]
    categorical_features = cols[3:5]
    immutable_features =  [cols[3]]
    main(df, [])"""
    start = time.time()
    continuous_features = ["LIMIT_BAL", "AGE"]+[f"PAY_{i}" for i in range(
        1, 7)]+[f"BILL_AMT{i}" for i in range(1, 7)]+[f"PAY_AMT{i}" for i in range(1, 7)]
    ordinal_features = ["EDUCATION"]
    ordinal_features_order = {"EDUCATION": [5,4,3, 2, 1]}
    unidirection_features = [[], ["EDUCATION"]]
    categorical_features = ["SEX", "MARRIAGE"]
    immutable_features = ["SEX", "MARRIAGE","AGE"]
    label_column = "default payment next month"
    positive_label = 0
    file_path = (
        "data/datasets/default of credit card clients.xls")
    df = pd.read_excel(file_path, header=1)
    print(df.shape)
    
    """df = df[df["EDUCATION"] < 4]
    df = df[df["EDUCATION"] != 0]"""
    
        
    df.loc[df['EDUCATION'].isin([0,6]), "EDUCATION"] = 5

    df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)

    for pay_column in [f"PAY_{i}" for i in range(1, 7)]:
        df.loc[df[pay_column] < 0, pay_column] = 0
    print(df.shape)
    print(df.head(10))
    print(df[label_column].value_counts())
    print(df.head(10))
    print(df["EDUCATION"].value_counts())
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
    end=time.time()
    print("Total trials it took", end-start, "seconds")
"""   l2_path_len   l2_prox  l2_path_steps    poi_id  failures  diversity
0     6.412940  5.916325       5.753333  13360.57       0.0  20.313675
1     5.271778  4.829287       4.600000  15346.54       0.0  16.037583
2     5.895131  5.401101       5.226667  14084.27       0.0  18.583459
3     5.621173  5.128871       4.966667  14227.87       0.0  11.970823
4     5.455727  4.963547       4.806667  15280.67       0.0  11.980956
5     5.103489  4.625960       4.446667  14641.22       0.0  11.566918
6     6.306035  5.819355       5.633333  13710.64       0.0  19.903318
7     6.106217  5.615165       5.440000  15158.13       0.0  19.297862
8     6.107909  5.611600       5.436667  13473.59       0.0  19.170610
9     5.723530  5.254506       5.046667  13415.14       0.0  17.838251
l2_path_len          5.800393
l2_prox              5.316572
l2_path_steps        5.135667
poi_id           14269.864000
failures             0.000000
diversity           16.666346
datapoint_cluster
0                    2509
1                     641
2                       3
dtype: int64

   l2_path_len   l2_prox  l2_path_steps    poi_id  failures  diversity
0     3.915446  3.606614       3.213333  14706.87       0.0   7.291955
1     4.316808  4.014473       3.613333  14485.31       0.0   7.270343
2     3.657273  3.372227       2.953333  15206.71       0.0   6.986286
3     3.869785  3.570504       3.170000  14523.81       0.0   6.970675
4     3.853679  3.538208       3.156667  14832.93       0.0   7.242836
5     3.908230  3.594788       3.200000  13679.61       0.0   7.391750
6     4.138090  3.815509       3.436667  15850.74       0.0   7.825332
7     3.653037  3.362405       2.950000  16454.12       0.0   6.991980
8     3.988307  3.679640       3.290000  14062.75       0.0   7.196401
9     3.882212  3.586355       3.176667  14519.60       0.0   7.229340
l2_path_len          3.918287
l2_prox              3.614072
l2_path_steps        3.216000
poi_id           14832.245000
failures             0.000000"""

"""
   l2_path_len   l2_prox  l2_path_steps        poi_id  failures  diversity
0     3.864591  3.557796       3.163339  14497.226860       0.0   7.223542
1     3.873967  3.571242       3.169476  14582.450754       0.0   7.240687
2     3.780314  3.479454       3.075812  14555.834838       0.0   7.255691
3     3.889203  3.583980       3.186620  14528.613556       0.0   7.195526
4     3.891650  3.586259       3.187708  14558.191644       0.0   7.373703
5     3.798805  3.502963       3.095195  14191.590868       0.0   7.168242
6     3.873921  3.569077       3.170841  14944.787120       0.0   7.341374
7     3.930957  3.624167       3.227729  14681.146018       0.0   7.407660
8     3.772145  3.474245       3.068354  14959.816754       0.0   7.181096
9     3.785497  3.486673       3.081315  14806.092561       0.0   7.046005
l2_path_len          3.846105
l2_prox              3.543586
l2_path_steps        3.142639
poi_id           14630.575097
failures             0.000000
diversity            7.243353
dtype: float64
Total trials it took 1770.897633075714 seconds
datapoint_cluster
0                    2596
1                     675
2                     243
dtype: int64
"""