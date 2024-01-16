
#TODO: fix the mess of imports
from operator import index

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

#binary should be full pipeline data->model->recourse->eval

def main(data_interface, model_choice = "BaselineDNN"):
    start = time.time()
    data_interface.encode_data()
    data_interface.scale_data()
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = data_interface.split_data()
    if model_choice == "BaselineDNN":
        model = PyTorchModel(BaselineDNN(feats_train.shape[1]),validation_features=feats_valid,
                             validation_labels=labels_valid, batch_size=1, epochs=10, lr=0.002)
    elif model_choice == "LogisticRegressionPT":
        model = PyTorchModel(LogisticRegressionPT(feats_train.shape[1]),validation_features=feats_valid,
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
    num_clusters = 1
    max_iter = 50
    step_rec = StEPRecourse(mi, data_interface, num_clusters, max_iter, confidence_threshold=0.7, random_seed=42, directions_rescaler="constant step size",step_size=0.1)
    preds = pd.Series(mi.predict(feats_test), index = feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index]
    recourse_results = neg_data.apply(lambda row: calc(row,step_rec,mi), axis=1)
    print(recourse_results)
    end = time.time()
    print("Model training took", start1-start, "seconds")
    print("Recourse took", end-start1, "seconds")
    print("Total it took", end-start, "seconds")

def calc(poi,step_rec,mi):
    poi = poi.to_frame().T
    cfs = step_rec.get_counterfactuals(poi)
    paths = step_rec.get_paths(poi)
    
    l2_path = []
    l2_prox = []
    for i, p in enumerate(paths):
        l2_path.append(recourse_metrics.compute_norm_path(p,2))
        l2_prox.append(recourse_metrics.compute_norm(poi,p[-1],ord=2))
    failures = len([j for j in mi.predict_proba(
        pd.concat(cfs, ignore_index=True)).flatten() if j < .5])
    return (np.mean(l2_path), np.mean(l2_prox), 
            recourse_metrics.compute_diversity(cfs), failures)

if __name__ == "__main__":
    """df = create_synthetic_data(5000)
    cols = list(df.columns)
    label_column = cols[-1]
    continuous_features = cols[:3]
    ordinal_features = [cols[4]]
    categorical_features = cols[3:5]
    immutable_features =  [cols[3]]
    main(df, [])"""

    continuous_features=["LIMIT_BAL", "AGE"]+[f"PAY_{i}" for i in range(0, 7)]+[f"BILL_AMT{i}" for i in range(1, 7)]+[f"PAY_AMT{i}" for i in range(1, 7)]
    ordinal_features=[]
    categorical_features=["SEX","EDUCATION"]
    immutable_features=["SEX"]
    label_column="default payment next month"
    positive_label=0
    file_path = (
    "data/datasets/default of credit card clients.xls")
    di = DataInterface(None, file_path, continuous_features, ordinal_features, 
                       categorical_features, immutable_features, label_column,
                       pos_label=positive_label, file_header_row=1,dropped_columns=["ID","MARRIAGE"])
    main(di,model_choice = "LogisticRegressionSK")