
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
import metrics.recourse_metrics as metrics
from sklearn.metrics import accuracy_score
import numpy as np

#binary should be full pipeline data->model->recourse->eval

def main(di):
    
    di.encode_data()
    di.scale_data()
    feats_train, feats_valid, feats_test, labels_train, labels_valid, labels_test = di.split_data()
    model = PyTorchModel(LogisticRegressionPT(feats_train.shape[1]),validation_features=feats_valid,
                         validation_labels=labels_valid, batch_size=1, epochs=1, lr=0.002)
    m1 = LogisticRegression(class_weight="balanced")
    #m1.fit(feats_train,labels_train)
    mi = ModelInterface(m1)
    mi.fit(feats_train, labels_train)
    
    num_clusters = 5
    max_iter = 50
    step_rec = StEPRecourse(mi, di, num_clusters, max_iter, confidence_threshold=0.7, random_seed=42, directions_rescaler="constant step size",step_size=1.0)
    preds = pd.Series(mi.predict(feats_test), index = feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index]
    l2_all_poi = []
    diver_li = []
    failures = 0
    for i, poi in neg_data.iterrows():
        poi = poi.to_frame().T
        cfs = step_rec.get_counterfactuals(poi)
        
        paths = step_rec.get_paths(poi)
        l2 = []
        for i, p in enumerate(paths):
            l2.append(metrics.compute_norm_path(p,2))
        l2_all_poi.append(np.mean(l2))
        diver_li.append(metrics.compute_diversity(cfs))
        failures += len([j for j in mi.predict_proba(pd.concat(cfs, ignore_index=True)).flatten() if j < .5])
    print(np.mean(l2_all_poi))
    print(np.mean(diver_li))
    print(failures)

if __name__ == "__main__":
    """df = create_synthetic_data(5000)
    cols = list(df.columns)
    label_column = cols[-1]
    continuous_features = cols[:3]
    ordinal_features = [cols[4]]
    categorical_features = cols[3:5]
    immutable_features =  [cols[3]]
    main(df, [])"""

    continuous_features=["LIMIT_BAL", "AGE"]+[f"PAY_{i}" for i in range(1, 7)]+[f"BILL_AMT{i}" for i in range(1, 7)]+[f"PAY_AMT{i}" for i in range(1, 7)]
    ordinal_features=[]
    categorical_features=[]
    immutable_features=[]
    label_column="default payment next month"
    positive_label=0
    file_path = (
    "data/datasets/default of credit card clients.xls")
    di = DataInterface(None, file_path, continuous_features, ordinal_features, 
                       categorical_features, immutable_features, label_column,
                       pos_label=positive_label, file_header_row=1,dropped_columns=["ID"])
    main(di)