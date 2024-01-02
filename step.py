
#TODO: fix the mess of imports
from operator import index
from models.model_interface import ModelInterface
from models.pytorch_wrapper import PyTorchModel
from models.pytorch_models.dnn_basic import BaselineDNN
from models.pytorch_models.logreg import LogisticRegression

from data.data_interface import DataInterface
from data.synthetic_data import create_synthetic_data
import pandas as pd
from recourse.recourse_interface import RecourseInterface
from recourse.step_lib import StEP, StEPRecourse
from typing import Optional
import metrics.recourse_metrics as metrics

#binary should be full pipeline data->model->recourse->eval

def main():
    
    df = create_synthetic_data(5000)
    cols = list(df.columns)
    targ = cols[-1]
    cont = cols[:3]
    ord = [cols[4]]
    cat = cols[3:5]
    imm =  [cols[3]]
    
    di = DataInterface(df, None, cont, ord, cat, imm, targ)
    di.encode_data()
    di.scale_data()
    feats_train, feats_test, labels_train, labels_test = di.split_data()
    model = PyTorchModel(LogisticRegression(feats_train.shape[1]), batch_size=1, epochs=1)
    mi = ModelInterface(model)
    mi.fit(feats_train, labels_train)
    num_clusters = 1
    max_iter = 5
    step_rec = StEPRecourse(mi, di, num_clusters, max_iter, confidence_threshold=0.51, random_seed=42)
    step_rec1 = StEPRecourse(mi, di, num_clusters, max_iter, confidence_threshold=0.51, random_seed=42, step_size=1.0)
    preds = pd.Series(mi.predict(feats_test), index = feats_test.index)
    neg_data = feats_test.loc[preds[preds != 1].index]
    poi = neg_data.iloc[:1]
    cfs = step_rec.get_counterfactuals(poi)
    cfs1 = step_rec1.get_counterfactuals(poi)
    
    paths = step_rec.get_paths(poi)
    paths1 = step_rec1.get_paths(poi)
    print("-----------------------")
    print(poi)
    for i, p in enumerate(paths):
        print("Path ", i)
        print(p)
        print("L0 ",metrics.compute_norm_path(p,0))
        print("L2 ",metrics.compute_norm_path(p,2))
        print("----")
    
    print("STEP SIZE")
    for i, p in enumerate(paths1):
        print("Path ", i)
        print(p)
        print("L0 ",metrics.compute_norm_path(p,0))
        print("L2 ",metrics.compute_norm_path(p,2))
        print("----")
    print(metrics.compute_diversity(cfs))
    print(cfs)
    print(cfs1)
    print(mi.predict_proba(pd.concat(cfs, ignore_index=True)))
    print(mi.predict_proba(pd.concat(cfs1, ignore_index=True)))

if __name__ == "__main__":
    main()