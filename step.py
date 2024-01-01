
#TODO: fix the mess of imports
from models.model_interface import ModelInterface
from models.pytorch_wrapper import PyTorchModel
from models.pytorch_models.dnn_basic import BaselineDNN

from data.data_interface import DataInterface
from data.synthetic_data import create_synthetic_data
import pandas as pd
from recourse.recourse_interface import RecourseInterface
from recourse.step_lib import StEP, StEPRecourse
from typing import Optional

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
    model = PyTorchModel(BaselineDNN(feats_train.shape[1]), batch_size=1)
    mi = ModelInterface(model)
    mi.fit(feats_train, labels_train)

if __name__ == "__main__":
    main()