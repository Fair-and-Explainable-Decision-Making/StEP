from models.model_interface import ModelInterface
from models.pytorch_wrapper import PyTorchModel
from models.pytorch_models.dnn_basic import BaselineDNN
from models.pytorch_models.logreg import LogisticRegression as LogisticRegressionPT
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_model_interface_by_name(**kwargs) -> ModelInterface:
    model_name = kwargs["name"]
    if model_name == "BaselineDNN":
        model = PyTorchModel(BaselineDNN(kwargs['feats_train'].shape[1]), validation_features=kwargs['feats_valid'],
                             validation_labels=kwargs['labels_valid'], batch_size=kwargs['batch_size'], epochs=kwargs['epochs'], lr=kwargs['lr'])
    elif model_name == "LogisticRegressionPT":
        model = PyTorchModel(LogisticRegressionPT(kwargs['feats_train'].shape[1]), validation_features=kwargs['feats_valid'],
                             validation_labels=kwargs['labels_valid'], batch_size=kwargs['batch_size'], epochs=kwargs['epochs'], lr=kwargs['lr'])
    elif model_name == "LogisticRegressionSK":
        model = LogisticRegression(class_weight="balanced")
    elif model_name == "RandomForestSK":
        model = RandomForestClassifier()
    else:
        raise Exception("Invalid model choice")
    return ModelInterface(model)