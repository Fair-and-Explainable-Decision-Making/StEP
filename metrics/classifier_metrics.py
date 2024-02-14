from sklearn.metrics import accuracy_score, f1_score, confusion_matrix#, classification_report
import pandas as pd

def run_classifier_tests(labels_true, pred, verbose=False):
    conf_matrix = confusion_matrix(labels_true, pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    neg_acc = tn/(tn+fp)
    pos_acc = tp/(tp+fn)
    acc = accuracy_score(labels_true, pred)
    f1 =f1_score(labels_true, pred)
    return pd.DataFrame([[acc, neg_acc, pos_acc, f1]],columns=['acc', 'neg_acc', 'pos_acc', 'f1'])
