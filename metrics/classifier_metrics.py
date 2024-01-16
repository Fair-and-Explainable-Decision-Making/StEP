from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report

def run_classifier_tests(labels_true, pred, verbose = False):
    if verbose:
        print(confusion_matrix(labels_true,pred))
        print(classification_report(labels_true,pred))
    return accuracy_score(labels_true,pred), f1_score(labels_true,pred), confusion_matrix(labels_true,pred)