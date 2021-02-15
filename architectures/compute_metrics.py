from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
def compute_metrics(pred):
    ytrue = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(f"ytrue : {ytrue.shape}, ypred : {preds.shape}")
    precision, recall, f1, _ = precision_recall_fscore_support(ytrue, preds, average='binary')
    acc = accuracy_score(ytrue, preds)
    #conf_mat = confusion_matrix(ytrue, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        #'confusion_matrix' : conf_mat
    }