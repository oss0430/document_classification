import modules
import numpy as np
import os

def evaluate_confusion_matrix(
    confusion_matrix : np.array
):
    num_classes = confusion_matrix.shape[0]

    # Calculate true positives, false positives, false negatives, and true negatives for each class
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    for i in range(num_classes):
        tp[i] = confusion_matrix[i, i]
        fp[i] = np.sum(confusion_matrix[:, i]) - tp[i]
        fn[i] = np.sum(confusion_matrix[i, :]) - tp[i]
        tn[i] = np.sum(confusion_matrix) - tp[i] - fp[i] - fn[i]

    # Calculate precision, recall, and F1 score for each class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    def logical_div(a, b):
        return b and a/b or 0 
    
    for i in range(num_classes):
        precision[i] = logical_div(tp[i],tp[i] + fp[i])
        recall[i] = logical_div(tp[i],tp[i] + fn[i])
        f1_score[i] = 2 * logical_div((precision[i] * recall[i]), (precision[i] + recall[i]))

    # Calculate weighted F1 score
    weighted_f1_score = np.sum(f1_score * np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix))

    precision = precision.tolist()
    recall = recall.tolist()
    f1_score = f1_score.tolist()
    
    return {
        "precision" : precision,
        "recall" : recall,
        "f1_score" : f1_score,
        "weighted_f1_score" : weighted_f1_score
    }
    
    
def qualatative(
    dataset, algorithm
):
    result = []
    return result