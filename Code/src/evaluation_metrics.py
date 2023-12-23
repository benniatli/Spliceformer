# +
import numpy as np
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.metrics import average_precision_score

def print_topl_statistics(y_true, y_pred):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:

        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]
        correct = np.size(np.intersect1d(idx_true, idx_pred))
        total = float(min(len(idx_pred), len(idx_true)))
        if top_length == 1:
            correct_1 = correct
            total_1 = total
        topkl_accuracy += [ correct/ total]
        threshold += [sorted_y_pred[-int(top_length*len(idx_true))]]

    auprc = average_precision_score(y_true, y_pred)
    threshold_print = [ "{:0.4f}".format(v) for v in threshold]
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
          np.round(topkl_accuracy[0],4), np.round(topkl_accuracy[1],4), np.round(topkl_accuracy[2],4),
          np.round(topkl_accuracy[3],4), np.round(auprc,4), threshold_print[0], threshold_print[1],
          threshold_print[2], threshold_print[3],correct_1,total_1, len(idx_true)))
    return (topkl_accuracy,[auprc],threshold)

def topk_statistics(y_true, y_pred,verbose=True):
    # Prints the following information: top-k statistics for k=0.5,1,2,4,
    # thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:

        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]

        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred)) \
                  / float(min(len(idx_pred), len(idx_true)))]
        threshold += [sorted_y_pred[-int(top_length*len(idx_true))]]

    
    if verbose:
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
              np.round(topkl_accuracy[0],4), np.round(topkl_accuracy[1],4), np.round(topkl_accuracy[2],4),
              np.round(topkl_accuracy[3],4), np.round(auprc,4), np.round(threshold[0],4), np.round(threshold[1],4),
              np.round(threshold[2],4), np.round(threshold[3],4), len(idx_true)))
    return (topkl_accuracy,threshold,len(idx_true))

def cross_entropy_2d(y_true,y_pred):
            eps = np.finfo(np.float32).eps
            return -np.sum(y_true[:, :, 0]*np.log(y_pred[:, :, 0]+eps) + y_true[:, :, 1]*np.log(y_pred[:, :, 1]+eps) + y_true[:, :, 2]*np.log(y_pred[:, :, 2]+eps))/(y_true.shape[0]*y_true.shape[1])
        
        
def kullback_leibler_divergence_2d(y_true,y_pred):
        eps = np.finfo(np.float32).eps
        return -np.mean((y_true[:, :, 0]*np.log(y_pred[:, :, 0]/(y_true[:, :, 0]+eps)+eps) + y_true[:, :, 1]*np.log(y_pred[:, :, 1]/(y_true[:, :, 1]+eps)+eps) + y_true[:, :, 2]*np.log(y_pred[:, :, 2]/(y_true[:, :, 2]+eps)+eps)))        
