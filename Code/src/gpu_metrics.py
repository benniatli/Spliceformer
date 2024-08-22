import torch
import numpy as np
from tqdm import tqdm


def torch_intersect(a,b):
    a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)
    intersection = a_cat_b[torch.where(counts.gt(1))]
    return intersection

def topk_statistics_cuda(y_true, y_pred):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.
    idx_true = torch.nonzero(y_true == 1)[:,0]
    sorted_y_pred, argsorted_y_pred = torch.sort(y_pred)
    top_length = 1
    idx_pred = argsorted_y_pred[-int(top_length*idx_true.size()[0]):]
    correct = (torch_intersect(idx_true, idx_pred)).size()[0]
    total = float(min(len(idx_pred), len(idx_true)))
    topk_accuracy = correct/ total
    threshold = sorted_y_pred[-int(top_length*len(idx_true))]
    return (topk_accuracy,threshold)

def average_precision_score(targets, predictions, device):
    """
    Calculate the average precision score given the target and prediction tensors.
    
    Args:
    targets (torch.Tensor): Tensor of ground truth binary labels (0s and 1s).
    predictions (torch.Tensor): Tensor of predicted probabilities (floats between 0 and 1).
    
    Returns:
    float: Average precision score.
    """
    # Sort by predicted score
    sorted_indices = torch.argsort(predictions, descending=True)
    sorted_targets = targets[sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    # Calculate cumulative sums of true positives and false positives
    cum_true_positives = torch.cumsum(sorted_targets, dim=0)
    cum_false_positives = torch.cumsum(1 - sorted_targets, dim=0)

    # Calculate precision and recall
    precision = cum_true_positives / (cum_true_positives + cum_false_positives)
    recall = cum_true_positives / cum_true_positives[-1]

    # Insert a zero at the beginning of recall to calculate differences
    recall = torch.cat([torch.tensor([0.0]).to(device), recall])
    precision = torch.cat([torch.tensor([0.0]).to(device), precision])

    # Calculate average precision
    average_precision = torch.sum((recall[1:] - recall[:-1]) * precision[1:])

    return average_precision.item()

def resample(tensor, replace=True, n_samples=None, random_state=None,device=None):
    """
    Resample the input tensor with or without replacement.

    Args:
    tensor (torch.Tensor): The input tensor to resample.
    replace (bool): Whether to sample with replacement. Default is True.
    n_samples (int): Number of samples to draw. If None, defaults to the size of the input tensor.
    random_state (int): Seed for the random number generator. Default is None.

    Returns:
    torch.Tensor: The resampled tensor.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples = n_samples or tensor.size(0)
    indices = torch.randint(0, tensor.size(0), (n_samples,), dtype=torch.long, device=device) if replace else torch.randperm(tensor.size(0))[:n_samples]
    
    return indices

#prc_auc = (aps_acceptor+aps_donor)/2
def calculate_topk(y_true_acceptor,y_pred_acceptor,y_true_donor,y_pred_donor):
    topk_acceptor,_ = topk_statistics_cuda(y_true_acceptor, y_pred_acceptor)
    topk_donor,_ = topk_statistics_cuda(y_true_donor, y_pred_donor)
    return (topk_acceptor+topk_donor)/2

def calculate_ap(y_true_acceptor,y_pred_acceptor,y_true_donor,y_pred_donor,device0,device1):
    ap_acceptor = average_precision_score(y_true_acceptor, y_pred_acceptor,device0)
    ap_donor = average_precision_score(y_true_donor, y_pred_donor,device1)
    return (ap_acceptor+ap_donor)/2

def run_bootstrap(Y_true_acceptor,Y_pred_acceptor,Y_true_donor,Y_pred_donor,n_bootstraps = 1000):
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    Y_true_acceptor_cuda = torch.as_tensor(Y_true_acceptor, dtype=torch.int8).to(device0)
    Y_pred_acceptor_cuda = torch.as_tensor(Y_pred_acceptor, dtype=torch.float32).to(device0)
    Y_true_donor_cuda = torch.as_tensor(Y_true_donor, dtype=torch.int8).to(device1)
    Y_pred_donor_cuda = torch.as_tensor(Y_pred_donor, dtype=torch.float32).to(device1)
    topk_score = calculate_topk(Y_true_acceptor_cuda,Y_pred_acceptor_cuda,Y_true_donor_cuda,Y_pred_donor_cuda)
    ap_score = calculate_ap(Y_true_acceptor_cuda,Y_pred_acceptor_cuda,Y_true_donor_cuda,Y_pred_donor_cuda,device0,device1)
    print(topk_score,ap_score)
    ap_scores = []
    topk_scores = []
    for i in tqdm(range(n_bootstraps)):
        resampled_indices = resample(Y_true_acceptor_cuda, random_state=i,device=device2)
        resampled_y_true_acceptor = Y_true_acceptor_cuda[resampled_indices]
        resampled_y_pred_acceptor = Y_pred_acceptor_cuda[resampled_indices]
        resampled_y_true_donor = Y_true_donor_cuda[resampled_indices]
        resampled_y_pred_donor = Y_pred_donor_cuda[resampled_indices]
        
        del resampled_indices
        torch.cuda.empty_cache()
        
        topk = calculate_topk(resampled_y_true_acceptor,resampled_y_pred_acceptor,resampled_y_true_donor,resampled_y_pred_donor)
        ap = calculate_ap(resampled_y_true_acceptor,resampled_y_pred_acceptor,resampled_y_true_donor,resampled_y_pred_donor,device0,device1)
        
        ap_scores.append(ap)
        topk_scores.append(topk)
        
    ci_lower = np.percentile(ap_scores, 2.5)
    ci_upper = np.percentile(ap_scores, 97.5)
    print(f"average precision score = {ap_score} (95% confidence interval: [{ci_lower}, {ci_upper}])")
    ci_lower = np.percentile(topk_scores, 2.5)
    ci_upper = np.percentile(topk_scores, 97.5)
    print(f"topk score = {topk_score} (95% confidence interval: [{ci_lower}, {ci_upper}])")