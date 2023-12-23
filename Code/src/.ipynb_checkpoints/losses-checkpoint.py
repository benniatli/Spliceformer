# +
import torch

class categorical_crossentropy_2d:
    def __init__(self, weights=None,mask=False):
        self.weights = weights
        self.mask = mask
        self.eps = torch.finfo(torch.float32).eps
        
    def loss(self,y_pred,y_true):
        if self.mask:
            loss_sum = torch.sum(self.weights[0]*y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+self.eps) + self.weights[1]*y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+self.eps) + self.weights[2]*y_true[:, 2, :]*torch.log(y_pred[:, 2, :]+self.eps))
            weight_sum = torch.sum(self.weights[0]*y_true[:, 0, :] + self.weights[1]*y_true[:, 1, :] + self.weights[2]*y_true[:, 2, :])+self.eps
            return -loss_sum/weight_sum
        else:
            prob_sum = torch.sum(y_true)
            return -torch.sum(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+self.eps) + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+self.eps) + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]+self.eps))/prob_sum
        #loss_sum = torch.sum(self.weights[0]*y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+1e-10) + self.weights[1]*y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+1e-10) + self.weights[2]*y_true[:, 2, :]*torch.log(y_pred[:, 2, :]+1e-10))
        #weight_sum = torch.sum(self.weights[0]*y_true[:, 0, :] + self.weights[1]*y_true[:, 1, :] + self.weights[2]*y_true[:, 2, :])
        #return -loss_sum/weight_sum
        
class binary_crossentropy_2d:
    def __init__(self):
        self.eps = torch.finfo(torch.float32).eps
        
    def loss(self,y_pred,y_true):
        loss = torch.mean(y_true*torch.log(y_pred+self.eps) + (1-y_true)*torch.log(1-y_pred+self.eps))
        return -loss
    


# -

class kl_div_2d:
    def __init__(self,temp=1):
        self.eps = torch.finfo(torch.float32).eps
        self.temp = temp
        
    def loss(self,y_pred,y_true):
        if self.temp!=1:
            y_true = torch.nn.Softmax(dim=1)(torch.log(y_true+self.eps)/self.temp)
        return -torch.mean((y_true[:, 0, :]*torch.log(y_pred[:, 0, :]/(y_true[:, 0, :]+self.eps)+self.eps) + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]/(y_true[:, 1, :]+self.eps)+self.eps) + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]/(y_true[:, 2, :]+self.eps)+self.eps))*self.temp**2)
        #x = -torch.sum(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]/(y_true[:, 0, :]+self.eps)+self.eps) + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]/(y_true[:, 1, :]+self.eps)+self.eps) + y_true[:, 2, :]*torch.log(y_pred[:, 2, :]/(y_true[:, 2, :]+self.eps)+self.eps))*self.temp**2
        #return x/(y_pred.shape[0]*y_pred.shape[2])
