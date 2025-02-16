import torch

MAXDOC = 50


def list_pairwise_loss(score_1, score_2, delta):
    """
    Computes the accuracy and list-pairwise loss based on the difference between two scores. 
    
    :param score_1: A tensor containing the scores for the first set of items.
    :param score_2: A tensor containing the scores for the second set of items.
    :param delta: A tensor representing the weight for each pairwise comparison.
    :return: The accuracy and the computed list-pairwise loss.
    """
    acc = torch.Tensor.sum((score_1-score_2)>0).item()/float(score_1.shape[0])
    loss = -torch.sum(delta * torch.Tensor.log(1e-8+torch.sigmoid(score_1 - score_2)))/float(score_1.shape[0])
    return acc, loss


def ndcg_loss(score, div):
    """
    Computes the NDCG (Normalized Discounted Cumulative Gain) loss for ranking tasks. 
    
    :param score: A tensor containing the predicted scores for each document.
    :param div: A tensor containing the relevance or diversity values for each document.
    :return: The computed NDCG loss.
    """
    T = 0.1
    temp_list = [(score[i,:] - score[i,:].reshape(1,MAXDOC).permute(1,0))/T for i in range(score.shape[0])]
    temp = torch.stack(temp_list, dim=0).float()
    sigmoid_temp = torch.sigmoid(temp)
    C_li = torch.bmm(sigmoid_temp, div) - 0.5 * div
    pw = torch.pow(0.5, C_li)
    top = torch.mul(div, pw)
    top = torch.sum(top, dim=-1)
    R_i = 0.5 + torch.sum(sigmoid_temp, dim = -1)
    bottom = torch.log2(1+R_i)
    #loss = 1.0 / (1e-6 + torch.sum(top/bottom))
    loss = - torch.sum(top/bottom)
    print('loss = ', loss)
    return loss


