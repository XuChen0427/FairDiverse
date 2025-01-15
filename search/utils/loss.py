import torch

MAXDOC = 50


def list_pairwise_loss(score_1, score_2, delta):
    acc = torch.Tensor.sum((score_1-score_2)>0).item()/float(score_1.shape[0])
    loss = -torch.sum(delta * torch.Tensor.log(1e-8+torch.sigmoid(score_1 - score_2)))/float(score_1.shape[0])
    return acc, loss


def ndcg_loss(score, div):
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


