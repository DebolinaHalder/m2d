import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter

def __kl_divergence(p, q):
    """
    Calculate KL-Divergence between P and Q, with epsilon to avoid divide by zero.
    :param p: PyTorch tensor p distribution.
    :param q: PyTorch tensor q distribution.
    :return: KL-Divergence score.
    """
    epsilon = 1e-7  # Epsilon is used here to avoid P or Q is equal to 0.
    p = p + epsilon
    q = q + epsilon

    return torch.sum(p * torch.log(p / q))


def NDKL(ranking, item_group_dict):
    """
    Calculate Normalized Discounted KL-Divergence Score (Geyik et al.).
    :param ranking: PyTorch tensor of ranking(s).
    :param item_group_dict: Dictionary of items (keys) and their group membership (values).
    :return: NDKL value.
    """
    if ranking.dim() > 1:
        raise AssertionError("NDKL can only be calculated on a single ranking.")

    single_ranking = ranking.squeeze()  # ensure 1D tensor
    single_ranking = single_ranking[~torch.isnan(single_ranking)]  # drop any NaNs

    group_ids = torch.tensor([item_group_dict[c.item()] for c in single_ranking])
    unique_grps = torch.unique(group_ids)
    group_ids = torch.tensor([torch.where(unique_grps == grp_of_item)[0][0] for grp_of_item in group_ids])
    num_groups = group_ids.max().item()
    num_items = len(group_ids)

    dr = __distributions(group_ids, num_groups)  # Distributions per group
    Z = __Z_Vector(num_items)  # Array of Z scores

    # Eq. 4 in Geyik et al.
    return (1 / Z.sum()) * sum(
        Z[i] * __kl_divergence(__distributions(group_ids[:i+1], num_groups), dr)
        for i in range(num_items)
    )

def __distributions(ranking, num_groups):
    """
    Calculate the proportion of each group
    :param ranking: PyTorch tensor of group id represented in the ranking.
    :param num_groups: Int, number of distinct groups
    :return: PyTorch tensor of each group's proportion.
    """
    return torch.tensor([(ranking == i).sum() / len(ranking) for i in range(num_groups + 1)])

def __Z_Vector(k):
    """
    Calculate Z score
    :param k: Int, position of ranking.
    :return: PyTorch tensor of Z values.
    """
    return 1 / torch.log2(torch.arange(k, dtype=torch.float)+2)







def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.num_hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.num_hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

class Baseline(nn.Module):

    def __init__(self, nfeat, args):
        super(Baseline,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.alpha = args.alpha
        
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,1)
        

        self.G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(self.G_params, lr = args.lr, weight_decay = args.weight_decay)
        

        self.args = args
        
        #self.criterion = nn.BCEWithLogitsLoss()
        
        self.criterion = nn.BCELoss()
        self.G_loss = 0
        

    def forward(self,g,x):
        z = self.GNN(g,x)
        y = self.classifier(z)
        y = torch.sigmoid(y)
        return y
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train):
        self.train()
        self.optimizer_G.zero_grad()
        h = self.GNN(g,x)
        y = self.classifier(h)
        y = torch.sigmoid(y)
        self.cls_loss = self.criterion(y[idx_train].float(),labels[idx_train].unsqueeze(1).float())
        indexes = (torch.argsort(y,dim=0,descending=True).squeeze(0)).view(-1)
        #print(indexes)
        item_group_dict = {}
        [item_group_dict.setdefault(c.item(),sens[c.item()].item()) for c in indexes]
        self.ndkl = NDKL(indexes,item_group_dict)
        self.G_loss = self.cls_loss + self.alpha * self.ndkl
        self.G_loss.backward()
        self.optimizer_G.step()

