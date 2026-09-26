import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out 


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        print(logits)
        print(Yg)
        loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)
        print(loss)
        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        print(p)
        Yg = torch.gather(logits, 1, torch.unsqueeze(targets, 1))
        print(Yg)
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.FloatTensor)

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

class Rbfsens(nn.Module):
    def __init__(self, nfeat, args, trainset_size):
        super(Rbfsens,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.GNN = get_model(nfeat,args)
        self.classifier = NormedLinear(nhid,2)
        

        G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
        

        self.args = args
        self.criterion = TruncatedLoss(trainset_size=trainset_size)

        self.G_loss = 0

    def forward(self,g,x):
        z = self.GNN(g,x)
        y = self.classifier(z)
        return y
    
    def optimize(self,g,x,labels,idx_train,sens,idx_sens_train,epoch):
        self.train()

        ### update E, G
        

        
        
        self.optimizer_G.zero_grad()

        
        h = self.GNN(g,x)
        y = self.classifier(h)
        if (epoch+1) % 10 == 0:
            self.criterion.update_weight(y[idx_train].float(), labels[idx_train],idx_train)


        self.G_loss = self.criterion(y[idx_train].float(),labels[idx_train],idx_train)
                       
        
        
        self.G_loss.backward()
    
        self.optimizer_G.step()