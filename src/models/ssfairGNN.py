import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
import torch
import torch.nn.functional as F

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

class SSFairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(SSFairGNN,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.GNN = get_model(nfeat,args)
        self.adv = nn.Linear(nhid,1)
        self.classifier = nn.Linear(nhid,1)

        self.optimizer_G = torch.optim.Adam(self.GNN.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.optimizer_A = torch.optim.Adam(self.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)

        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()

        self.GNN_loss = 0
        self.A_loss = 0

    def forward(self,g,x):
        z = self.GNN(g,x)
        return z
    
    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.args.tau)
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        refl_sim = f(torch.mm(z1, z1.t()))
        between_sim = f(torch.mm(z1, z2.t()))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    
    def loss(self, z1,z2):
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret
    
    def optimize(self,g,x,new_g1,new_x1,new_g2,new_x2,sens):
        self.train()
        self.adv.requires_grad_(False)
        self.optimizer_G.zero_grad()

        z1 = self.GNN(new_g1,new_x1)
        z2 = self.GNN(new_g2,new_x2)
        z_org = self.GNN(g,x)
        s_g = self.adv(z_org)
        self.cons_loss = self.loss(z1,z2)
        self.adv_loss = self.criterion(s_g,sens.unsqueeze(1).float())
        self.GNN_loss = self.cons_loss - self.args.alpha * self.adv_loss
        self.GNN_loss.backward()
        self.optimizer_G.step()

        self.adv.requires_grad_(True)
        self.optimizer_A.zero_grad()
        s_g = self.adv(z_org.detach())
        self.A_loss = self.criterion(s_g,sens.unsqueeze(1).float())
        self.A_loss.backward()
        self.optimizer_A.step()

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

class TaskNode(nn.Module):
    def __init__(self, args):
        super(TaskNode,self).__init__()

        nhid = args.num_hidden
        self.classifier = nn.Linear(nhid,1)
        self.args = args
        self.optimizer_C = torch.optim.Adam(self.classifier.parameters(), lr = args.lr2, weight_decay = args.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss = 0

    def forward(self,x):
        y = self.classifier(x)
        return y
    
    def optimize(self,z,labels,idx_train):
        self.train()
        self.optimizer_C.zero_grad()
        y = self.classifier(z)
        self.cls_loss = self.criterion(y[idx_train],labels[idx_train].unsqueeze(1).float())
        self.loss = self.cls_loss
        self.loss.backward(retain_graph=True)
        self.optimizer_C.step()