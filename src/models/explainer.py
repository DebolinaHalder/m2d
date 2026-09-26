import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
from models.MLPA import MLPA
import torch
import pyro
import dgl
import numpy as np
from torch import linalg as LA
import scipy as sp
import torch.nn.functional as F

def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat,args.hidden,args.dropout)
    elif args.model == "GAT":
        heads =  ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT_body(args.num_layers,nfeat,args.hidden,heads,args.dropout,args.attn_drop,args.negative_slope,args.residual)
    else:
        print("Model not implement")
        return

    return model

class Explainer(nn.Module):

    def __init__(self, nfeat, args,g,temperature = 1):
        super(Explainer,self).__init__()

        self.temperature = temperature
        self.GNN = get_model(nfeat,args)
        #self.classifier = nn.Linear(args.hidden,1)
        self.adj = MLPA(in_feats = args.hidden, dim_h = args.num_hidden, dim_z =nfeat)
        #self.G_params = (list(self.GNN.parameters()) + list(self.classifier.parameters()))
        self.G_params = (list(self.GNN.parameters()))
        self.optimizer_G = torch.optim.Adam(self.G_params, lr = args.lr,weight_decay=args.weight_decay)

        self.adj_params = list(self.adj.parameters())
        self.optimizer_adj = torch.optim.Adam(self.adj_params, lr = args.lr,weight_decay=args.weight_decay)
        
        self.args = args
        self.g = g
        #self.criterion_weighted = nn.BCEWithLogitsLoss()
        
        self.criterion_adj = nn.L1Loss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.G_loss = 0
        self.adj_loss = 0
        

    def forward(self,g,x):
        z = self.GNN(g,x)
        #y = self.classifier(z)
        return z,self.g
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
    def normalize_adj(self,adj):
        
        # normalize adj with A = D^{-1/2} @ A @ D^{-1/2}
        D_norm = torch.diag(torch.pow(adj.sum(1), -0.5))
        adj = D_norm @ adj @ D_norm
        return adj
    

    
    def optimize(self,x,rep):
        self.train()
        
        h = self.GNN(self.g,x)
        #y = self.classifier(h)
        adj_logits = self.adj(h)
        edge_probs = torch.sigmoid(adj_logits)
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, probs=edge_probs).rsample()
        
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        adj_sampled.fill_diagonal_(1)
        adj_sampled = self.normalize_adj(adj_sampled)
        adj_norm = adj_sampled.norm(p=1)
        src, dst = np.nonzero(adj_sampled.detach().numpy())
        self.g = dgl.graph((src, dst))
        adj = self.g.adj(scipy_fmt='coo')
        print(np.unique(adj.toarray(),return_counts=True))
        
        self.optimizer_adj.zero_grad()
         
        self.adj_loss = self.loss(h,rep)+self.args.beta*adj_norm
        #self.adj_loss = self.criterion(y[validation_set],output_org[validation_set].unsqueeze(1).float())+self.args.beta*adj_norm
        self.adj_loss.backward(retain_graph=True)
        self.optimizer_adj.step()
        #g = dgl.DGLGraph()
        #g.from_scipy_sparse_matrix(adj_sampled.detach().numpy())
        self.optimizer_G.zero_grad()
        h = self.GNN(self.g,x)
        #y = self.classifier(h)
        
        self.G_loss = self.loss(h,rep)   
        self.G_loss.backward(retain_graph=True)   
        self.optimizer_G.step()

        