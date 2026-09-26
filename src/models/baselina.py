import torch.nn as nn
from models.GCN import GCN,GCN_Body
from models.GAT import GAT,GAT_body
from models.SGC import SGC, SGC_Body
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter





def get_model(nfeat, args):
    if args.model == "SGC":
        print("here")
        model = SGC_Body(nfeat,args.num_hidden,k=2)
    elif args.model == "GCN":
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
        
        self.GNN = get_model(nfeat,args)
        self.classifier = nn.Linear(nhid,args.num_classes)
        
        self.G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        
        self.optimizer_G = torch.optim.Adam(self.G_params, lr = args.lr)
        

        self.args = args
        
        self.criterion = nn.CrossEntropyLoss()
        
        #self.criterion = nn.BCELoss()
        self.G_loss = 0
        

    def forward(self,adj,x):
        z = self.GNN(adj,x)
        y = self.classifier(z)
        #y = torch.sigmoid(y)
        return z,y
    
    def optimize(self,adj,x,labels,idx_train):
        self.train()
        self.optimizer_G.zero_grad()
        h = self.GNN(adj,x)
        y = self.classifier(h)
        #y = torch.sigmoid(y)
        self.cls_loss = self.criterion(y[idx_train].float(),labels[idx_train])
        self.G_loss = self.cls_loss
        self.G_loss.backward()
        self.optimizer_G.step()

