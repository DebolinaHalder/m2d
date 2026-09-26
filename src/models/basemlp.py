import torch.nn as nn
from models.MLPX import MLPX
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter







class Basemlp(nn.Module):

    def __init__(self, nfeat, args):
        super(Basemlp,self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        
        self.mlp = MLPX(nfeat,nhid,nhid)
        self.classifier = nn.Linear(nhid,args.num_classes)
        

        self.G_params = list(self.mlp.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(self.G_params, lr = args.lr)
        

        self.args = args
        
        self.criterion = nn.CrossEntropyLoss()
        
        #self.criterion = nn.BCELoss()
        self.G_loss = 0
        

    def forward(self,x):
        z = self.mlp(x)
        y = self.classifier(z)
        #y = torch.sigmoid(y)
        return z,y
    
    def optimize(self,x,labels,idx_train):
        self.train()
        self.optimizer_G.zero_grad()
        h = self.mlp(x)
        y = self.classifier(h)
        #y = torch.sigmoid(y)
        self.G_loss = self.criterion(y[idx_train].float(),labels[idx_train])
        self.G_loss.backward()
        self.optimizer_G.step()

